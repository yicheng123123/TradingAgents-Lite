"""Local model inference, persistent embedding cache and Qdrant primitives."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3
from typing import Protocol

from qdrant_client import QdrantClient, models

from .documents import digest


class Encoder(Protocol):
    metadata: dict
    def encode(self, texts: list[str], *, query: bool = False) -> list[list[float]]: ...


class OnnxModel:
    def __init__(self, root: Path, repository: str, revision: str, *, device: str, threads: int, allow_download: bool = True):
        from hashlib import file_digest
        from shutil import copyfileobj
        from tempfile import NamedTemporaryFile
        from urllib.request import urlopen

        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = root / "models" / revision
        model_file = model_dir / "onnx/model_quantized.onnx"
        tokenizer_file = model_dir / "tokenizer.json"
        hashes = {}
        for path in (model_file, tokenizer_file):
            if not path.is_file():
                if not allow_download:
                    raise FileNotFoundError("Local model files are required")
                path.parent.mkdir(parents=True, exist_ok=True)
                relative = path.relative_to(model_dir).as_posix()
                url = f"https://huggingface.co/{repository}/resolve/{revision}/{relative}"
                with NamedTemporaryFile(dir=path.parent, delete=False) as temp:
                    try:
                        with urlopen(url, timeout=60) as response:
                            copyfileobj(response, temp)
                        temp.flush()
                        Path(temp.name).replace(path)
                    finally:
                        Path(temp.name).unlink(missing_ok=True)
            with path.open("rb") as stream:
                hashes[path.name] = file_digest(stream, "sha256").hexdigest()
        providers = ["CPUExecutionProvider"]
        if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        try:
            self.session = ort.InferenceSession(str(model_file), sess_options=options, providers=providers)
        except (RuntimeError, ValueError):
            if len(providers) == 1:
                raise
            self.session = ort.InferenceSession(str(model_file), sess_options=options,
                                                providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(str(tokenizer_file))
        self.tokenizer.enable_truncation(max_length=512)
        self.tokenizer.enable_padding()
        self.metadata = {"name": repository, "revision": revision, "weight_hashes": hashes,
                         "device": self.session.get_providers()[0], "max_model_tokens": 512}

    def infer(self, examples):
        import numpy as np

        tokens = self.tokenizer.encode_batch(examples)
        fields = {"input_ids": "ids", "attention_mask": "attention_mask", "token_type_ids": "type_ids"}
        inputs = {item.name: np.asarray([getattr(t, fields[item.name]) for t in tokens], dtype=np.int64)
                  for item in self.session.get_inputs()}
        return self.session.run(None, inputs)[0]


class LocalEmbedding:
    MODEL = "Xenova/bge-small-en-v1.5"
    REVISION = "ea104dacec62c0de699686887e3f920caeb4f3e3"

    def __init__(self, root: Path, *, device: str = "cpu", threads: int = 4, batch: int = 32,
                 allow_download: bool = True):
        self.engine = OnnxModel(root, self.MODEL, self.REVISION, device=device, threads=threads, allow_download=allow_download)
        self.metadata = {**self.engine.metadata, "base_model": "BAAI/bge-small-en-v1.5",
                         "dimension": 384, "normalization": "CLS then L2"}
        self.cache_identity = {k: v for k, v in self.metadata.items() if k != "device"}
        self.batch = batch
        self.path = root / "embeddings.sqlite"
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

    def encode(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        import numpy as np

        if not texts:
            return []
        keys = [digest(json.dumps([self.cache_identity, query, text], sort_keys=True).encode()) for text in texts]
        vectors = {}
        with closing(sqlite3.connect(self.path, timeout=30)) as db, db:
            missing = {}
            for key, text in zip(keys, texts, strict=True):
                if key in vectors or key in missing:
                    continue
                row = db.execute("SELECT value FROM vectors WHERE key = ?", (key,)).fetchone()
                if row:
                    vectors[key] = json.loads(row[0])
                else:
                    missing[key] = text
            pairs = list(missing.items())
            prefix = "Represent this sentence for searching relevant passages: " if query else ""
            for start in range(0, len(pairs), self.batch):
                batch = pairs[start:start + self.batch]
                hidden = self.engine.infer([prefix + text for _, text in batch])[:, 0, :]
                normalized = hidden / np.maximum(np.linalg.norm(hidden, axis=1, keepdims=True), 1e-12)
                for (key, _), vector in zip(batch, normalized, strict=True):
                    vectors[key] = vector.tolist()
            db.executemany("INSERT OR IGNORE INTO vectors VALUES (?, ?)",
                           [(key, json.dumps(vectors[key])) for key in missing])
        return [vectors[key] for key in keys]


class LocalReranker:
    MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
    REVISION = "a09144355adeed5f58c8ed011d209bf8ee5a1fec"

    def __init__(self, root: Path, *, allow_download: bool = True):
        self.engine = OnnxModel(root, self.MODEL, self.REVISION, device="cpu", threads=4,
                                allow_download=allow_download)
        self.metadata = self.engine.metadata

    def score(self, query: str, texts: list[str]) -> list[float]:
        scores = []
        for start in range(0, len(texts), 8):
            values = self.engine.infer([(query, text) for text in texts[start:start + 8]])
            scores.extend(float(value) for value in values.reshape(-1))
        return scores


def match(key: str, value: str) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


def date_limit(key: str, timestamp: float, *, minimum: float | None = None) -> models.FieldCondition:
    return models.FieldCondition(key=key, range=models.Range(gte=minimum, lte=timestamp))


class VectorStore:
    def __init__(self, path: Path, encoder: Encoder | None, name: str):
        self.encoder = encoder
        self.name = name
        self.revision = 0
        self.client = QdrantClient(path=str(path))
        identity = {k: v for k, v in encoder.metadata.items() if k != "device"} if encoder else {}
        marker = path / f"{name}-encoder.json"
        try:
            if self.client.collection_exists(name):
                if json.loads(marker.read_text()) != identity:
                    raise ValueError("Index embedding identity changed; build a new index")
            else:
                config = (models.VectorParams(size=identity["dimension"], distance=models.Distance.COSINE)
                          if encoder else {})
                self.client.create_collection(name, vectors_config=config)
                marker.write_text(json.dumps(identity, indent=2) + "\n")
        except Exception:
            self.client.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.client.close()

    def put(self, rows: list[dict], *, id_key: str, text_key: str) -> None:
        if rows:
            self.revision += 1
        for start in range(0, len(rows), 128):
            batch = rows[start:start + 128]
            vectors = (self.encoder.encode([row[text_key] for row in batch])
                       if self.encoder is not None else [{} for _ in batch])
            self.client.upsert(self.name, [
                models.PointStruct(id=row[id_key], vector=vector, payload=row)
                for row, vector in zip(batch, vectors, strict=True)], wait=True)

    def scan(self, filters: models.Filter | None = None):
        offset = None
        while True:
            points, offset = self.client.scroll(self.name, scroll_filter=filters, limit=256, offset=offset)
            yield from points
            if offset is None:
                break

    def search(self, query: str, filters: models.Filter, limit: int):
        if self.encoder is None:
            raise ValueError("This index only supports lexical retrieval")
        return self.client.query_points(self.name, query=self.encoder.encode([query], query=True)[0],
                                        query_filter=filters, limit=limit, with_payload=True).points
