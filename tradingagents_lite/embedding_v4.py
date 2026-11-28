"""Pinned dense-only BGE-M3 encoder for the isolated A1-V4 experiment."""
from __future__ import annotations

from importlib.metadata import version
from math import isfinite
from pathlib import Path
from time import perf_counter

MODEL_ID = "BAAI/bge-m3"
MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
DIMENSION = 1024
MAX_LENGTH = 512
BATCH_LIMIT = 24
PADDED_TOKEN_LIMIT = 8192


def length_batches(lengths: list[int], batch_limit: int = BATCH_LIMIT,
                   token_limit: int = PADDED_TOKEN_LIMIT) -> list[list[int]]:
    """Pack short sequences together while bounding padded tokens per batch."""
    if batch_limit < 1 or token_limit < 1:
        raise ValueError("Batch limits must be positive")
    if any(length < 1 for length in lengths):
        raise ValueError("Token lengths must be positive")
    ordered = sorted(range(len(lengths)), key=lambda i: (lengths[i], i))
    result, batch, maximum = [], [], 0
    for index in ordered:
        length = lengths[index]
        if batch and (len(batch) >= batch_limit or (len(batch) + 1) * max(maximum, length) > token_limit):
            result.append(batch)
            batch, maximum = [], 0
        batch.append(index)
        maximum = max(maximum, length)
    if batch:
        result.append(batch)
    return result


class BGEM3Encoder:
    """BGE-M3 dense encoder; query and document paths use identical model text."""

    metadata: dict

    def __init__(self, model_root: Path, *, device: str = "cuda", max_length: int = MAX_LENGTH,
                 batch_limit: int = BATCH_LIMIT, token_limit: int = PADDED_TOKEN_LIMIT):
        import torch
        from transformers import AutoModel, AutoTokenizer

        if device not in {"cuda", "cpu"}:
            raise ValueError("device must be cuda or cpu")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA unavailable; choose --device cpu explicitly")
        if not 16 <= max_length <= 8192:
            raise ValueError("max_length must be in [16, 8192]")
        if device == "cpu":
            torch.set_num_threads(4)
        self.torch = torch
        self.device = torch.device(device)
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        self.max_length = max_length
        self.batch_limit = batch_limit
        self.token_limit = token_limit
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_root, local_files_only=True, trust_remote_code=False,
        )
        self.model = AutoModel.from_pretrained(
            model_root, local_files_only=True, trust_remote_code=False,
            use_safetensors=False, dtype=self.dtype, attn_implementation="sdpa",
        ).to(self.device).eval()
        actual_dim = int(getattr(self.model.config, "hidden_size", 0))
        if actual_dim != DIMENSION:
            raise ValueError(f"Unexpected BGE-M3 hidden size: {actual_dim}")
        self.metadata = {
            "name": MODEL_ID,
            "revision": MODEL_REVISION,
            "dimension": DIMENSION,
            "pooling": "last_hidden_state[:, 0] (CLS)",
            "normalization": "L2",
            "query_instruction": None,
            "dense_only": True,
            "sparse": False,
            "colbert": False,
            "input_limit": max_length,
            "batch_limit": batch_limit,
            "padded_token_limit": token_limit,
            "device": device,
            "dtype": str(self.dtype),
            "torch": version("torch"),
            "transformers": version("transformers"),
            "cuda": torch.version.cuda if device == "cuda" else None,
            "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        }

    def token_lengths(self, texts: list[str], *, truncate: bool = False) -> list[int]:
        if not texts:
            return []
        return [len(ids) for ids in self.tokenizer(
            texts, add_special_tokens=True, truncation=truncate,
            max_length=self.max_length if truncate else None,
            padding=False, return_attention_mask=False,
        )["input_ids"]]

    def _encode_batch(self, encodings: list[dict], indices: list[int], vectors: list | None) -> int:
        torch = self.torch
        inputs = self.tokenizer.pad([encodings[i] for i in indices], padding=True, return_tensors="pt")
        device_inputs = {key: value.to(self.device) for key, value in inputs.items()}
        try:
            with torch.inference_mode():
                hidden = self.model(**device_inputs, return_dict=True).last_hidden_state
                dense = torch.nn.functional.normalize(hidden[:, 0, :].float(), p=2, dim=-1)
                values = dense.cpu().tolist()
        except torch.cuda.OutOfMemoryError:
            if self.device.type != "cuda" or len(indices) <= 1:
                raise
            del device_inputs, inputs
            torch.cuda.empty_cache()
            middle = len(indices) // 2
            return 1 + self._encode_batch(encodings, indices[:middle], vectors) + self._encode_batch(
                encodings, indices[middle:], vectors)
        if len(values) != len(indices) or any(len(row) != DIMENSION for row in values):
            raise ValueError("BGE-M3 returned an invalid vector shape")
        for index, row in zip(indices, values, strict=True):
            if any(not isfinite(value) for value in row):
                raise ValueError("BGE-M3 returned NaN or Inf")
            norm = sum(value * value for value in row) ** 0.5
            if abs(norm - 1.0) > 2e-4:
                raise ValueError(f"BGE-M3 vector is not L2 normalized: {norm}")
            vectors[index] = row
        return 0

    def encode_with_stats(self, texts: list[str]) -> tuple[list[list[float]], dict]:
        if not texts:
            return [], {"milliseconds": 0.0, "tokens": 0, "peak_allocated_bytes": 0,
                        "peak_reserved_bytes": 0, "oom_batch_splits": 0}
        torch = self.torch
        on_gpu = self.device.type == "cuda"
        if on_gpu:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = perf_counter()
        encodings = self.tokenizer(
            texts, add_special_tokens=True, truncation=True, max_length=self.max_length,
            padding=False, return_attention_mask=True,
        )
        lengths = [len(ids) for ids in encodings["input_ids"]]
        examples = [{key: values[i] for key, values in encodings.items()} for i in range(len(texts))]
        vectors: list[list[float] | None] = [None] * len(texts)
        splits = 0
        for indices in length_batches(lengths, self.batch_limit, self.token_limit):
            splits += self._encode_batch(examples, indices, vectors)
        if on_gpu:
            torch.cuda.synchronize()
        if any(vector is None for vector in vectors):
            raise ValueError("BGE-M3 omitted an embedding")
        milliseconds = (perf_counter() - started) * 1000
        result = [vector for vector in vectors if vector is not None]
        return result, {
            "milliseconds": milliseconds,
            "tokens": sum(lengths),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated() if on_gpu else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved() if on_gpu else None,
            "oom_batch_splits": splits,
        }

    def encode(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        # BGE-M3's official dense path uses the same input for query and passage.
        del query
        return self.encode_with_stats(texts)[0]
