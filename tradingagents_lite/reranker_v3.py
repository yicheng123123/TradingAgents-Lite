"""Local fixed-revision cross encoders for the A1-V3 experiment."""
from importlib.metadata import version
from math import isfinite
from time import perf_counter

from .retrieval_v2 import tree_hashes
from .vectors import LocalReranker

BGE_ID = "BAAI/bge-reranker-v2-m3"
BGE_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
BGE_WEIGHTS_SHA256 = "d9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286"
MAX_LENGTH = 512
BATCH_LIMIT = 24
PADDED_TOKEN_LIMIT = 8192


def batches(lengths, batch_limit, token_limit):
    ordered = sorted(range(len(lengths)), key=lambda i: (lengths[i], i))
    batch = []
    for i in ordered:
        if batch and (len(batch) >= batch_limit or (len(batch) + 1) * lengths[i] > token_limit):
            yield batch
            batch = []
        batch.append(i)
    if batch:
        yield batch


class BGEReranker:
    def __init__(self, model_root, device):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA unavailable. Choose --device cpu explicitly.")
        files = tree_hashes(model_root)
        if files.get("model.safetensors") != BGE_WEIGHTS_SHA256:
            raise ValueError("BGE weights do not match the pinned official SHA256")
        self.torch = torch
        self.device = torch.device(device)
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        if device == "cpu":
            torch.set_num_threads(4)
        self.tokenizer = AutoTokenizer.from_pretrained(model_root, local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_root, local_files_only=True, trust_remote_code=False,
            use_safetensors=True, dtype=self.dtype, attn_implementation="sdpa",
        ).to(self.device).eval()
        self.metadata = {
            "name": BGE_ID, "revision": BGE_REVISION, "files_sha256": files,
            "input_limit": MAX_LENGTH, "truncation": "only_second", "batch_limit": BATCH_LIMIT,
            "padded_token_limit": PADDED_TOKEN_LIMIT, "device": device, "dtype": str(self.dtype),
            "torch": version("torch"), "transformers": version("transformers"),
            "cuda": torch.version.cuda if device == "cuda" else None,
            "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        }

    def _forward(self, encoded, indices, scores):
        torch = self.torch
        tensors = self.tokenizer.pad([{key: values[i] for key, values in encoded.items()} for i in indices],
                                     padding=True, return_tensors="pt")
        device_inputs = {key: value.to(self.device) for key, value in tensors.items()}
        try:
            with torch.inference_mode():
                logits = self.model(**device_inputs).logits.reshape(-1).float().cpu().tolist()
        except torch.cuda.OutOfMemoryError:
            if len(indices) == 1:
                raise
            retry = True
        else:
            retry = False
        if retry:
            del device_inputs
            torch.cuda.empty_cache()
            half = len(indices) // 2
            return 1 + self._forward(encoded, indices[:half], scores) + self._forward(encoded, indices[half:], scores)
        if len(logits) != len(indices) or any(not isfinite(value) for value in logits):
            raise ValueError("Nonfinite or incomplete BGE scores")
        for i, value in zip(indices, logits, strict=True):
            scores[i] = value
        return 0

    def score(self, query, texts):
        torch = self.torch
        on_gpu = self.device.type == "cuda"
        if on_gpu:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = perf_counter()
        scores = [0.] * len(texts)
        retries = 0
        if texts:
            encoded = self.tokenizer([query] * len(texts), texts, truncation="only_second", max_length=MAX_LENGTH)
            for indices in batches([len(ids) for ids in encoded["input_ids"]], BATCH_LIMIT, PADDED_TOKEN_LIMIT):
                retries += self._forward(encoded, indices, scores)
        if on_gpu:
            torch.cuda.synchronize()
        return scores, {
            "milliseconds": (perf_counter() - started) * 1000,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated() if on_gpu else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved() if on_gpu else None,
            "oom_batch_splits": retries,
        }


class MiniLMReranker:
    def __init__(self, model_root):
        self.encoder = LocalReranker(model_root, allow_download=False)
        self.metadata = {**self.encoder.metadata, "batch_limit": 8, "input_limit": 512,
                         "runtime": version("onnxruntime"), "dtype": "quantized ONNX", "device": "cpu"}

    def score(self, query, texts):
        started = perf_counter()
        scores = self.encoder.score(query, texts)
        if len(scores) != len(texts) or any(not isfinite(value) for value in scores):
            raise ValueError("Nonfinite or incomplete MiniLM scores")
        return scores, {"milliseconds": (perf_counter() - started) * 1000, "peak_allocated_bytes": None,
                        "peak_reserved_bytes": None, "oom_batch_splits": 0}

