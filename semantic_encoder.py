import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def simple_tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text.lower())


def build_hash_features(texts: Iterable[str], feature_dim: int) -> torch.Tensor:
    texts = list(texts)
    vectors = np.zeros((len(texts), feature_dim), dtype=np.float32)
    for row, text in enumerate(texts):
        tokens = simple_tokenize(text) or [text.lower() or "[empty]"]
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            index = int(digest, 16) % feature_dim
            vectors[row, index] += 1.0
        norm = np.linalg.norm(vectors[row]) + 1e-12
        vectors[row] /= norm
    return torch.tensor(vectors, dtype=torch.float32)


def _resolve_dtype(precision: str, device: str):
    if not device.startswith("cuda"):
        return None
    precision = precision.lower()
    if precision == "float32":
        return None
    if precision == "float16":
        return torch.float16
    if precision == "bfloat16":
        return torch.bfloat16
    if precision == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    raise ValueError(f"Unsupported semantic precision: {precision}")


def _resolve_pooling(pooling: str, model_name: str, model) -> str:
    pooling = pooling.lower()
    if pooling != "auto":
        return pooling
    model_name_lower = model_name.lower()
    model_type = str(getattr(model.config, "model_type", "")).lower()
    is_decoder = bool(getattr(model.config, "is_decoder", False))
    decoder_markers = ("qwen", "llama", "mistral", "gemma", "gpt")
    if is_decoder or any(marker in model_type or marker in model_name_lower for marker in decoder_markers):
        return "last"
    if "bge" in model_name_lower:
        return "cls"
    return "mean"


def _pool_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor, pooling: str) -> torch.Tensor:
    if pooling == "cls":
        return hidden[:, 0]
    if pooling == "last":
        sequence_lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
        batch_indices = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[batch_indices, sequence_lengths]
    if pooling == "mean":
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    raise ValueError(f"Unsupported semantic pooling: {pooling}")


def _cache_path(
        cache_dir: str,
        cache_name: str,
        texts: List[str],
        backend: str,
        feature_dim: int,
        model_name: str,
        batch_size: int,
        max_length: int,
        pooling: str,
        normalize: bool,
        precision: str,
) -> Path:
    payload = {
        "texts": texts,
        "backend": backend,
        "feature_dim": feature_dim,
        "model_name": model_name,
        "batch_size": batch_size,
        "max_length": max_length,
        "pooling": pooling,
        "normalize": normalize,
        "precision": precision,
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name or backend)
    return Path(cache_dir) / f"{cache_name}_{backend}_{safe_model}_{digest}.pt"


def build_transformer_features(
        texts: Iterable[str],
        model_name: str,
        batch_size: int = 32,
        device: str = "cpu",
        max_length: int = 128,
        pooling: str = "auto",
        normalize: bool = True,
        precision: str = "auto",
        trust_remote_code: bool = True,
        local_files_only: bool = False,
) -> torch.Tensor:
    from transformers import AutoModel, AutoTokenizer

    texts = list(texts)
    print(
        {
            "stage": "semantic_transformer_load_start",
            "model": model_name,
            "texts": len(texts),
            "batch_size": batch_size,
            "device": device,
            "local_files_only": local_files_only,
        },
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = _resolve_dtype(precision, device)
    model_kwargs = {
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModel.from_pretrained(model_name, **model_kwargs).to(device)
    model.eval()
    resolved_pooling = _resolve_pooling(pooling, model_name, model)
    print(
        {
            "stage": "semantic_transformer_encode_start",
            "model": model_name,
            "texts": len(texts),
            "pooling": resolved_pooling,
            "precision": precision,
        },
        flush=True,
    )

    outputs = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(texts), batch_size), desc=f"semantic-{model_name}", mininterval=3.0):
            batch_texts = texts[start: start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device, non_blocking=True) for key, value in encoded.items()}
            model_output = model(**encoded)
            hidden = model_output.last_hidden_state
            pooled = _pool_hidden(hidden, encoded["attention_mask"], resolved_pooling)
            if normalize:
                pooled = F.normalize(pooled, p=2, dim=-1)
            outputs.append(pooled.float().cpu())
    features = torch.cat(outputs, dim=0)
    print(
        {
            "stage": "semantic_transformer_encode_done",
            "model": model_name,
            "texts": len(texts),
            "feature_shape": list(features.shape),
        },
        flush=True,
    )
    return features


def build_text_features(
        texts: Iterable[str],
        backend: str = "transformer",
        feature_dim: int = 256,
        transformer_name: str = "BAAI/bge-m3",
        transformer_batch_size: int = 32,
        device: str = "cpu",
        max_length: int = 128,
        pooling: str = "auto",
        normalize: bool = True,
        precision: str = "auto",
        trust_remote_code: bool = True,
        local_files_only: bool = False,
        cache_dir: str = "",
        cache_name: str = "semantic",
        cache_features: bool = True,
        force_rebuild_cache: bool = False,
) -> torch.Tensor:
    texts = list(texts)
    cache_file = None
    if cache_dir and cache_features:
        cache_file = _cache_path(
            cache_dir=cache_dir,
            cache_name=cache_name,
            texts=texts,
            backend=backend,
            feature_dim=feature_dim,
            model_name=transformer_name,
            batch_size=transformer_batch_size,
            max_length=max_length,
            pooling=pooling,
            normalize=normalize,
            precision=precision,
        )
        if cache_file.exists() and not force_rebuild_cache:
            print(
                {
                    "stage": "semantic_cache_hit",
                    "cache_name": cache_name,
                    "cache_file": str(cache_file),
                    "texts": len(texts),
                },
                flush=True,
            )
            return torch.load(cache_file, map_location="cpu")
        print(
            {
                "stage": "semantic_cache_miss",
                "cache_name": cache_name,
                "cache_file": str(cache_file),
                "texts": len(texts),
                "force_rebuild": bool(force_rebuild_cache),
            },
            flush=True,
        )

    if backend in {"none", "zero", "zeros"}:
        print({"stage": "semantic_zero_features", "cache_name": cache_name, "texts": len(texts)}, flush=True)
        features = torch.zeros((len(texts), feature_dim), dtype=torch.float32)
    elif backend == "transformer":
        features = build_transformer_features(
            texts=texts,
            model_name=transformer_name,
            batch_size=transformer_batch_size,
            device=device,
            max_length=max_length,
            pooling=pooling,
            normalize=normalize,
            precision=precision,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
    elif backend == "hash":
        print({"stage": "semantic_hash_features_start", "cache_name": cache_name, "texts": len(texts)}, flush=True)
        features = build_hash_features(texts=texts, feature_dim=feature_dim)
        print({"stage": "semantic_hash_features_done", "cache_name": cache_name, "texts": len(texts)}, flush=True)
    else:
        raise ValueError(f"Unsupported semantic backend: {backend}")

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        print(
            {
                "stage": "semantic_cache_save_start",
                "cache_name": cache_name,
                "cache_file": str(cache_file),
                "feature_shape": list(features.shape),
            },
            flush=True,
        )
        torch.save(features.cpu(), cache_file)
        print({"stage": "semantic_cache_save_done", "cache_name": cache_name}, flush=True)
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic feature builder")
    parser.add_argument("--text", nargs="+", default=["Temporal knowledge graph"])
    parser.add_argument("--backend", type=str, default="transformer", choices=["none", "hash", "transformer"])
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--transformer-name", type=str, default="BAAI/bge-m3")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pooling", type=str, default="auto", choices=["auto", "mean", "cls", "last"])
    args = parser.parse_args()

    features = build_text_features(
        texts=args.text,
        backend=args.backend,
        feature_dim=args.feature_dim,
        transformer_name=args.transformer_name,
        transformer_batch_size=args.batch_size,
        device=args.device,
        pooling=args.pooling,
    )
    print({"shape": list(features.shape), "first_vector_prefix": features[0, :8].tolist()})


if __name__ == "__main__":
    main()
