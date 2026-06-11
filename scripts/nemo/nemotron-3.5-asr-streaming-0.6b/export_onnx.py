#!/usr/bin/env python3
# Copyright      2026  Xiaomi Corp.        (authors: Fangjun Kuang)
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import nemo.collections.asr as nemo_asr
import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic


def add_meta_data(filename: str, meta_data: Dict[str, str]):
    """Add meta data to an ONNX model. It is changed in-place."""
    model = onnx.load(filename)

    while len(model.metadata_props):
        model.metadata_props.pop()

    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    external_filename = filename.split(".onnx")[0]
    onnx.save(
        model,
        filename,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_filename + ".data",
    )


def _to_plain_container(obj: Any) -> Any:
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf

        if isinstance(obj, (DictConfig, ListConfig)):
            return OmegaConf.to_container(obj, resolve=True)
    except ImportError:
        pass

    return obj


def _is_prompt_dictionary(obj: Any) -> bool:
    obj = _to_plain_container(obj)
    if not isinstance(obj, dict):
        return False

    try:
        normalized = {str(k): int(v) for k, v in obj.items()}
    except (TypeError, ValueError):
        return False

    return normalized.get("auto") == 101 and "en" in normalized and "ja" in normalized


def _normalize_prompt_dictionary(obj: Any) -> Dict[str, int]:
    obj = _to_plain_container(obj)
    assert isinstance(obj, dict), type(obj)
    return {str(k): int(v) for k, v in obj.items()}


def _walk_values(obj: Any, max_depth: int = 8) -> Iterable[Any]:
    seen = set()

    def visit(value: Any, depth: int):
        value = _to_plain_container(value)
        obj_id = id(value)
        if obj_id in seen or depth > max_depth:
            return

        seen.add(obj_id)
        yield value

        if isinstance(value, dict):
            for k, v in value.items():
                if "prompt" in str(k).lower() or isinstance(v, (dict, list, tuple)):
                    yield from visit(v, depth + 1)
        elif isinstance(value, (list, tuple)):
            for v in value:
                yield from visit(v, depth + 1)

    yield from visit(obj, 0)


def get_prompt_dictionary(asr_model) -> Dict[str, int]:
    """Return the model's language prompt dictionary from NeMo artifacts."""
    roots = [
        getattr(asr_model, "cfg", None),
        getattr(asr_model, "encoder", None),
        asr_model,
    ]

    attr_names = [
        "prompt_dictionary",
        "prompt_dict",
        "lang_prompt_dictionary",
        "language_prompt_dictionary",
    ]

    for root in roots:
        if root is None:
            continue
        for name in attr_names:
            if hasattr(root, name):
                value = getattr(root, name)
                if _is_prompt_dictionary(value):
                    ans = _normalize_prompt_dictionary(value)
                    assert ans["auto"] == 101, ans["auto"]
                    return ans

    for root in roots:
        if root is None:
            continue
        for value in _walk_values(root):
            if _is_prompt_dictionary(value):
                ans = _normalize_prompt_dictionary(value)
                assert ans["auto"] == 101, ans["auto"]
                return ans

    raise RuntimeError("Could not find prompt_dictionary in the NeMo model")


def _find_sentencepiece_processor(obj: Any, max_depth: int = 5) -> Optional[Any]:
    seen = set()

    def is_sentencepiece_processor(value: Any) -> bool:
        return callable(getattr(value, "get_piece_size", None)) and callable(
            getattr(value, "id_to_piece", None)
        )

    def visit(value: Any, depth: int) -> Optional[Any]:
        if value is None or depth > max_depth:
            return None

        if is_sentencepiece_processor(value):
            return value

        obj_id = id(value)
        if obj_id in seen:
            return None
        seen.add(obj_id)

        for name in ["tokenizer", "sp_model", "model", "processor"]:
            if hasattr(value, name):
                found = visit(getattr(value, name), depth + 1)
                if found is not None:
                    return found

        if isinstance(value, dict):
            for v in value.values():
                found = visit(v, depth + 1)
                if found is not None:
                    return found

        return None

    return visit(obj, 0)


def save_tokens(asr_model, filename: str = "tokens.txt") -> int:
    sp = _find_sentencepiece_processor(getattr(asr_model, "tokenizer", None))
    if sp is None:
        raise RuntimeError("Could not find the SentencePiece tokenizer in the model")

    vocab_size = sp.get_piece_size()
    with open(filename, "w", encoding="utf-8") as f:
        for i in range(vocab_size):
            f.write(f"{sp.id_to_piece(i)} {i}\n")
        f.write(f"<blk> {vocab_size}\n")

    print(f"Saved {filename}")
    return vocab_size


def remove_export_scratch_files():
    patterns = [
        "Constant_*_attr__value",
        "onnx__MatMul_*",
        "layers.*.conv*",
        "pre_encode.conv.*.weight",
    ]
    for pattern in patterns:
        for p in Path(".").glob(pattern):
            p.unlink()


@torch.no_grad()
def main():
    model_name = "nvidia/nemotron-3.5-asr-streaming-0.6b"

    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)

    vocab_size = save_tokens(asr_model)
    if vocab_size != asr_model.decoder.vocab_size:
        raise ValueError(
            f"SentencePiece vocab size {vocab_size} != decoder vocab size "
            f"{asr_model.decoder.vocab_size}"
        )

    prompt_dictionary = get_prompt_dictionary(asr_model)
    auto_prompt_id = prompt_dictionary["auto"]
    if auto_prompt_id != 101:
        raise ValueError(f"Expected auto prompt id 101, got {auto_prompt_id}")

    asr_model.eval()

    assert asr_model.encoder.streaming_cfg is not None
    print("streaming_cfg", asr_model.encoder.streaming_cfg)
    print("prompt_dictionary", prompt_dictionary)

    chunk_size_ms_list = [80, 160, 560, 1120]
    for ms in chunk_size_ms_list:
        chunk_size = ms // 80 - 1
        print("chunk_size", chunk_size)
        asr_model.encoder.set_default_att_context_size([70, chunk_size])

        print("streaming_cfg", asr_model.encoder.streaming_cfg)
        print("att_context_size", asr_model.encoder.att_context_size)
        print(
            "pre_encode_cache_size",
            asr_model.encoder.streaming_cfg.pre_encode_cache_size,
        )

        if isinstance(asr_model.encoder.streaming_cfg.pre_encode_cache_size, list):
            pre_encode_cache_size = (
                asr_model.encoder.streaming_cfg.pre_encode_cache_size[1]
            )
        else:
            pre_encode_cache_size = (
                asr_model.encoder.streaming_cfg.pre_encode_cache_size
            )

        if isinstance(asr_model.encoder.streaming_cfg.chunk_size, list):
            chunk_size = asr_model.encoder.streaming_cfg.chunk_size[1]
        else:
            chunk_size = asr_model.encoder.streaming_cfg.chunk_size

        window_size = chunk_size + pre_encode_cache_size

        print("chunk_size", chunk_size)
        print("pre_encode_cache_size", pre_encode_cache_size)
        print("window_size", window_size)

        chunk_shift = chunk_size

        # cache_last_channel: (batch_size, dim1, dim2, dim3)
        cache_last_channel_dim1 = len(asr_model.encoder.layers)
        cache_last_channel_dim2 = (
            asr_model.encoder.streaming_cfg.last_channel_cache_size
        )
        cache_last_channel_dim3 = asr_model.encoder.d_model

        # cache_last_time: (batch_size, dim1, dim2, dim3)
        cache_last_time_dim1 = len(asr_model.encoder.layers)
        cache_last_time_dim2 = asr_model.encoder.d_model
        cache_last_time_dim3 = asr_model.encoder.conv_context_size[0]

        asr_model.set_export_config({"cache_support": True})

        asr_model.encoder.export("encoder.onnx")
        asr_model.decoder.export("decoder.onnx")
        asr_model.joint.export("joiner.onnx")

        normalize_type = asr_model.cfg.preprocessor.normalize
        if normalize_type == "NA":
            normalize_type = ""

        meta_data = {
            "vocab_size": asr_model.decoder.vocab_size,
            "window_size": window_size,
            "chunk_size_ms": ms,
            "chunk_shift": chunk_shift,
            "normalize_type": normalize_type,
            "cache_last_channel_dim1": cache_last_channel_dim1,
            "cache_last_channel_dim2": cache_last_channel_dim2,
            "cache_last_channel_dim3": cache_last_channel_dim3,
            "cache_last_time_dim1": cache_last_time_dim1,
            "cache_last_time_dim2": cache_last_time_dim2,
            "cache_last_time_dim3": cache_last_time_dim3,
            "pred_rnn_layers": asr_model.decoder.pred_rnn_layers,
            "pred_hidden": asr_model.decoder.pred_hidden,
            "subsampling_factor": 8,
            "feat_dim": 128,
            "model_type": type(asr_model).__name__,
            "version": "1",
            "model_author": "NeMo",
            "url": f"https://huggingface.co/{model_name}",
            "comment": "Only the transducer branch is exported",
            "prompt_dictionary": json.dumps(prompt_dictionary, sort_keys=True),
            "auto_prompt_id": auto_prompt_id,
        }
        print("meta_data", meta_data)
        add_meta_data("encoder.onnx", meta_data)

        for m in ["encoder", "decoder", "joiner"]:
            quantize_dynamic(
                model_input=f"{m}.onnx",
                model_output=f"{m}.int8.onnx",
                weight_type=QuantType.QUInt8,
            )

        Path(str(ms)).mkdir(exist_ok=True)
        for suffix in ["onnx", "data"]:
            for p in Path(".").glob(f"*.{suffix}"):
                p.rename(Path(str(ms)) / p.name)

        print(meta_data)
        print(f"Saved exported models to {ms}")
        remove_export_scratch_files()
        os.system(f"ls -lh {ms}")


if __name__ == "__main__":
    main()
