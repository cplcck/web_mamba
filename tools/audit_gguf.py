#!/usr/bin/env python3
"""Exact independent Mamba-1 artifact audit, not a numerical inference PASS."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import TYPE_CHECKING, Final, Literal
from typing_extensions import assert_never

import numpy as np
from safetensors import safe_open
import torch
from transformers import AutoTokenizer, MambaConfig, MambaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import prepare_model as provenance

SOURCE: Final = Path("/home/cplcck/llama.cpp-ssm")
REVISION: Final = "1e76775f628fbf1350fbe4dbb3d971ba64af25a1"
GGUF_SHA: Final = "af897eef0adfda25444be8fd2e6cbe3e858a107f7926625f7fb71d37f51bcaae"
if TYPE_CHECKING:
    from gguf import GGUFReader


@dataclass(frozen=True, slots=True)
class AuditError(Exception):
    field: str
    def __str__(self) -> str:
        return self.field


def require(condition: bool, field: str) -> None:
    if not condition:
        raise AuditError(field)


@dataclass(frozen=True, slots=True)
class TensorAudit:
    sourceName: str
    targetName: str
    sourceShape: tuple[int, ...]
    logicalShape: tuple[int, ...]
    nativeShape: tuple[int, ...]
    sourceSha256: str
    expectedSha256: str
    targetSha256: str
    transformation: str
    payloadOffset: int
    payloadBytes: int
    sourceDtype: str = "F32"
    targetDtype: str = "F32"


@dataclass(frozen=True, slots=True)
class MetadataAudit:
    name: str
    types: tuple[str, ...]
    sha256: str
    count: int


@dataclass(frozen=True, slots=True)
class ArtifactAudit:
    sourceTensorCount: int
    targetTensorCount: int
    blockCount: int
    tensors: tuple[TensorAudit, ...]
    metadata: tuple[MetadataAudit, ...]
    outputBinding: str
    vocabularySize: int
    tokenizerVocabularySize: int
    paddedTokenIds: tuple[int, ...]
    uniquePayloadBytes: int


def vocabulary_size(config: MambaConfig) -> int:
    multiple = getattr(config, "pad_vocab_size_multiple", 8)
    require(type(multiple) is int and multiple > 0, "config.pad_vocab_size_multiple")
    return (config.vocab_size + multiple - 1) // multiple * multiple


def audit_pair(snapshot: Path, target: Path) -> ArtifactAudit:
    """Audit real files, including tiny same-schema fixtures, without conversion."""
    provenance.source_identity(SOURCE)  # Reject product patches before importing its reader.
    sys.path.insert(0, str(SOURCE / "gguf-py"))
    import gguf
    require(Path(gguf.__file__).resolve().parent == SOURCE / "gguf-py/gguf", "reader.path")
    reader = gguf.GGUFReader(target)
    torch.set_num_threads(1)
    config = MambaConfig.from_pretrained(snapshot, local_files_only=True)
    require(config.model_type == "mamba" and config.intermediate_size == 2 * config.hidden_size,
            "config.architecture")
    h, i, s, k, r = (config.hidden_size, config.intermediate_size, config.state_size,
                      config.conv_kernel, config.time_step_rank)
    require(all(type(n) is int and n > 0 for n in (h, i, s, k, r, config.num_hidden_layers, config.vocab_size)),
            "config.dimensions")
    require(config.use_bias is False and config.use_conv_bias is True, "config.bias")
    padded = vocabulary_size(config)
    require(sorted(snapshot.glob("*.safetensors")) == [snapshot / "model.safetensors"], "source.files")
    specs: dict[str, tuple[str, tuple[int, ...], Literal["identity", "squeeze-axis-1", "negative-exp-f32"]]] = {
        "backbone.embeddings.weight": ("token_embd.weight", (padded, h), "identity"),
        "backbone.norm_f.weight": ("output_norm.weight", (h,), "identity"),
    }
    # Exhaustive names, not suffix recognition. Shapes are HF logical order.
    for block in range(config.num_hidden_layers):
        for suffix, name, shape, operation in (
            ("norm.weight", "attn_norm.weight", (h,), "identity"),
            ("mixer.in_proj.weight", "ssm_in.weight", (2 * i, h), "identity"),
            ("mixer.conv1d.weight", "ssm_conv1d.weight", (i, 1, k), "squeeze-axis-1"),
            ("mixer.conv1d.bias", "ssm_conv1d.bias", (i,), "identity"),
            ("mixer.x_proj.weight", "ssm_x.weight", (r + 2 * s, i), "identity"),
            ("mixer.dt_proj.weight", "ssm_dt.weight", (i, r), "identity"),
            ("mixer.dt_proj.bias", "ssm_dt.bias", (i,), "identity"),
            ("mixer.A_log", "ssm_a", (i, s), "negative-exp-f32"),
            ("mixer.D", "ssm_d", (i,), "identity"),
            ("mixer.out_proj.weight", "ssm_out.weight", (h, i), "identity"),
        ):
            specs[f"backbone.layers.{block}.{suffix}"] = (f"blk.{block}.{name}", shape, operation)
    targets = {tensor.name: tensor for tensor in reader.tensors}
    # Pinned reader rejects duplicate names and derives numel/nbytes from shape/type.
    records: list[TensorAudit] = []
    with safe_open(snapshot / "model.safetensors", framework="pt", device="cpu") as source:
        source_names = set(source.keys())
        for name in sorted((source_names - {"lm_head.weight"}) ^ specs.keys()):
            raise AuditError(f"source.{name}.presence")
        output_binding = "explicit-output"
        if "lm_head.weight" in source_names:
            specs["lm_head.weight"] = ("output.weight", (padded, h), "identity")
            if "output.weight" not in targets:
                require(torch.equal(source.get_tensor("lm_head.weight"), source.get_tensor("backbone.embeddings.weight")),
                        "source.lm_head.weight.tiedEquality")
                specs["lm_head.weight"] = ("token_embd.weight", (padded, h), "identity")
                output_binding = "converter-omitted-equal-output"
        else:
            require(config.tie_word_embeddings is True, "source.lm_head.weight.tiedContract")
            with torch.device("meta"):
                model = MambaForCausalLM(config)
            require(model.lm_head.weight is model.backbone.embeddings.weight, "source.lm_head.weight.tiedContract")
            output_binding = "hf-serialized-tied-omission"
        for name in sorted(targets.keys() ^ {spec[0] for spec in specs.values()}):
            raise AuditError(f"target.{name}.presence")
        for name, (mapped, shape, operation) in sorted(specs.items()):
            tensor = source.get_tensor(name)
            require(tensor.dtype == torch.float32, f"source.{name}.dtype")
            require(tuple(tensor.shape) == shape, f"source.{name}.shape")
            require(bool(torch.isfinite(tensor).all()), f"source.{name}.finite")
            match operation:
                case "identity":
                    expected = tensor
                case "negative-exp-f32":
                    expected = -torch.exp(tensor)
                case "squeeze-axis-1":
                    expected = tensor[:, 0, :]
                case unreachable:
                    assert_never(unreachable)
            actual = targets[mapped]
            require(actual.tensor_type == gguf.GGMLQuantizationType.F32, f"target.{mapped}.dtype")
            require(tuple(actual.data.shape) == tuple(expected.shape), f"target.{mapped}.logicalShape")
            require(tuple(actual.shape) == tuple(reversed(expected.shape)), f"target.{mapped}.nativeShape")
            source_hash = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
            expected_hash = hashlib.sha256(expected.numpy().tobytes()).hexdigest()
            target_hash = hashlib.sha256(actual.data.tobytes()).hexdigest()
            require(bool(np.isfinite(actual.data).all()), f"target.{mapped}.finite")
            require(expected_hash == target_hash, f"target.{mapped}.values")
            records.append(TensorAudit(name, mapped, shape, tuple(expected.shape), tuple(map(int, actual.shape)),
                                       source_hash, expected_hash, target_hash, operation, actual.data_offset, actual.n_bytes))
    ranges = sorted((t.data_offset, t.data_offset + t.n_bytes) for t in targets.values())
    require(all(end <= following for (_, end), (following, _) in zip(ranges, ranges[1:])), "target.payload.overlap")
    metadata, tokenizer_size, pads = audit_metadata(snapshot, reader, config)
    return ArtifactAudit(len(source_names), len(targets), config.num_hidden_layers, tuple(records), metadata,
                         output_binding, padded, tokenizer_size, pads, sum(t.n_bytes for t in targets.values()))


def audit_metadata(snapshot: Path, reader: GGUFReader, config: MambaConfig) -> tuple[tuple[MetadataAudit, ...], int, tuple[int, ...]]:
    """Compare all GGUF fields, including every vocabulary index and merge rank."""
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    vocab = tokenizer.get_vocab()
    size = vocabulary_size(config)
    reverse = {index: token for token, index in vocab.items()}
    require(len(reverse) == len(vocab) and all(0 <= n < size for n in reverse), "tokenizer.vocabulary.indices")
    added = tokenizer.added_tokens_decoder
    tokens, types = [], []
    for index in range(size):
        token = reverse.get(index, f"[PAD{index}]")
        kind = 1 if index in reverse else 5
        if index in added:
            entry = added[index]
            if not entry.normalized:
                token = tokenizer.decode(tokenizer.encode(token, add_special_tokens=False))
            special = entry.special or token in ("<pad>", "<mask>", "<2mass>", "[@BOS@]") or token.startswith("<|") and token.endswith("|>")
            kind = 3 if special else 4
            if kind == 4:
                token = token.replace("\u2581", " ")
        tokens.append(token)
        types.append(kind)
    raw = json.loads((snapshot / "tokenizer.json").read_text())
    require(raw["model"]["type"] == "BPE" and raw["normalizer"] == {"type": "NFC"}, "tokenizer.model")
    require(raw["pre_tokenizer"] == {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True}, "tokenizer.pre_tokenizer")
    merges = [entry if isinstance(entry, str) else " ".join(part.replace(" ", "\u0120") for part in entry)
              for entry in raw["model"]["merges"]]
    expected = {
        "general.architecture": ("STRING", "mamba"), "general.type": ("STRING", "model"),
        "general.name": ("STRING", snapshot.name), "general.finetune": ("STRING", snapshot.name),
        "general.file_type": ("UINT32", 0), "general.quantization_version": ("UINT32", 2),
        "mamba.context_length": ("UINT32", 2**20), "mamba.embedding_length": ("UINT32", config.hidden_size),
        "mamba.feed_forward_length": ("UINT32", 0), "mamba.attention.head_count": ("UINT32", 0),
        "mamba.block_count": ("UINT32", config.num_hidden_layers), "mamba.ssm.conv_kernel": ("UINT32", config.conv_kernel),
        "mamba.ssm.inner_size": ("UINT32", config.intermediate_size), "mamba.ssm.state_size": ("UINT32", config.state_size),
        "mamba.ssm.time_step_rank": ("UINT32", config.time_step_rank), "mamba.ssm.dt_b_c_rms": ("BOOL", False),
        "mamba.attention.layer_norm_rms_epsilon": ("FLOAT32", float(np.float32(config.layer_norm_epsilon))),
        "tokenizer.ggml.model": ("STRING", "gpt2"), "tokenizer.ggml.pre": ("STRING", "olmo"),
        "tokenizer.ggml.tokens": ("ARRAY,STRING", tokens), "tokenizer.ggml.token_type": ("ARRAY,INT32", types),
        "tokenizer.ggml.merges": ("ARRAY,STRING", merges),
    }
    total = sum(t.n_elements for t in reader.tensors)
    scale, suffix = (1e6, "M") if total > 1e6 else (1e3, "K")
    scaled = total / scale
    expected["general.size_label"] = ("STRING", f"{scaled:.{max(2 - len(str(round(scaled)).lstrip('0')), 0)}f}{suffix}")
    for short, long in (("bos", "bos"), ("eos", "eos"), ("unk", "unknown"), ("pad", "padding")):
        expected[f"tokenizer.ggml.{long}_token_id"] = ("UINT32", getattr(tokenizer, f"{short}_token_id"))
    expected.update({"GGUF.version": ("UINT32", 3), "GGUF.tensor_count": ("UINT64", len(reader.tensors)),
                     "GGUF.kv_count": ("UINT64", len(expected))})
    for field in sorted(reader.fields.keys() ^ expected.keys()):
        raise AuditError(f"metadata.{field}.presence")
    records: list[MetadataAudit] = []
    for name, (dtype, value) in expected.items():
        field = reader.fields[name]
        require(",".join(t.name for t in field.types) == dtype, f"metadata.{name}.type")
        require(field.contents() == value, f"metadata.{name}.value")
        records.append(MetadataAudit(name, tuple(dtype.split(",")), provenance.json_hash(value), len(field.data)))
    return tuple(records), len(vocab), tuple(n for n in range(size) if n not in reverse)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        require(args.out.resolve().is_relative_to(Path(__file__).resolve().parents[1]), "out.worktree")
        manifest = provenance.read_json(args.manifest)
        require(type(manifest) is dict, "manifest.type")
        identity = {key: value for key, value in manifest.items() if key != "captureId"}
        require(manifest["captureId"] == "prepare-" + provenance.json_hash(identity), "manifest.captureId")
        require(manifest["schemaVersion"] == 1 and manifest["repoId"] == provenance.REPO_ID, "manifest.schema")
        require(manifest["revision"] == REVISION, "manifest.revision")
        require(manifest["ggufSha256"] == GGUF_SHA, "manifest.ggufSha256")
        require(manifest["outtype"] == "f32", "manifest.outtype")
        require(provenance.source_identity(SOURCE) == manifest["source"], "manifest.source")
        require(provenance.environment_receipt() == manifest["environment"], "manifest.environment")
        snapshot = Path(manifest["snapshotPath"])
        provenance.verify_payloads(snapshot, manifest["payloads"])
        require(provenance.read_json(snapshot / "config.json") == manifest["config"], "manifest.config")
        for name, fixture in manifest["fixtures"].items():
            require(provenance.sha256(fixture["path"]) == fixture["sha256"], f"manifest.fixtures.{name}.sha256")
        require(provenance.sha256(manifest["ggufPath"]) == manifest["ggufSha256"], "manifest.ggufPath.sha256")
        audit = audit_pair(snapshot, Path(manifest["ggufPath"]))
        report = {"schemaVersion": 1, "status": "exact", "captureId": manifest["captureId"],
                  "manifestSha256": provenance.sha256(args.manifest), "ggufSha256": manifest["ggufSha256"],
                  "auditorSha256": provenance.sha256(Path(__file__)), "fixtures": manifest["fixtures"],
                  "source": manifest["source"], "environment": manifest["environment"], "audit": asdict(audit)}
        provenance.atomic_json(args.out, report)
    except (AuditError, ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(json.dumps({"status": "failed", "field": str(error)}), file=sys.stderr)
        return 1
    print(json.dumps({"status": "exact", "out": str(args.out), "sha256": provenance.sha256(args.out),
                      "sourceTensorCount": audit.sourceTensorCount, "targetTensorCount": audit.targetTensorCount}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
