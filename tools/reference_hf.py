#!/usr/bin/env python3
"""Capture local pinned HF schedules; exit zero means captured, not numeric PASS.

Run with the manifest's locked interpreter: python -B tools/reference_hf.py
--manifest MODEL.json --tokens FIXTURE.json --out NEW_DIRECTORY.
No dependency bootstrap: the immutable preparation lock owns this environment.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Final

# Set these before importing the numerical libraries; no optional kernel may load.
for _key, _value in {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                     "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}.items():
    os.environ[_key] = _value
if TYPE_CHECKING:
    import numpy as np
    import torch
    from transformers.models.mamba import modeling_mamba as mamba
    from tools import prepare_model as prepare
else:
    np = importlib.import_module("numpy")
    torch = importlib.import_module("torch")
    prepare = importlib.import_module("tools.prepare_model" if __package__ else "prepare_model")

OPTIONAL: Final = ("mamba_ssm", "causal_conv1d", "mambapy", "kernels")


@dataclass(frozen=True, slots=True)
class ReferenceError(Exception):
    field: str

    def __str__(self) -> str:
        return self.field


def require(condition: bool, field: str) -> None:
    if not condition:
        raise ReferenceError(field)


def eager_only() -> None:
    for name in OPTIONAL:
        require(importlib.util.find_spec(name) is None, "optionalKernel." + name)
    require(not torch.compiler.is_compiling(), "execution.compilation")


eager_only()
if not TYPE_CHECKING:
    mamba = importlib.import_module("transformers.models.mamba.modeling_mamba")
MODELING_SHA: Final = "4a5deb1cc9c22757826ac5dda343f5a0c3b928c50ac5d9c7c860e247bce06cdf"


@dataclass(frozen=True, slots=True)
class Schedule:
    token_ids: tuple[int, ...]
    vocab_size: int
    fixture_payload: bytes

    @property
    def fixture_sha256(self) -> str:
        return hashlib.sha256(self.fixture_payload).hexdigest()

    @classmethod
    def parse(cls, path: Path, vocab_size: int) -> Schedule:
        payload = path.read_bytes()
        document = json.loads(payload)
        expected = {"schemaVersion": 1, "kind": "synthetic-token-ids", "sequenceId": 0,
                    "prefillLength": 16, "positions": list(range(17)), "outputsPerCall": 1,
                    "implicitSpecialTokens": False}
        for key, value in expected.items():
            require(prepare.json_hash(document.get(key)) == prepare.json_hash(value), "tokens." + key)
        ids = document.get("tokenIds")
        require(type(ids) is list and len(ids) == 17, "tokens.tokenIds.length")
        require(all(type(token) is int and 0 <= token < vocab_size for token in ids), "tokens.tokenIds.vocabulary")
        return cls(tuple(ids), vocab_size, payload)


def verify_execution(model: mamba.MambaForCausalLM) -> None:
    eager_only()
    require(prepare.sha256(mamba.__file__) == MODELING_SHA, "runtime.modelingSha256")
    require(torch.is_inference_mode_enabled(), "execution.inferenceMode")
    require(torch.get_num_threads() == torch.get_num_interop_threads() == 1, "execution.threads")
    require(not any(module.training for module in model.modules()), "execution.eval")
    require(not model.config.use_mambapy, "execution.use_mambapy")
    for name, tensor in (*model.named_parameters(), *model.named_buffers()):
        require(tensor.device.type == "cpu", "execution.device." + name)
        require(not tensor.is_floating_point() or tensor.dtype == torch.float32, "execution.dtype." + name)


def snapshot(cache: mamba.MambaCache) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack allocates independent storage, never a view of mutable per-layer cache."""
    conv, ssm = torch.stack(cache.conv_states), torch.stack(cache.ssm_states)
    require(tuple(conv.shape) == (24, 1, 1536, 4), "cache.conv.shape")
    require(tuple(ssm.shape) == (24, 1, 1536, 16), "cache.ssm.shape")
    for tensor in (conv, ssm):
        require(tensor.dtype == torch.float32 and tensor.device.type == "cpu", "cache.dtypeDevice")
        require(bool(torch.isfinite(tensor).all()), "cache.finite")
    return conv, ssm


def row(logits: torch.Tensor, span: range, schedule: Schedule) -> torch.Tensor:
    """Select the last observed input row, preserving vocabulary ID order."""
    require((span.start, span.stop) in ((0, 16), (16, 17), (0, 17)), "logits.inputIndex")
    require(tuple(logits.shape) == (1, len(span), schedule.vocab_size), "logits.vocabularyShape")
    require(logits.dtype == torch.float32 and logits.device.type == "cpu", "logits.dtypeDevice")
    require(bool(torch.isfinite(logits).all()), "logits.finite")
    return logits[0, len(span) - 1, :].detach().clone()


def capture(model: mamba.MambaForCausalLM, schedule: Schedule, out: Path) -> None:
    """Run real HF calls; retain original history plus its source-grounded R slice."""
    verify_execution(model)
    arrays, records = {}, {}
    for label, spans in (("fresh", (range(17),)), ("split", (range(16), range(16, 17)))):
        cache = mamba.MambaCache(model.config, 1, dtype=torch.float32, device="cpu")
        for span in spans:
            phase = "prefill" if span.stop == 16 else "final"
            before = snapshot(cache)
            for kind, tensor in zip(("conv", "S"), before):
                arrays[f"{label}.before_{phase}.{kind}"] = tensor.numpy().copy()
            # HF 4.57.6 uses four cache slots as its prefill sentinel, NOT input positions.
            cache_positions = torch.arange(4) if span.start == 0 else torch.tensor([span.start])
            logits, returned = model(torch.tensor([schedule.token_ids[span.start:span.stop]], dtype=torch.long),
                                     cache_params=cache, cache_position=cache_positions,
                                     use_cache=True, return_dict=False)
            require(returned is cache, "cache.retainedIdentity")
            selected = row(logits, span, schedule)
            key = f"{label}.{phase}.logits"
            arrays[key] = selected.numpy().copy()
            records[key] = {"inputIndex": span.stop - 1, "tokenId": schedule.token_ids[span.stop - 1],
                            "sourceRow": len(span) - 1, "inputPositions": list(span),
                            "cachePositionArgument": cache_positions.tolist(), "sequenceId": 0,
                            "axes": ["vocabularyTokenId"]}
            for kind, tensor in zip(("conv", "S"), snapshot(cache)):
                arrays[f"{label}.after_{phase}.{kind}"] = tensor.numpy().copy()
    # Retained after-prefill copies must still match independently copied before-decode bytes.
    immutable = all(arrays[f"split.after_prefill.{kind}"].tobytes() ==
                    arrays[f"split.before_final.{kind}"].tobytes() for kind in ("conv", "S"))
    require(immutable, "cache.prefillSnapshotImmutable")
    for name, array in tuple(arrays.items()):
        if name.endswith(".conv"):
            # update_conv_state rolls left and appends; oldest of 4 is dropped on next decode.
            arrays[name.removesuffix("conv") + "R"] = array[..., 1:].copy()
    for name, array in arrays.items():
        records.setdefault(name, {"axes": ["layer", "sequence", "intermediate", "history" if name.endswith((".R", ".conv")) else "state"]})
        records[name].update(shape=list(array.shape), dtype=str(array.dtype),
                             sha256=hashlib.sha256(array.tobytes()).hexdigest())
    diagnostics = {}
    for name in ("logits", "R", "S"):
        fresh_key = "fresh.final.logits" if name == "logits" else "fresh.after_final." + name
        split_key = "split.final.logits" if name == "logits" else "split.after_final." + name
        references = [arrays[fresh_key]] if name == "logits" else arrays[fresh_key]
        candidates = [arrays[split_key]] if name == "logits" else arrays[split_key]
        diagnostics[name] = []
        for reference, candidate in zip(references, candidates):
            error = candidate.astype(np.float64) - reference.astype(np.float64)
            energy, squared = np.square(reference.astype(np.float64)).sum(), np.square(error).sum()
            diagnostics[name].append({"maxAbs": float(np.abs(error).max()), "rmse": float(np.sqrt(np.mean(error**2))),
                "nmse": float(squared / energy) if energy else (0.0 if squared == 0 else "infinity"),
                "absolutePercentiles": np.percentile(np.abs(error), [50, 95, 99]).tolist(),
                "nonzeroComponents": int(np.count_nonzero(error)), "toleranceViolationCount": None})
    out.mkdir(parents=True, exist_ok=False)
    np.savez(out / "arrays.npz", **arrays)
    prepare.atomic_json(out / "schedule.json", {"arrays": records, "diagnostics": diagnostics,
                                               "prefillSnapshotImmutable": immutable})


def run(manifest: Path, tokens: Path, out: Path) -> Path:
    document = prepare.read_json(manifest)
    require(document["captureId"] == "prepare-" + prepare.json_hash(
        {key: value for key, value in document.items() if key != "captureId"}), "manifest.captureId")
    local = Path(document["snapshotPath"])
    require(local.is_dir(), "manifest.snapshotPath")
    require(document["revision"] == "1e76775f628fbf1350fbe4dbb3d971ba64af25a1", "manifest.revision")
    prepare.verify_payloads(local, document["payloads"])
    require(prepare.read_json(local / "config.json") == document["config"], "manifest.config")
    require(prepare.sha256(document["ggufPath"]) == document["ggufSha256"], "manifest.ggufSha256")
    environment = prepare.environment_receipt()
    require(environment == document["environment"], "manifest.environment")
    schedule = Schedule.parse(tokens, prepare.validate_config(document["config"]))
    for entry in document["fixtures"].values():
        require(prepare.sha256(entry["path"]) == entry["sha256"], "manifest.fixtures")
    allowed = [prepare.read_json(document["fixtures"]["token-ids.json"]["path"])["tokenIds"]]
    allowed.extend(case["tokenIds"] for case in prepare.read_json(document["fixtures"]["calibration-ids.json"]["path"])["fixtures"])
    require(list(schedule.token_ids) in allowed, "tokens.predeclaredIds")
    require(not out.exists(), "out.alreadyExists")
    eager_only()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model = mamba.MambaForCausalLM.from_pretrained(str(local), local_files_only=True,
                                                dtype=torch.float32, use_safetensors=True).cpu().eval()
    with torch.inference_mode():
        capture(model, schedule, out)
    result = prepare.read_json(out / "schedule.json")
    result.update(schemaVersion=1, numericalStatus="unvalidated", manifestPath=str(manifest.resolve()),
        manifestSha256=prepare.sha256(manifest), preparationCaptureId=document["captureId"],
        ggufSha256=document["ggufSha256"], revision=document["revision"], snapshotPath=str(local),
        fixturePath=str(tokens.resolve()), fixtureSha256=schedule.fixture_sha256, fixture=json.loads(schedule.fixture_payload),
        source=document["source"], environment=environment, config=document["config"],
        runtime={"modelingPath": mamba.__file__, "modelingSha256": prepare.sha256(mamba.__file__),
                 "cacheClass": "transformers.models.mamba.modeling_mamba.MambaCache",
                 "branch": "MambaMixer.slow_forward", "dtype": "float32", "device": "cpu",
                 "inferenceMode": True, "eval": True, "optionalKernels": False,
                 "threads": torch.get_num_threads(), "interopThreads": torch.get_num_interop_threads(),
                 "torchBuild": torch.__config__.show(), "mkldnnEnabled": torch.backends.mkldnn.enabled},
        layout={"conv": [24, 1, 1536, 4], "R": "conv[...,1:] oldest-to-newest; per-layer [1,1536,3]",
                "S": "identity; per-layer [1,1536,16]", "prefillCachePosition": [0, 1, 2, 3],
                "decodeCachePosition": [16], "inputPositions": list(range(17))},
        scriptSha256=prepare.sha256(Path(__file__)), archiveSha256=prepare.sha256(out / "arrays.npz"))
    result["captureId"] = "hf-" + prepare.json_hash(result)
    prepare.atomic_json(out / "reference.json", result)
    return out / "reference.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "tokens", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = run(args.manifest, args.tokens, args.out)
    except (ReferenceError, ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 1
    print(json.dumps({"reference": str(receipt), "sha256": prepare.sha256(receipt), "numericalStatus": "unvalidated"}))
    return 0


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
