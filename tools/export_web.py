"""Audit-bound static website publication; raw model arrays never cross this boundary."""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import importlib
import json
import os
from pathlib import Path
from types import MappingProxyType
import subprocess
import sys
import tempfile
from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:
    from tools import capture_data as data, prepare_model as provenance, validate as validation, web_graph
else:
    data = importlib.import_module("tools.capture_data" if __package__ else "capture_data")
    provenance = importlib.import_module("tools.prepare_model" if __package__ else "prepare_model")


ROOT: Final = Path(__file__).resolve().parents[1]
AUDIT: Final = ROOT.parent / "mamba1-130m-visualizer-evidence/.artifacts/validation/artifact-audit.json"
PUBLIC_BUDGET: Final = 8 * 1024 * 1024


def verify_inputs(manifest: Path, audit: Path) -> data.Inputs:
    """Recheck immutable preparation and exact artifact audit before using captures."""
    d = provenance.read_json(manifest)
    data.require(type(d) is dict and d.get("captureId") == "prepare-" + provenance.json_hash({k: v for k, v in d.items() if k != "captureId"}), "manifest.identity")
    data.require(d["schemaVersion"] == 1 and d["repoId"] == "state-spaces/mamba-130m-hf" and d["outtype"] == "f32", "manifest.format")
    data.require(d["revision"] == "1e76775f628fbf1350fbe4dbb3d971ba64af25a1" and d["ggufSha256"] == "af897eef0adfda25444be8fd2e6cbe3e858a107f7926625f7fb71d37f51bcaae", "manifest.model")
    data.require(provenance.validate_config(d["config"]) == 50280, "manifest.configuration")
    data.require(provenance.source_identity(Path(d["source"]["path"])) == d["source"], "manifest.source")
    data.require(provenance.environment_receipt() == d["environment"], "manifest.environment")
    provenance.verify_payloads(Path(d["snapshotPath"]), d["payloads"])
    data.require(provenance.sha256(d["ggufPath"]) == d["ggufSha256"], "manifest.ggufHash")
    for fixture in d["fixtures"].values():
        data.require(provenance.sha256(fixture["path"]) == fixture["sha256"], "manifest.fixtureHash")
    audited = provenance.read_json(audit)
    data.require(audited["schemaVersion"] == 1 and audited["status"] == "exact", "audit.status")
    data.require(audited["manifestSha256"] == provenance.sha256(manifest) and audited["captureId"] == d["captureId"] and audited["ggufSha256"] == d["ggufSha256"], "audit.identity")
    data.require(audited["source"] == d["source"] and audited["environment"] == d["environment"] and audited["fixtures"] == d["fixtures"], "audit.configuration")
    data.require(audited["auditorSha256"] == provenance.sha256(Path(__file__).with_name("audit_gguf.py")), "audit.implementation")
    # The accepted exact audit is an independent prerequisite, not a candidate's declaration.
    data.require(provenance.sha256(audit) == "1f2f02c063a9766df28b979a07d6b0ca429bd92e191694c26b2e8552c47e91b7", "audit.receiptHash")
    canonical = provenance.read_json(d["fixtures"]["token-ids.json"]["path"])
    calibration = provenance.read_json(d["fixtures"]["calibration-ids.json"]["path"])
    cases = {"canonical": tuple(canonical["tokenIds"]), **{case["id"]: tuple(case["tokenIds"]) for case in calibration["fixtures"]}}
    return data.Inputs(manifest.resolve(), provenance.sha256(manifest), d["captureId"], d["ggufSha256"],
        provenance.json_hash(d["environment"]), provenance.json_hash(d["config"]), provenance.json_hash(d["source"]),
        provenance.json_hash({k: d["source"][k] for k in ("commit", "productPatchSha256", "status", "worktreeDiffSha256")}), MappingProxyType(cases))


def checked_report(corpus: validation.Corpus, policy_source: str | None, supplied: str) -> str:
    """Recompute current inputs; a copied PASS report cannot substitute for validation."""
    numeric = importlib.import_module("tools.validate" if __package__ else "validate")
    current = numeric.evaluate(corpus, policy_source)
    data.require(json.loads(current) == json.loads(supplied), "export.staleReport")
    return current


def symbolic_shapes(graph: web_graph.Graph) -> str:
    """Compile pinned semantic stage rules, never infer axes by matching extents."""
    module = importlib.import_module("tools.web_graph" if __package__ else "web_graph")
    entities = module.hierarchy(graph)
    by_id = {n.id: n for n in graph.nodes}
    owners = {e.outputTensorIds[0]: e.parentId for e in entities if e.kind == "operator"}
    def const(n: int):
        return {"op": "const", "value": n}
    def dim(n: str):
        return {"op": "dim", "name": n}
    shapes = {n.id: [const(v) for v in reversed(n.nativeShape)] for n in graph.nodes}
    if sum(e.kind == "block" for e in entities) != 24:
        return json.dumps(shapes)  # Tiny test graphs are not the pinned model architecture.
    p, t, q, o = (dim(n) for n in ("P", "T", "Q", "O"))
    for node in graph.nodes:
        stage = owners.get(node.id, "").split("/")[-1]
        shape = None
        if node.name == "inp_tokens":
            shape = [q, 1, 1, 1]
        elif any(n.op == "GET_ROWS" and owners.get(n.id) == "block.23/residual" and n.src[1] == node.id for n in graph.nodes):
            shape = [o, 1, 1, 1]
        elif stage in ("embedding", "normalization", "final-normalization", "final-projection", "residual"):
            width = 50280 if stage == "final-projection" else 768
            rows = o if stage.startswith("final-") or owners[node.id].startswith("block.23/residual") else q
            shape = [width, rows, 1, 1]
        elif stage == "input-projection-split":
            shape = [t, 1536, p, 1] if node.op == "TRANSPOSE" else [{"RESHAPE": 768, "MUL_MAT": 3072, "VIEW": 1536}[node.op], t, p, 1]
        elif stage == "convolution-history" and node.op in ("CONCAT", "TRANSPOSE", "SSM_CONV", "ADD", "UNARY"):
            shape = ([{"op": "add", "left": t, "right": const(3)}, 1536, p, 1] if node.op == "CONCAT" else
                     [t, 1536, p, 1] if node.op == "TRANSPOSE" else [1536, t, p, 1])
        elif stage == "dt-b-c-projection":
            if node.op == "VIEW":
                bc = any(n.op == "SSM_SCAN" and node.id in n.src[4:6] for n in graph.nodes)
                shape = [16, 1, t, p] if bc else [48, t, p, 1]
            else:
                selection = node.op == "MUL_MAT" and by_id[node.src[0]].name.endswith("ssm_x.weight")
                shape = [80 if selection else 1536, t, p, 1]
        elif stage == "selective-scan-state" and node.op == "SSM_SCAN":
            shape = [{"op": "mul", "left": {"op": "mul", "left": const(1536), "right": p},
                      "right": {"op": "add", "left": t, "right": const(16)}}, 1, 1, 1]
        elif stage == "selective-scan-state" and any(n.op == "SSM_SCAN" and node.id == n.src[1] for n in graph.nodes):
            shape = [1, 1536, t, p]
        elif stage == "skip-gating":
            shape = [1536, t, p, 1]
        elif stage == "output-projection":
            shape = [768, q, 1, 1] if node.op == "RESHAPE" else [768, t, p, 1]
        if shape is not None:
            shapes[node.id] = [const(v) if type(v) is int else v for v in reversed(shape)]
    return json.dumps(shapes)


def formula_values(source: str, dimensions: Mapping[str, int]) -> tuple[int, ...]:
    """Evaluate compiler-produced closed ASTs; all results stay safe nonnegative integers."""
    def value(expression) -> int:
        if expression["op"] == "const":
            return expression["value"]
        if expression["op"] == "dim":
            return dimensions[expression["name"]]
        left, right = value(expression["left"]), value(expression["right"])
        result = {"add": lambda: left+right, "mul": lambda: left*right, "sub": lambda: left-right}[expression["op"]]()
        data.require(type(result) is int and 0 <= result < 2**53, "formula.integer")
        return result
    return tuple(value(expression) for expression in json.loads(source))


def state_summary(run: data.Capture, capture_id: str, scenario: str) -> str:
    """Publish bounded F64 statistics and source-bound mapping, never state vectors."""
    before, after = run.snapshots[:2] if scenario == "prefill" else run.snapshots[2:]
    result = {"nativeCaptureId": run.capture_id, "sequenceId": 0}
    for key, snapshot in (("before", before), ("after", after)):
        arrays = []
        for index, observation in enumerate(snapshot.arrays):
            values = observation.values.astype(np.float64)
            count = values.size
            arrays.append({"layer": index // 2, "family": "R" if index % 2 == 0 else "S", "dtype": "F32",
                "shape": list(values.shape), "axisLabels": ["cell" if snapshot.active_row is None else "sequence", "inner", "history" if index % 2 == 0 else "state"],
                "nativeShape": [count, 1, 1, 1], "nativeStrides": [4, count*4, count*4, count*4],
                "numel": count, "logicalBytes": count*4, "sha256": observation.sha256, "sourceRow": snapshot.active_row,
                "fullInitialBuffer": snapshot.active_row is None,
                "summary": {"min": float(values.min()), "max": float(values.max()), "mean": float(values.mean()), "rms": float(np.sqrt(np.square(values).mean())), "finite": True}})
        result[key] = {"id": run.capture_id + "/" + snapshot.label, "captureId": capture_id,
            "label": snapshot.label, "position": snapshot.position, "epoch": snapshot.epoch,
            "mapping": json.loads(snapshot.mapping_json), "arrays": arrays}
    left, right = run.snapshots[1:3]
    data.require(left.mapping_json == right.mapping_json and [a.sha256 for a in left.arrays] == [a.sha256 for a in right.arrays], "export.cacheContinuity")
    result["continuity"] = {"fromSnapshotId": run.capture_id + "/" + left.label, "toSnapshotId": run.capture_id + "/" + right.label,
        "mappingEqual": True, "arraysByteEqual": True, "comparedArrays": len(left.arrays)}
    return json.dumps(result, allow_nan=False)


def symbolic_support(scenario: str) -> str:
    """Separate permitted payload estimates from canonical numerical observations."""
    return json.dumps({"classification": "symbolic-estimate", "P": {"min": 1, "max": 1},
        "T": {"min": 1, "max": 32 if scenario == "prefill" else 1}, "O": {"min": 1, "maxDimension": "Q"},
        "Q": {"op": "mul", "left": {"op": "dim", "name": "P"}, "right": {"op": "dim", "name": "T"}},
        "storageAllocation": "observed-only"})


def render(corpus: validation.Corpus, inputs: data.Inputs, report_source: str) -> Mapping[str, str]:
    """Build coupled metadata documents; publication authorization remains separate."""
    graph_module = importlib.import_module("tools.web_graph" if __package__ else "web_graph")
    manifest = provenance.read_json(inputs.manifest_path)
    audit = provenance.read_json(AUDIT)["audit"]
    report = json.loads(report_source)
    canonical = next(run for run in corpus.captures if run.directory.name == "canonical-1-llama-split")
    native = provenance.read_json(canonical.directory / "capture.json")
    identity = "web-" + provenance.json_hash({"configurationHash": corpus.configuration_hash,
        "nativeCaptureId": canonical.capture_id, "reportSha256": hashlib.sha256(report_source.encode()).hexdigest()})
    header = {"schemaVersion": 1, "captureId": identity, "ggufSha256": inputs.gguf_sha}
    weights = {t["targetName"]: t for t in audit["tensors"]}
    payloads = {name: (t["payloadOffset"], t["payloadBytes"]) for name, t in weights.items()}
    files, captures = {}, {}
    for scenario, graph in zip(("prefill", "decode"), canonical.graphs):
        observed_weights = [t for t in graph.nodes if t.role == "weight"]
        data.require({t.name for t in observed_weights} == set(weights), "export.weightInventory")
        for tensor in observed_weights:
            weight = weights[tensor.name]
            data.require(tensor.dtype == weight["targetDtype"] and list(tensor.nativeShape) == weight["nativeShape"] + [1] * (4-len(weight["nativeShape"])), "export.auditedWeightLayout")
        context = graph_module.DocumentIdentity(identity + "-" + scenario, inputs.gguf_sha, scenario)
        document = json.loads(graph_module.document(graph, context, payloads))
        document.update(numericalStatus=report["status"], policySha256=report["policySha256"],
                        recurrentState=json.loads(state_summary(canonical, context.capture_id, scenario)))
        files[scenario + ".json"] = json.dumps(document, separators=(",", ":"), allow_nan=False) + "\n"
        captures[scenario] = {"captureId": context.capture_id, "path": scenario + ".json",
            "sha256": hashlib.sha256(files[scenario + ".json"].encode()).hexdigest(),
            "tensorCount": len(graph.nodes), "operatorCount": graph.event_count // 2,
            "emptyTensorCount": sum(t.numel == 0 for t in graph.nodes)}
    public_manifest = {**header, "modelId": "mamba-130m", "captures": captures, "artifactAuditStatus": "exact",
        "numericalStatus": report["status"], "policySha256": report["policySha256"], "reviewSha256": report["reviewSha256"],
        "observedDimensions": {name: json.loads(files[name + ".json"])["scenario"]["dimensions"] for name in captures},
        "symbolicSupport": {name: json.loads(symbolic_support(name)) for name in captures},
        "publicBudgetBytes": PUBLIC_BUDGET, "supported": {"P": [1], "T": [1, 16], "O": [1], "context": 256, "stateCapacity": 1,
            "backend": "CPU", "dtype": "F32", "offload": False, "threads": 1, "otherConfigurations": "unvalidated"}}
    source = {k: v for k, v in manifest["source"].items() if k not in ("path", "status")}
    environment = {k: v for k, v in manifest["environment"].items() if k != "pythonExecutable"}
    build = native["identity"]["build"]
    public_provenance = {**header, "repoId": manifest["repoId"], "revision": manifest["revision"],
        "manifestSha256": inputs.manifest_sha, "preparationCaptureId": inputs.preparation_id,
        "source": source, "environment": environment, "artifactAuditSha256": report["bindings"]["auditSha256"],
        "compiler": {k: build[k] for k in ("compilerId", "compilerVersion", "compilerSha256")},
        "buildFiles": {Path(name).name: digest for name, digest in build["files"].items()},
        "cmakeFlagsSha256": provenance.json_hash(build["cmakeFlags"]),
        "cmakeFlags": {k: v for k, v in build["cmakeFlags"].items() if k.startswith(("GGML_", "LLAMA_")) and not isinstance(v, str) or k.startswith(("GGML_", "LLAMA_")) and "/" not in v},
        "configured": native["run"]["configured"], "effective": native["run"]["effective"],
        "nativeEnvironment": native["identity"]["environment"], "systemInfo": native["identity"]["systemInfo"],
        "configurationHash": corpus.configuration_hash, "canonicalNativeCaptureId": canonical.capture_id,
        "numericalStatus": report["status"], "policySha256": report["policySha256"], "reviewSha256": report["reviewSha256"]}
    model = {**header, "modelId": "mamba-130m", "architecture": "Mamba1", "configuration": manifest["config"],
        "blockCount": audit["blockCount"], "vocabularySize": audit["vocabularySize"],
        "tokenizerVocabularySize": audit["tokenizerVocabularySize"], "paddedTokenIds": audit["paddedTokenIds"],
        "weightDirectory": audit["tensors"], "weightAccounting": {"uniqueGgufPayloadBytes": audit["uniquePayloadBytes"],
            "physicalTensorCount": audit["targetTensorCount"], "ggufFileBytes": Path(manifest["ggufPath"]).stat().st_size,
            "tiedOutput": audit["outputBinding"], "embeddingAndOutputTensor": "token_embd.weight",
            "runtimePhysicalWeightRecordsPerCall": len(observed_weights),
            "runtimeWeightAddressedBytesPerCall": sum(t.storage.ggmlNbytes for t in observed_weights),
            "peakMemory": None, "peakMemoryReason": "retained graphs do not measure allocator high-water memory"}}
    for name, content in (("manifest", public_manifest), ("model", model), ("provenance", public_provenance), ("validation", {**report, **header})):
        files[name + ".json"] = json.dumps(content, separators=(",", ":"), allow_nan=False) + "\n"
    return MappingProxyType(files)


def write_bundle(files: Mapping[str, str], destination: Path, draft: bool) -> None:
    """Check the complete bounded bundle before replacing files; manifest goes last."""
    data.require(set(files) == {name + ".json" for name in ("manifest", "model", "provenance", "validation", "prefill", "decode")}, "export.fileInventory")
    data.require(sum(len(value.encode()) for value in files.values()) <= PUBLIC_BUDGET, "export.publicBudget")
    data.require(all("/home/" not in value and "/tmp/" not in value and "file://" not in value for value in files.values()), "export.privatePath")
    if draft:
        data.require(destination.resolve().is_relative_to(ROOT / ".artifacts/task-7") or destination.resolve().is_relative_to(ROOT / ".omo/evidence/mamba1-130m-visualizer/task-7"), "export.draftBoundary")
    else:
        data.require(json.loads(files["manifest.json"])["numericalStatus"] == "pass", "export.unvalidated")
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".export-", dir=ROOT / ".omo/evidence/mamba1-130m-visualizer/task-7") as temporary:
        staged = Path(temporary)
        for name, source in files.items():
            (staged / name).write_text(source)
        for name in [n for n in files if n != "manifest.json"] + ["manifest.json"]:
            os.replace(staged / name, destination / name)


def main() -> int:
    numeric = importlib.import_module("tools.validate" if __package__ else "validate")
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "captures", "validation", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    parser.add_argument("--policy", type=Path, default=ROOT / "fixtures/numerical-policy.json")
    parser.add_argument("--draft", action="store_true")
    args = parser.parse_args()
    try:
        data.require(not args.policy.resolve().is_relative_to(args.captures.resolve()), "policy.candidateControlled")
        inputs = verify_inputs(args.manifest, args.audit)
        corpus = numeric.collect(inputs, args.captures, provenance.sha256(args.audit))
        source = args.policy.read_text() if args.policy.is_file() else None
        report = checked_report(corpus, source, args.validation.read_text())
        data.require(args.draft or json.loads(report)["status"] == "pass", "export.unvalidated")
        files = render(corpus, inputs, report)
        write_bundle(files, args.out, args.draft)
    except (data.ValidationError, ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(json.dumps({"status": "failed", "field": str(error)}), file=sys.stderr)
        return 1
    print(json.dumps({"status": "draft-unvalidated" if args.draft else "pass", "bytes": sum(len(s.encode()) for s in files.values()),
        "files": {name: provenance.sha256(args.out / name) for name in files}}))
    return 2 if args.draft else 0


if __name__ == "__main__":
    raise SystemExit(main())
