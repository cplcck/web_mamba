"""Tiny deterministic on-disk evidence fixtures; no ignored model artifacts."""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import ModuleType
from typing import Literal, TypedDict
import numpy as np


def array_fixture(directory: Path):
    payload = np.array([1, 2, 4, 8], dtype="<f4").tobytes()
    (directory / "arrays.bin").write_bytes(payload)
    return {"file": "arrays.bin", "offsetBytes": 0, "bytes": len(payload),
        "shape": [4], "dtype": "f32", "byteOrder": "little", "order": "C",
        "axisLabels": ["vocabulary"], "epoch": 3,
        "sha256": hashlib.sha256(payload).hexdigest()}


def graph_fixture() -> str:
    """A real-format dependency leaf, arithmetic node, and zero-element view."""
    import json
    tensors = []
    for number, op, shape, source in ((0, "NONE", [4, 1, 1, 1], None),
                                     (1, "SCALE", [4, 1, 1, 1], "t0"),
                                     (2, "VIEW", [0, 1, 1, 1], "t1")):
        view = number == 2
        size = 0 if view else 16
        tensors.append({"id": f"t{number}", "name": f"tensor{number}", "op": op,
            "role": "activation", "dtype": "f32", "nativeShape": shape, "strides": [4, 16, 16, 16],
            "numel": 0 if view else 4, "logicalBytes": size, "typeBlockSize": 1, "typeSize": 4,
            "src": [source, *([None] * 9)], "viewSourceId": source if view else None,
            "viewOffsetBytes": 0 if view else None, "opParamsI32": [1065353216 if number == 1 else 0] + [0] * 15,
            "schedulerObserved": number > 0, "arithmeticExecution": number == 1,
            "metadataOnly": number != 1, "classification": "observed",
            "storage": {"ggmlNbytes": size, "requiredAllocBytes": None if view else size,
                "bufferId": "b", "bufferOffsetBytes": 0, "bufferBytes": 32,
                "observationEpoch": number * 2 + 1, "allocatorSlotBytes": None,
                "allocatorSlotReason": "not exposed"}})
    events = [{"tensorId": f"t{number}", "phase": phase, "epoch": number * 2 + offset}
              for number in (1, 2) for phase, offset in (("ask", 1), ("complete", 2))]
    return json.dumps({"tensors": tensors, "events": events})


def state_fixture(directory: Path) -> str:
    """All 24 layer identities share tiny zero payload fixtures, not model arrays."""
    import json
    payload_r = np.zeros((1, 1536, 3), dtype="<f4").tobytes()
    payload_s = np.zeros((1, 1536, 16), dtype="<f4").tobytes()
    (directory / "arrays.bin").write_bytes(payload_r + payload_s)
    snapshots = []
    for label, position, epoch in (("before-prefill", -1, 1), ("after-prefill", 16, 2)):
        row = None if position == -1 else 0
        arrays = []
        for layer in range(24):
            for family, width, payload, offset in (("R", 3, payload_r, 0), ("S", 16, payload_s, len(payload_r))):
                arrays.append({"file": "arrays.bin", "offsetBytes": offset, "bytes": len(payload),
                    "shape": [1, 1536, width], "dtype": "f32", "byteOrder": "little", "order": "C",
                    "axisLabels": ["cell" if row is None else "sequence", "inner", "history" if family == "R" else "state"],
                    "epoch": epoch, "sha256": hashlib.sha256(payload).hexdigest(), "snapshot": label,
                    "family": family, "layer": layer, "sourceRow": row, "fullInitialBuffer": row is None,
                    "nativeShape": [1536 * width, 1, 1, 1], "nativeStrides": [4, 1536 * width * 4, 1536 * width * 4, 1536 * width * 4]})
        snapshots.append({"label": label, "position": position, "epoch": epoch, "activeCell": row,
            "activeRow": row, "size": 1, "head": 0, "n_rs_seq": 0, "rs_idx": [0], "arrays": arrays,
            "cells": [{"index": 0, "pos": position, "src": -1 if row is None else row,
                       "src0": -1 if row is None else row, "tail": -1 if row is None else row,
                       "sequenceIds": [] if row is None else [0]}]})
    return json.dumps(snapshots)


def hierarchy_fixture():
    """A tiny explicit one-block topology, not measured model execution."""
    from dataclasses import replace
    from tools.web_graph import Graph, Node, Storage
    nodes = []
    prototype = Node("input", "input", "NONE", "activation", "F32", (1, 1, 1, 1), (4, 4, 4, 4),
        1, 1, 4, Storage(4, 4, "b", 0, 4, 0, None, "unexposed"), (None,) * 10, None, None, (0,) * 16, False, False)

    def emit(name: str, op: str, sources: tuple[str, ...]) -> str:
        """Accumulate a test graph with independent stable source identities."""
        identity = f"t{len(nodes)}"
        role = "weight" if name.startswith("blk.") or name.endswith(".weight") else "activation"
        node = replace(prototype, id=identity, name=name, op=op, role=role,
            src=sources + (None,) * (10-len(sources)), schedulerObserved=op != "NONE", arithmeticExecution=op != "NONE")
        if op == "CPY":
            node = replace(node, role="state", viewSourceId=sources[1], viewOffsetBytes=0,
                           storage=replace(node.storage, requiredAllocBytes=None))
        nodes.append(node)
        return identity

    embedding_weight = emit("token_embd.weight", "NONE", ())
    embedded = emit("embd", "GET_ROWS", (embedding_weight,))
    norm = emit("norm-0", "RMS_NORM", (embedded,))
    norm = emit("attn_norm-0", "MUL", (norm, emit("blk.0.attn_norm.weight", "NONE", ())))
    xz = emit("xz", "MUL_MAT", (emit("blk.0.ssm_in.weight", "NONE", ()), norm))
    r = emit("cache_r_l0", "NONE", ())
    emit("cache_r_l0 (copy)", "CPY", (xz, r))
    conv = emit("conv", "SSM_CONV", (xz, emit("blk.0.ssm_conv1d.weight", "NONE", ())))
    conv = emit("conv_bias", "ADD", (conv, emit("blk.0.ssm_conv1d.bias", "NONE", ())))
    conv = emit("conv_act", "UNARY", (conv,))
    xp = emit("xp", "MUL_MAT", (emit("blk.0.ssm_x.weight", "NONE", ()), conv))
    dt = emit("dt", "MUL_MAT", (emit("blk.0.ssm_dt.weight", "NONE", ()), xp))
    dt = emit("dt_bias", "ADD", (dt, emit("blk.0.ssm_dt.bias", "NONE", ())))
    state = emit("cache_s_l0", "NONE", ())
    scan = emit("scan", "SSM_SCAN", (state, conv, dt, emit("blk.0.ssm_a", "NONE", ()), xp, xp))
    emit("cache_s_l0 (copy)", "CPY", (scan, state))
    skip = emit("skip", "MUL", (conv, emit("blk.0.ssm_d", "NONE", ())))
    gated = emit("gated", "GLU", (xz, emit("skip_add", "ADD", (scan, skip))))
    out = emit("out", "MUL_MAT", (emit("blk.0.ssm_out.weight", "NONE", ()), gated))
    out = emit("l_out-0", "ADD", (out, embedded))
    final = emit("norm", "RMS_NORM", (out,))
    final = emit("result_norm", "MUL", (final, emit("output_norm.weight", "NONE", ())))
    emit("result_output", "MUL_MAT", (embedding_weight, final))
    return Graph(tuple(nodes), sum(n.schedulerObserved for n in nodes) * 2)


def corpus_fixture():
    """One independently specified finite comparison; never a real approval."""
    from tools.capture_data import Observation
    from tools.validate import Comparison, Corpus
    reference = np.array([1, 2, 4, 8], dtype="<f4").tobytes()
    candidate = np.array([1, 3, 4, 8], dtype="<f4").tobytes()
    pair = Comparison("fixture/H_F-H_C/logits",
        Observation(np.frombuffer(reference, dtype="<f4"), hashlib.sha256(reference).hexdigest(), 0),
        Observation(np.frombuffer(candidate, dtype="<f4"), hashlib.sha256(candidate).hexdigest(), 0))
    return Corpus((pair,), (), "fixture-configuration", '{"auditSha256":"fixture-audit"}')


def envelope_fixture() -> str:
    """Explicit tiny-array bounds, unrelated to any model policy or approval."""
    import json
    return json.dumps({"referenceSha256": hashlib.sha256(np.array([1, 2, 4, 8], dtype="<f4").tobytes()).hexdigest(),
        "shape": [4], "absoluteBound": 1, "maxAbs": 1, "rmse": 0.5, "nmse": 1 / 85,
        "absolutePercentiles": [1, 1, 1]})


class ApprovalAttackResult(TypedDict):
    mutation: Literal["self-approval", "self-rehashed-policy"]
    argv: list[str]
    exit: int
    stdout: str
    stderr: str
    candidateSha256: str
    policySha256: str


def isolated_validator(directory: Path) -> ModuleType:
    """Load unchanged copied tools with their own fixed authority root, not a ROOT mock."""
    import importlib.util
    import shutil
    import sys
    from unittest.mock import patch
    tools = directory / "tools"
    tools.mkdir()
    (directory / "fixtures").mkdir()
    source = Path(__file__).resolve().parents[1] / "tools"
    for name in ("validate.py", "export_web.py", "capture_data.py", "web_graph.py", "prepare_model.py"):
        shutil.copyfile(source / name, tools / name)
    specification = importlib.util.spec_from_file_location("isolated_validator", tools / "validate.py")
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    with patch.object(sys, "path", [str(tools), *sys.path]), patch.dict(sys.modules):
        for name in ("validate", "export_web", "capture_data", "web_graph", "prepare_model"):
            sys.modules.pop(name, None)
        sys.modules[specification.name] = module
        specification.loader.exec_module(module)
    return module


def authority_fixture(directory: Path, decision: Literal["approved", "rejected"]) -> tuple[ModuleType, Path, Path]:
    """Provision hash-valid synthetic authority so tests reach the review-decision gate."""
    import json
    numeric = isolated_validator(directory)
    candidate = directory / "synthetic-candidate.json"
    candidate.write_text(numeric.candidate_policy(corpus_fixture()))
    policy = json.loads(candidate.read_text())
    review = directory / "fixtures/numerical-review.json"
    identity = {"reviewerId": "synthetic-independent-reviewer", "candidateSha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
                "configurationHash": policy["configurationHash"], "auditSha256": policy["auditSha256"]}
    review.write_text(numeric.encoded({"decision": decision, **identity}))
    (directory / "fixtures/numerical-authorization.json").write_text(numeric.encoded({"schemaVersion": 1,
        "status": "approved", **identity, "reviewSha256": hashlib.sha256(review.read_bytes()).hexdigest()}))
    return numeric, candidate, review


def approval_attack_results(directory: Path) -> list[ApprovalAttackResult]:
    """Replay NUM-01 with real CLIs and unchanged archives; caller owns sandbox cleanup."""
    import json
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[1]
    captures = root / ".artifacts/task-7/captures"
    candidate_path = root / ".artifacts/task-7/validation/review-candidate-policy.json"
    manifest = root.parent / "mamba1-130m-visualizer-foundation/.artifacts/manifests/model.json"
    from tools.validate import encoded
    controlled = directory / "candidate-controlled-captures"
    controlled.mkdir()
    for run in captures.iterdir():
        (controlled / run.name).symlink_to(run, target_is_directory=True)
    results: list[ApprovalAttackResult] = []
    for changed in (False, True):
        candidate = json.loads(candidate_path.read_text())
        if changed:
            candidate["envelopes"][next(iter(candidate["envelopes"]))]["absoluteBound"] += 1
        digest = hashlib.sha256(encoded(candidate).encode()).hexdigest()
        receipt = controlled / "candidate-authored-review.json"
        receipt.write_text(encoded({"decision": "approved", "reviewerId": "st_01a075d0", "candidateSha256": digest,
            "configurationHash": candidate["configurationHash"], "auditSha256": candidate["auditSha256"], "adversarialFixture": True}))
        policy = {**candidate, "status": "approved", "approval": {"receipt": str(receipt),
            "receiptSha256": hashlib.sha256(receipt.read_bytes()).hexdigest(), "candidateSha256": digest}}
        policy_path = directory / "adversarial-policy.json"
        policy_path.write_text(encoded(policy))
        command = [sys.executable, "-B", "tools/validate.py", "--manifest", str(manifest), "--captures", str(controlled),
            "--policy", str(policy_path), "--out", str(directory / "attack-report.json")]
        completed = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=240)
        results.append({"mutation": "self-rehashed-policy" if changed else "self-approval", "argv": command,
            "exit": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr,
            "candidateSha256": digest, "policySha256": hashlib.sha256(policy_path.read_bytes()).hexdigest()})
    return results


def call_fixture() -> str:
    """Canonical-sized row metadata without model values or model files."""
    import json
    return json.dumps({"P": 1, "T": 17, "Q": 17, "O": 1, "inputBegin": 0, "inputEndExclusive": 17,
        "position": 16, "sequenceId": 0, "decodeExit": 0, "positions": list(range(17)), "previousPosition": -1,
        "logits": {"inputIndex": 16, "tokenId": 16, "position": 16, "outputRow": 0, "batchTokenIndex": 16,
            "vocabStart": 0, "vocabEndExclusive": 50280, "sequenceId": 0, "shape": [50280],
            "axisLabels": ["vocabulary"], "dtype": "f32"}})
