"""Immutable typed evidence ingestion for the pinned CPU capture formats.

The preparation lock owns dependencies. Raw JSON exists only in these parsers;
consumers receive frozen records and bytes-backed read-only NumPy arrays.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import math
import importlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, TypedDict

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from tools import prepare_model as provenance
    from tools.web_graph import Graph
else:
    provenance = importlib.import_module("tools.prepare_model" if __package__ else "prepare_model")


@dataclass(frozen=True, slots=True)
class ValidationError(Exception):
    field: str

    def __str__(self) -> str:
        return self.field


def require(condition: bool, field: str) -> None:
    if not condition:
        raise ValidationError(field)


class ArrayDescriptor(TypedDict):
    file: str
    offsetBytes: int
    bytes: int
    shape: list[int]
    dtype: str
    byteOrder: str
    order: str
    axisLabels: list[str]
    epoch: int
    sha256: str


@dataclass(frozen=True, slots=True)
class Observation:
    values: NDArray[np.float32] | NDArray[np.int32]
    sha256: str
    epoch: int


def read_array(directory: Path, descriptor: ArrayDescriptor) -> Observation:
    """Check descriptor and payload before exposing an immutable logical array."""
    require(descriptor["file"] == "arrays.bin", "array.file")
    require(descriptor["dtype"] in ("f32", "i32") and descriptor["byteOrder"] == "little"
            and descriptor["order"] == "C", "array.encoding")
    shape = descriptor["shape"]
    require(type(shape) is list and all(type(n) is int and 0 <= n <= 2**53 - 1 for n in shape), "array.shape")
    offset, size = descriptor["offsetBytes"], descriptor["bytes"]
    require(type(offset) is int and offset >= 0 and type(size) is int and size == math.prod(shape) * 4, "array.range")
    require(size <= (directory / "arrays.bin").stat().st_size - offset, "array.range")
    require(len(descriptor["axisLabels"]) == len(shape), "array.axes")
    require(type(descriptor["epoch"]) is int and descriptor["epoch"] >= 0, "array.epoch")
    with (directory / "arrays.bin").open("rb") as stream:
        stream.seek(offset)
        payload = stream.read(size)
    require(len(payload) == size and hashlib.sha256(payload).hexdigest() == descriptor["sha256"], "array.sha256")
    values = np.frombuffer(payload, dtype={"f32": "<f4", "i32": "<i4"}[descriptor["dtype"]]).reshape(shape)
    require(bool(np.isfinite(values).all()), "array.finite")
    return Observation(values, descriptor["sha256"], descriptor["epoch"])


@dataclass(frozen=True, slots=True)
class Inputs:
    manifest_path: Path
    manifest_sha: str
    preparation_id: str
    gguf_sha: str
    environment_hash: str
    config_hash: str
    source_hash: str
    native_source_hash: str
    token_cases: Mapping[str, tuple[int, ...]]


@dataclass(frozen=True, slots=True)
class Snapshot:
    label: str
    position: int
    epoch: int
    active_row: int | None
    mapping_json: str
    arrays: tuple[Observation, ...]


@dataclass(frozen=True, slots=True)
class Capture:
    directory: Path
    receipt_sha: str
    capture_id: str
    archive_sha: str
    runtime_hash: str
    fixture_sha: str
    token_ids: tuple[int, ...]
    arrays: Mapping[str, Observation]
    graphs: tuple[Graph, ...]
    snapshots: tuple[Snapshot, ...]


def parse_states(directory: Path, source: str) -> tuple[Snapshot, ...]:
    """Select active rows by cell.src, independently of arena capacity or head."""
    result = []
    for state in json.loads(source):
        require(state["size"] == 1 and state["n_rs_seq"] == 0 and state["rs_idx"] == [0], "state.configuration")
        cells = state["cells"]
        require(len(cells) == state["size"] and [c["index"] for c in cells] == list(range(state["size"])), "state.cells")
        active = [c for c in cells if 0 in c["sequenceIds"]]
        require(len(active) == (state["position"] >= 0), "state.activeCell")
        row = (active[0]["src"] if active[0]["src"] >= 0 else active[0]["index"]) if active else None
        require(state["activeCell"] == (active[0]["index"] if active else None) and state["activeRow"] == row, "state.mapping")
        require(row is None or 0 <= row < state["size"] and active[0]["pos"] == state["position"], "state.sourceRow")
        descriptors = state["arrays"]
        require([(a["layer"], a["family"]) for a in descriptors] == [(i, f) for i in range(24) for f in ("R", "S")], "state.layers")
        observations = []
        for a in descriptors:
            width = {"R": 3, "S": 16}[a["family"]]
            require(a["shape"] == [1, 1536, width] and a["nativeShape"] == [1536 * width, 1, 1, 1], "state.shape")
            require(a["nativeStrides"] == [4, *([1536 * width * 4] * 3)], "state.strides")
            require(a["sourceRow"] == row and a["fullInitialBuffer"] == (row is None), "state.arrayMapping")
            require(a["snapshot"] == state["label"] and a["epoch"] == state["epoch"] and a["dtype"] == "f32", "state.epoch")
            require(a["axisLabels"] == ["cell" if row is None else "sequence", "inner", "history" if width == 3 else "state"], "state.axes")
            observed = read_array(directory, a)
            require(row is not None or not np.any(observed.values), "state.initialZero")
            observations.append(observed)
        mapping = json.dumps({key: state[key] for key in ("cells", "head", "activeCell", "activeRow", "position")}, sort_keys=True)
        result.append(Snapshot(state["label"], state["position"], state["epoch"], row, mapping, tuple(observations)))
    return tuple(result)


def check_rows(source: str, token_ids: tuple[int, ...], span: tuple[int, int]) -> None:
    """Parse strict integer row coordinates before comparing any numeric payload."""
    call, (begin, end) = json.loads(source), span
    fields = ("P", "T", "Q", "O", "inputBegin", "inputEndExclusive", "position", "sequenceId", "decodeExit", "previousPosition")
    require(all(type(call[k]) is int for k in fields) and [call[k] for k in fields] == [1, end-begin, end-begin, 1, begin, end, end-1, 0, 0, begin-1], "native.row")
    require(all(type(p) is int for p in call["positions"]) and call["positions"] == list(range(begin, end)), "native.positions")
    a = call["logits"]
    fields = ("inputIndex", "tokenId", "position", "outputRow", "batchTokenIndex", "vocabStart", "vocabEndExclusive", "sequenceId")
    require(all(type(a[k]) is int for k in fields) and [a[k] for k in fields] == [end-1, token_ids[end-1], end-1, 0, end-begin-1, 0, 50280, 0], "native.logits.rowVocabulary")
    require(a["shape"] == [50280] and a["axisLabels"] == ["vocabulary"] and a["dtype"] == "f32", "native.logits.axes")


def load_native(directory: Path, inputs: Inputs) -> Capture:
    """Require a completed immutable native run, including every graph/value event."""
    graph_module = importlib.import_module("tools.web_graph" if __package__ else "web_graph")
    require(not (directory / "incomplete.json").exists(), "native.incomplete")
    d, complete = provenance.read_json(directory / "capture.json"), provenance.read_json(directory / "complete.json")
    identity, run = d["identity"], d["run"]
    require(d["schemaVersion"] == 1 and d["rawFormat"] == "mamba-native-capture-v1", "native.format")
    require(d["status"] == complete["status"] == "captured-unvalidated", "native.status")
    require(provenance.sha256(directory / "capture.json") == complete["captureSha256"], "native.captureHash")
    require(provenance.sha256(directory / "arrays.bin") == complete["arraysSha256"] == identity["arraysSha256"], "native.archiveHash")
    def ordered_hash(value) -> str:
        return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    require(identity["runMetadataSha256"] == ordered_hash(run), "native.metadataHash")
    require(d["captureId"] == complete["captureId"] == "llama-" + ordered_hash(identity), "native.identity")
    require(identity["manifestSha256"] == inputs.manifest_sha and identity["preparationCaptureId"] == inputs.preparation_id
            and identity["ggufSha256"] == d["ggufSha256"] == inputs.gguf_sha, "native.manifest")
    require(provenance.json_hash(identity["source"]) == inputs.native_source_hash, "native.source")
    for path, digest in identity["build"]["files"].items():
        require(provenance.sha256(path) == digest, "native.buildHash")
    require(provenance.sha256(identity["build"]["compiler"]) == identity["build"]["compilerSha256"], "native.compiler")
    require(identity["environment"] == {k: "1" for k in ("GGML_CPU_DISABLE_FUSION", "LLAMA_GRAPH_REUSE_DISABLE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}, "native.environment")
    require(identity["positions"] == list(range(17)) and tuple(identity["tokenIds"]) in inputs.token_cases.values(), "native.fixture")
    require(provenance.sha256(identity["tokensPath"]) == identity["tokensSha256"], "native.fixtureHash")
    spans = {"fresh": [(0, 17)], "split": [(0, 16), (16, 17)]}[identity["schedule"]]
    require(identity["scheduleCalls"] == [list(s) for s in spans] and identity["scheduleSha256"] == ordered_hash(identity["scheduleCalls"]), "native.schedule")
    require(len(run["calls"]) == len(spans) and run["observerEnabled"] is True, "native.calls")
    require(run["effective"] == {"n_ctx": 256, "n_batch": 32, "n_ubatch": 32, "n_seq_max": 1, "n_rs_seq": 0,
        "n_threads": 1, "n_threads_batch": 1, "offload_kqv": False, "op_offload": False, "warmup": False, "backends": ["CPU"]}, "native.configuration")
    arrays, graphs = {}, []
    for call, (begin, end) in zip(run["calls"], spans):
        check_rows(json.dumps(call), tuple(identity["tokenIds"]), (begin, end))
        a = call["logits"]
        phase = "prefill" if end == 16 else "final"
        arrays[f"{identity['schedule']}.{phase}.logits"] = read_array(directory, a)
        values = {a["tensorId"]: read_array(directory, a) for a in call["arrays"]}
        require(len(values) == len(call["arrays"]), "native.arrayDuplicate")
        graph = graph_module.parse_graph(json.dumps(call), values)
        require(len(graph.nodes) == 1453 and graph.event_count == 2320, "native.graphInventory")
        output = next(t for t in graph.nodes if t.name == "result_output")
        require(values[output.id].values.tobytes() == arrays[f"{identity['schedule']}.{phase}.logits"].values.tobytes(), "native.logits.callback")
        tokens = next(t for t in graph.nodes if t.name == "inp_tokens")
        require(values[tokens.id].values.reshape(-1).tolist() == identity["tokenIds"][begin:end], "native.inputTokens")
        arrays.update({f"graph.{phase}.{key}": value for key, value in values.items()})
        graphs.append(graph)
    snapshots = parse_states(directory, json.dumps(run["snapshots"]))
    expected = [("before-prefill", -1), ("after-prefill", spans[0][1]-1)] + ([("before-decode", 15), ("after-decode", 16)] if len(spans) == 2 else [])
    require([(s.label, s.position) for s in snapshots] == expected, "native.stateSchedule")
    if len(spans) == 2:
        require(snapshots[1].mapping_json == snapshots[2].mapping_json and [a.sha256 for a in snapshots[1].arrays] == [a.sha256 for a in snapshots[2].arrays], "native.cacheContinuity")
    for snap in snapshots:
        phase = {"before-prefill": "before_prefill" if len(spans) == 2 else "before_final", "after-prefill": "after_prefill" if len(spans) == 2 else "after_final", "before-decode": "before_final", "after-decode": "after_final"}[snap.label]
        arrays.update({f"{identity['schedule']}.{phase}.{family}.{i}": snap.arrays[i * 2 + j] for i in range(24) for j, family in enumerate(("R", "S"))})
    runtime = provenance.json_hash({k: identity[k] for k in ("build", "environment", "modelConfigured", "systemInfo", "cpu", "source")})
    return Capture(directory, complete["captureSha256"], d["captureId"], complete["arraysSha256"], runtime,
                   identity["tokensSha256"], tuple(identity["tokenIds"]), MappingProxyType(arrays), tuple(graphs), snapshots)


def load_hf(directory: Path, inputs: Inputs) -> Capture:
    """Check the independent HF receipt, row semantics and deep-copy cache history."""
    d = provenance.read_json(directory / "reference.json")
    require(d["schemaVersion"] == 1 and d["numericalStatus"] == "unvalidated", "hf.format")
    require(d["captureId"] == "hf-" + provenance.json_hash({k: v for k, v in d.items() if k != "captureId"}), "hf.identity")
    require(d["manifestSha256"] == inputs.manifest_sha and d["preparationCaptureId"] == inputs.preparation_id and d["ggufSha256"] == inputs.gguf_sha, "hf.manifest")
    require(provenance.json_hash(d["environment"]) == inputs.environment_hash and provenance.json_hash(d["source"]) == inputs.source_hash and provenance.json_hash(d["config"]) == inputs.config_hash, "hf.configuration")
    require(provenance.sha256(directory / "arrays.npz") == d["archiveSha256"] and provenance.sha256(d["fixturePath"]) == d["fixtureSha256"], "hf.artifactHash")
    require(provenance.sha256(Path(__file__).with_name("reference_hf.py")) == d["scriptSha256"], "hf.harnessHash")
    r = d["runtime"]
    require(provenance.sha256(r["modelingPath"]) == r["modelingSha256"] == "4a5deb1cc9c22757826ac5dda343f5a0c3b928c50ac5d9c7c860e247bce06cdf", "hf.modelingHash")
    require([r[k] for k in ("branch", "dtype", "device", "inferenceMode", "eval", "optionalKernels", "threads", "interopThreads")] == ["MambaMixer.slow_forward", "float32", "cpu", True, True, False, 1, 1], "hf.runtime")
    tokens = tuple(d["fixture"]["tokenIds"])
    require(tokens in inputs.token_cases.values() and d["fixture"]["positions"] == list(range(17)), "hf.fixture")
    phases = ("fresh.before_final", "fresh.after_final", "split.before_prefill", "split.after_prefill", "split.before_final", "split.after_final")
    logits = {"fresh.final.logits": (0, 17), "split.prefill.logits": (0, 16), "split.final.logits": (16, 17)}
    expected = set(logits) | {f"{phase}.{family}" for phase in phases for family in ("conv", "R", "S")}
    arrays = {}
    with np.load(directory / "arrays.npz", allow_pickle=False) as archive:
        require(set(archive.files) == set(d["arrays"]) == expected, "hf.arrayInventory")
        for key in archive.files:
            value, meta = archive[key], d["arrays"][key]
            require(value.dtype == np.float32 and meta["dtype"] == "float32" and list(value.shape) == meta["shape"] and np.isfinite(value).all(), "hf.arrayLayout")
            payload = value.tobytes()
            require(hashlib.sha256(payload).hexdigest() == meta["sha256"], "hf.arrayHash")
            if key in logits:
                begin, end = logits[key]
                require(value.shape == (50280,) and [meta[k] for k in ("inputIndex", "tokenId", "sourceRow", "sequenceId", "axes", "inputPositions", "cachePositionArgument")] == [end-1, tokens[end-1], end-begin-1, 0, ["vocabularyTokenId"], list(range(begin, end)), [0, 1, 2, 3] if begin == 0 else [16]], "hf.rowVocabulary")
                arrays[key] = Observation(np.frombuffer(payload, dtype=np.float32).reshape(value.shape), meta["sha256"], 0)
            else:
                family = key.rsplit(".", 1)[1]
                width = {"conv": 4, "R": 3, "S": 16}[family]
                require(value.shape == (24, 1, 1536, width) and meta["axes"] == ["layer", "sequence", "intermediate", "state" if family == "S" else "history"], "hf.stateLayout")
                for layer in range(24):
                    part = value[layer].tobytes()
                    arrays[f"{key}.{layer}"] = Observation(np.frombuffer(part, dtype=np.float32).reshape(1, 1536, width), hashlib.sha256(part).hexdigest(), 0)
        for phase in phases:
            require(archive[phase + ".conv"][..., 1:].tobytes() == archive[phase + ".R"].tobytes(), "hf.historyMapping")
        for family in ("conv", "R", "S"):
            require(archive["split.after_prefill." + family].tobytes() == archive["split.before_final." + family].tobytes(), "hf.cacheContinuity")
            require(not np.any(archive["fresh.before_final." + family]) and not np.any(archive["split.before_prefill." + family]), "hf.initialZero")
    return Capture(directory, provenance.sha256(directory / "reference.json"), d["captureId"], d["archiveSha256"],
        provenance.json_hash(r), d["fixtureSha256"], tokens, MappingProxyType(arrays), (), ())
