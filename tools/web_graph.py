"""Actual GGML graph ingestion and semantic ownership; no synthetic operators."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from graphlib import CycleError, TopologicalSorter
import importlib
import json
import math
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tools import capture_data as data
else:
    data = importlib.import_module("tools.capture_data" if __package__ else "capture_data")


@dataclass(frozen=True, slots=True)
class Storage:
    ggmlNbytes: int
    requiredAllocBytes: int | None
    bufferId: str | None
    bufferOffsetBytes: int | None
    bufferBytes: int | None
    observationEpoch: int
    allocatorSlotBytes: None
    allocatorSlotReason: str


@dataclass(frozen=True, slots=True)
class Node:
    id: str
    name: str
    op: str
    role: str
    dtype: str
    nativeShape: tuple[int, ...]
    strides: tuple[int, ...]
    numel: int
    typeBlockSize: int
    logicalBytes: int
    storage: Storage
    src: tuple[str | None, ...]
    viewSourceId: str | None
    viewOffsetBytes: int | None
    opParamsI32: tuple[int, ...]
    schedulerObserved: bool
    arithmeticExecution: bool


@dataclass(frozen=True, slots=True)
class Graph:
    nodes: tuple[Node, ...]
    event_count: int


def parse_graph(source: str, arrays: Mapping[str, data.Observation]) -> Graph:
    """Require reciprocal scheduler evidence, layouts, dependencies and value epochs."""
    raw = json.loads(source)
    nodes = []
    for t in raw["tensors"]:
        ne, nb, storage = t["nativeShape"], t["strides"], t["storage"]
        data.require(len(t["src"]) == 10 and len(t["opParamsI32"]) == 16, "graph.parameters")
        data.require(len(ne) == len(nb) == 4 and all(type(n) is int and 0 <= n < 2**53 for n in ne + nb), "graph.layout")
        data.require(t["dtype"] in ("f32", "i32") and t["typeSize"] == 4 and t["typeBlockSize"] == 1, "graph.dtype")
        numel = math.prod(ne)
        span = 4 + sum((n - 1) * stride for n, stride in zip(ne, nb)) if numel else 0
        data.require(t["numel"] == numel and t["logicalBytes"] == numel * 4 and storage["ggmlNbytes"] == span, "graph.bytes")
        data.require(t["role"] in ("weight", "activation", "state", "index") and t["classification"] == "observed", "graph.role")
        for key in ("id", "name", "op"):
            data.require(type(t[key]) is str and bool(t[key]), "graph." + key)
        for key in ("schedulerObserved", "arithmeticExecution", "metadataOnly"):
            data.require(type(t[key]) is bool, "graph." + key)
        data.require(t["schedulerObserved"] == (t["op"] != "NONE"), "graph.observed")
        arithmetic = t["op"] not in ("NONE", "VIEW", "RESHAPE", "TRANSPOSE", "PERMUTE") and numel > 0
        data.require(t["arithmeticExecution"] == arithmetic, "graph.arithmetic")
        data.require(storage["allocatorSlotBytes"] is None and bool(storage["allocatorSlotReason"]), "graph.allocatorSlot")
        for key in ("requiredAllocBytes", "bufferOffsetBytes", "bufferBytes", "observationEpoch"):
            data.require(storage[key] is None or type(storage[key]) is int and storage[key] >= 0, "graph.storage." + key)
        data.require(type(storage["observationEpoch"]) is int, "graph.epoch")
        data.require(storage["requiredAllocBytes"] is None or storage["requiredAllocBytes"] >= span, "graph.requiredAlloc")
        known = storage["bufferId"] is not None
        data.require(known == (storage["bufferOffsetBytes"] is not None) == (storage["bufferBytes"] is not None), "graph.buffer")
        if known:
            data.require(storage["bufferOffsetBytes"] + span <= storage["bufferBytes"], "graph.bufferRange")
        data.require((t["viewSourceId"] is None) == (t["viewOffsetBytes"] is None), "graph.viewOffset")
        data.require(t["viewSourceId"] is None or storage["requiredAllocBytes"] is None, "graph.viewAllocation")
        nodes.append(Node(t["id"], t["name"], t["op"], t["role"], t["dtype"].upper(), tuple(ne), tuple(nb), numel,
            1, numel * 4, Storage(**{k: storage[k] for k in Storage.__dataclass_fields__}), tuple(t["src"]),
            t["viewSourceId"], t["viewOffsetBytes"], tuple(t["opParamsI32"]), t["schedulerObserved"], arithmetic))
    by_id = {t.id: t for t in nodes}
    data.require(len(by_id) == len(nodes), "graph.duplicate")
    observed = [t for t in nodes if t.schedulerObserved]
    events = raw["events"]
    data.require(len(events) == len(observed) * 2, "graph.events.count")
    data.require([e["epoch"] for e in events] == sorted({e["epoch"] for e in events}), "graph.events.order")
    asks = events[::2]
    data.require([e["tensorId"] for e in asks] == [t.id for t in observed], "graph.events.nodes")
    for ask, complete in zip(asks, events[1::2]):
        node = by_id[ask["tensorId"]]
        data.require(ask["phase"] == "ask" and complete["phase"] == "complete" and complete["tensorId"] == node.id, "graph.events.phases")
        data.require(ask["epoch"] == node.storage.observationEpoch, "graph.events.epoch")
        if node.arithmeticExecution:
            data.require(node.id in arrays and arrays[node.id].epoch == complete["epoch"], "graph.values.epoch")
    for node in nodes:
        dependencies = [key for key in (*node.src, node.viewSourceId) if key is not None]
        data.require(all(key in by_id for key in dependencies), "graph.dependency")
        if node.id in arrays:
            array = arrays[node.id].values
            data.require(tuple(array.shape) == tuple(reversed(node.nativeShape)) and str(array.dtype).upper() == {"F32": "FLOAT32", "I32": "INT32"}[node.dtype], "graph.values.layout")
        if node.viewSourceId is not None:
            parent = by_id[node.viewSourceId]
            offset = node.viewOffsetBytes
            data.require(type(offset) is int and offset >= 0 and offset + max(node.logicalBytes, node.storage.ggmlNbytes) <= parent.storage.ggmlNbytes, "graph.viewRange")
            data.require(node.storage.bufferId == parent.storage.bufferId, "graph.viewBuffer")
            if parent.storage.bufferOffsetBytes is not None:
                data.require(node.storage.bufferOffsetBytes == parent.storage.bufferOffsetBytes + offset, "graph.viewBackingOffset")
    data.require(set(arrays) == {t.id for t in nodes if t.arithmeticExecution or t.op == "NONE" and t.dtype == "I32"}, "graph.values.inventory")
    try:
        TopologicalSorter({t.id: {s for s in (*t.src, t.viewSourceId) if s is not None} for t in nodes}).prepare()
    except CycleError as error:
        raise data.ValidationError("graph.cycle") from error
    return Graph(tuple(nodes), len(events))


@dataclass(frozen=True, slots=True)
class Entity:
    id: str
    kind: str
    parentId: str | None
    children: tuple[str, ...]
    inputTensorIds: tuple[str, ...]
    outputTensorIds: tuple[str, ...]
    weightTensorIds: tuple[str, ...]


def hierarchy(graph: Graph) -> tuple[Entity, ...]:
    """Trace each real block's anchors backwards; stop at previously owned boundaries.

    Anchor vocabulary follows pinned mamba.cpp/mamba-base.cpp. Metadata nodes and
    disconnected state copies are claimed through their actual dependencies, not
    by scheduler intervals (which interleave preparation of the following block).
    """
    by_id, by_name = {t.id: t for t in graph.nodes}, {t.name: t for t in graph.nodes}
    owners: dict[str, str] = {}
    parents = [("mamba-130m", "model", None)]

    def claim(stage: str, targets: list[Node]) -> None:
        data.require(bool(targets), "mapping.anchor." + stage)
        parents.append((stage, "stage", stage.split("/")[0] if stage.startswith("block.") else "mamba-130m"))
        pending = list(targets)
        while pending:
            node = pending.pop()
            if node.id in owners or not node.schedulerObserved:
                continue
            owners[node.id] = stage
            pending.extend(by_id[key] for key in node.src if key is not None)

    claim("model/embedding", [by_name["embd"]])
    layers = sorted(int(m[1]) for t in graph.nodes if (m := re.fullmatch(r"attn_norm-(\d+)", t.name)))
    data.require(layers == list(range(len(layers))) and bool(layers), "mapping.layers")
    for layer in layers:
        block = f"block.{layer}"
        parents.append((block, "block", "mamba-130m"))

        def weighted(suffix: str) -> list[Node]:
            return [t for t in graph.nodes if any(by_id[key].name == f"blk.{layer}.{suffix}" for key in t.src if key is not None)]

        claim(block + "/normalization", [by_name[f"attn_norm-{layer}"]])
        projected = weighted("ssm_in.weight")
        projected_ids = {t.id for t in projected}
        claim(block + "/input-projection-split", projected + [t for t in graph.nodes if t.viewSourceId in projected_ids])
        biased_ids = {t.id for t in weighted("ssm_conv1d.bias")}
        activated = [t for t in graph.nodes if t.op == "UNARY" and t.src[0] in biased_ids]
        conv_state = [t for t in graph.nodes if t.name.split(" ")[0] == f"cache_r_l{layer}" and t.schedulerObserved]
        claim(block + "/convolution-history", activated + conv_state)
        selected_ids = {t.id for t in weighted("ssm_x.weight")}
        claim(block + "/dt-b-c-projection", weighted("ssm_dt.bias") + [t for t in graph.nodes if t.viewSourceId in selected_ids])
        scans = weighted("ssm_a")
        data.require(len(scans) == 1 and scans[0].op == "SSM_SCAN", "mapping.scan")
        data.require(len(weighted("ssm_conv1d.weight")) == 1 and weighted("ssm_conv1d.weight")[0].op == "SSM_CONV", "mapping.conv")
        scan_state = [t for t in graph.nodes if t.name.split(" ")[0] == f"cache_s_l{layer}" and t.schedulerObserved]
        for family, candidates in (("R", conv_state), ("S", scan_state)):
            data.require(any(t.op == "CPY" and t.numel > 0 for t in candidates), "mapping.writeback." + family)
        claim(block + "/selective-scan-state", scans + scan_state)
        skips = {t.id for t in weighted("ssm_d")}
        gated = [t for t in graph.nodes if t.op == "GLU" and any(key in skips for key in by_id[t.src[1]].src)]
        claim(block + "/skip-gating", gated)
        outputs = weighted("ssm_out.weight")
        output_ids = {t.id for t in outputs}
        claim(block + "/output-projection", outputs + [t for t in graph.nodes if t.op == "RESHAPE" and t.viewSourceId in output_ids])
        claim(block + "/residual", [by_name[f"l_out-{layer}"]])
    claim("model/final-normalization", [by_name["result_norm"]])
    claim("model/final-projection", [by_name["result_output"]])
    data.require(set(owners) == {t.id for t in graph.nodes if t.schedulerObserved}, "mapping.unownedOperator")
    operators = []
    for index, node in enumerate(graph.nodes):
        if not node.schedulerObserved:
            continue
        stage = owners[node.id]
        inputs = tuple(dict.fromkeys(key for key in node.src if key is not None and by_id[key].role != "weight"))
        weights = tuple(dict.fromkeys(key for key in node.src if key is not None and by_id[key].role == "weight"))
        operators.append(Entity(f"{stage}/{node.op.lower()}.{index}", "operator", stage, (), inputs, (node.id,), weights))
    producers = {t.id: {e.id for e in operators if t.id in e.outputTensorIds} for t in graph.nodes}
    consumers = {t.id: {e.id for e in operators if t.id in (*e.inputTensorIds, *e.weightTensorIds)} for t in graph.nodes}
    result = []
    for identity, kind, parent in parents:
        members = [e for e in operators if identity == "mamba-130m" or e.parentId == identity or e.parentId.startswith(identity + "/")]
        member_ids = {e.id for e in members}
        input_ids, output_ids, weights = set(), set(), set()
        for entity in members:
            input_ids.update(key for key in entity.inputTensorIds if not producers[key] or not producers[key] <= member_ids)
            output_ids.update(key for key in entity.outputTensorIds if not consumers[key] or not consumers[key] <= member_ids)
            weights.update(entity.weightTensorIds)
        children = tuple(item[0] for item in parents if item[2] == identity) + tuple(e.id for e in operators if e.parentId == identity)
        result.append(Entity(identity, kind, parent, children, tuple(sorted(input_ids)), tuple(sorted(output_ids)), tuple(sorted(weights))))
    return tuple(result + operators)


@dataclass(frozen=True, slots=True)
class DocumentIdentity:
    capture_id: str
    gguf_sha: str
    scenario: str


def document(graph: Graph, identity: DocumentIdentity, payloads: Mapping[str, tuple[int, int]]) -> str:
    """Serialize all actual layouts and metadata, never their retained raw arrays."""
    data.require(identity.scenario in ("prefill", "decode"), "document.scenario")
    entities = hierarchy(graph)
    operators = [e for e in entities if e.kind == "operator"]
    exporter = importlib.import_module("tools.export_web" if __package__ else "export_web")
    shapes = json.loads(exporter.symbolic_shapes(graph))
    dimensions = {"P": 1, "T": 16 if identity.scenario == "prefill" else 1, "Q": 16 if identity.scenario == "prefill" else 1, "O": 1}
    tensors = []
    for node in graph.nodes:
        tensor = asdict(node)
        shape = shapes[node.id]
        data.require(exporter.formula_values(json.dumps(shape), dimensions) == tuple(reversed(node.nativeShape)), "document.formulaShape." + node.id)
        byte_formula = {"op": "const", "value": 4}
        for axis in shape:
            byte_formula = {"op": "mul", "left": byte_formula, "right": axis}
        symbolic = '"dim"' in json.dumps(shape)
        tensor.update(logicalShape=list(reversed(node.nativeShape)), axisLabels=["ne3", "ne2", "ne1", "ne0"],
            producerIds=[e.id for e in operators if node.id in e.outputTensorIds],
            consumerIds=[e.id for e in operators if node.id in (*e.inputTensorIds, *e.weightTensorIds)],
            captureId=identity.capture_id, classification="observed",
            provenance=("scheduler-ask" if node.schedulerObserved else "dependency-leaf") + f":epoch:{node.storage.observationEpoch}",
            shapeFormula=shape, logicalBytesFormula=byte_formula,
            formulaClassification="symbolic-estimate" if symbolic else "observed-constant",
            formulaSource="src/models/mamba-base.cpp:43-150; src/models/mamba.cpp:91-136 at 8144f319" if symbolic else "fixed weight/capacity/history layout or observation-only metadata; no allocation estimate")
        if node.role == "weight":
            data.require(node.name in payloads, "document.weightPayload")
            offset, size = payloads[node.name]
            data.require(size == node.logicalBytes, "document.weightPayloadSize")
            tensor["ggufPayload"] = {"fileId": "gguf:" + identity.gguf_sha, "offsetBytes": offset, "bytes": size}
        tensors.append(tensor)
    return json.dumps({"schemaVersion": 1, "captureId": identity.capture_id, "ggufSha256": identity.gguf_sha,
        "scenario": {"name": identity.scenario, "dimensions": {"P": 1, "T": 16 if identity.scenario == "prefill" else 1, "Q": 16 if identity.scenario == "prefill" else 1, "O": 1}},
        "entities": [asdict(e) for e in entities], "tensors": tensors,
        "symbolicSupport": json.loads(exporter.symbolic_support(identity.scenario)),
        "rawCoverage": {"tensorCount": len(graph.nodes), "schedulerNodeCount": len(operators),
            "leafCount": len(graph.nodes)-len(operators), "nodeToEntity": {e.outputTensorIds[0]: e.id for e in operators}},
        "formulaClassification": "source-derived payload estimates; changed dimensions and all storage allocation remain unvalidated",
        "kernelInternals": {"SSM_CONV": {"classification": "kernel-internal", "formula": "sum(history * kernel)"},
            "SSM_SCAN": {"classification": "kernel-internal", "formula": "s = exp(softplus(dt) * A) * s + B * x * softplus(dt); y = sum(s * C)"}},
        "offlineConversion": "A_log -> -exp(A_log); not a runtime EXP node"}, separators=(",", ":"), allow_nan=False)
