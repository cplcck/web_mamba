"""Configuration-bound numerical comparisons; no implicit tolerance.

Use the manifest's locked Python interpreter; dependencies belong to preparation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
from typing import TYPE_CHECKING, Final, Literal

if TYPE_CHECKING:
    from tools import capture_data as data, export_web as export, web_graph
else:
    data = importlib.import_module("tools.capture_data" if __package__ else "capture_data")
    export = importlib.import_module("tools.export_web" if __package__ else "export_web")
    web_graph = importlib.import_module("tools.web_graph" if __package__ else "web_graph")

import numpy as np
from numpy.typing import NDArray


ValidationError = data.ValidationError
require = data.require
provenance = data.provenance
ROOT: Final = Path(__file__).resolve().parents[1]
AUDIT: Final = ROOT.parent / "mamba1-130m-visualizer-evidence/.artifacts/validation/artifact-audit.json"


def encoded(value) -> str:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


@dataclass(frozen=True, slots=True)
class Metrics:
    max_abs: float
    rmse: float
    nmse: float | Literal["infinity"]
    percentiles: tuple[float, float, float]
    violations: int | None

    @property
    def accepted(self) -> bool:
        return self.violations == 0 and self.nmse != "infinity"


def compare(reference: NDArray[np.floating], candidate: NDArray[np.floating],
            bounds: NDArray[np.floating] | None) -> Metrics:
    """Measure in F64; absent bounds never authorize numerical acceptance."""
    require(reference.shape == candidate.shape and reference.size > 0, "comparison.shape")
    require(bool(np.isfinite(reference).all() and np.isfinite(candidate).all()), "comparison.finite")
    error = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))
    squared, energy = float(np.square(error).sum()), float(np.square(reference.astype(np.float64)).sum())
    nmse = squared / energy if energy else (0.0 if squared == 0 else "infinity")
    violations = None
    if bounds is not None:
        require(bounds.shape == reference.shape, "policy.bounds.shape")
        require(bool(np.isfinite(bounds).all() and (bounds >= 0).all()), "policy.bounds.finiteNonnegative")
        violations = int(np.count_nonzero(error > bounds))
    p50, p95, p99 = (float(value) for value in np.percentile(error, [50, 95, 99]))
    return Metrics(float(error.max()), float(np.sqrt(squared / error.size)), nmse,
                   (p50, p95, p99), violations)


@dataclass(frozen=True, slots=True)
class Comparison:
    name: str
    reference: data.Observation
    candidate: data.Observation


@dataclass(frozen=True, slots=True)
class Corpus:
    comparisons: tuple[Comparison, ...]
    captures: tuple[data.Capture, ...]
    configuration_hash: str
    bindings_json: str


def collect(inputs: data.Inputs, directory: Path, audit_sha: str) -> Corpus:
    """Load the entire predeclared four-case, two-repeat, four-schedule matrix."""
    captures, comparisons, configurations, receipts = [], [], set(), []
    for case, token_ids in inputs.token_cases.items():
        first = {}
        for runtime, loader in (("hf", data.load_hf), ("llama-fresh", data.load_native), ("llama-split", data.load_native)):
            runs = [loader(directory / f"{case}-{repeat}-{runtime}", inputs) for repeat in (1, 2)]
            require(all(run.token_ids == token_ids for run in runs), "corpus.fixtureCase")
            require(runs[0].archive_sha == runs[1].archive_sha and {k: v.sha256 for k, v in runs[0].arrays.items()} == {k: v.sha256 for k, v in runs[1].arrays.items()}, "corpus.repeatability")
            for run in runs:
                for graph in run.graphs:
                    entities = web_graph.hierarchy(graph)
                    require(sum(e.kind == "block" for e in entities) == 24, "corpus.blockInventory")
                configurations.add((runtime, run.runtime_hash, provenance.json_hash([asdict(g) for g in run.graphs])))
                receipts.append({"case": case, "runtime": runtime, "captureId": run.capture_id, "receiptSha256": run.receipt_sha,
                    "archiveSha256": run.archive_sha, "runtimeHash": run.runtime_hash, "fixtureSha256": run.fixture_sha, "arrayCount": len(run.arrays)})
            captures.extend(runs)
            first[runtime] = runs[0]
        h, fresh, split = first["hf"].arrays, first["llama-fresh"].arrays, first["llama-split"].arrays
        for name, reference, candidate in (("H_F-L_F", h["fresh.final.logits"], fresh["fresh.final.logits"]),
                ("H_F-H_C", h["fresh.final.logits"], h["split.final.logits"]),
                ("L_F-L_C", fresh["fresh.final.logits"], split["split.final.logits"]),
                ("H_C-L_C", h["split.final.logits"], split["split.final.logits"]),
                ("prefill", h["split.prefill.logits"], split["split.prefill.logits"])):
            comparisons.append(Comparison(f"{case}/{name}/logits", reference, candidate))
        for family in ("R", "S"):
            for layer in range(24):
                key = f"{family}.{layer}"
                comparisons.append(Comparison(f"{case}/L_F-L_C/{key}", fresh["fresh.after_final."+key], split["split.after_final."+key]))
                comparisons.append(Comparison(f"{case}/H_F-H_C/{key}", h["fresh.after_final."+key], h["split.after_final."+key]))
    require(len(configurations) == 3, "corpus.configurationDrift")
    binding = {"manifestSha256": inputs.manifest_sha, "ggufSha256": inputs.gguf_sha, "auditSha256": audit_sha,
        "preparationCaptureId": inputs.preparation_id, "configurations": sorted(configurations), "captures": receipts}
    configuration_hash = provenance.json_hash({k: v for k, v in binding.items() if k != "captures"})
    return Corpus(tuple(comparisons), tuple(captures), configuration_hash, encoded(binding))


def candidate_policy(corpus: Corpus) -> str:
    """Propose measured envelopes only; this can never approve or freeze itself."""
    envelopes = {}
    for pair in corpus.comparisons:
        metrics = compare(pair.reference.values, pair.candidate.values, None)
        require(metrics.nmse != "infinity", "characterization.zeroEnergyFailure")
        envelopes[pair.name] = {"referenceSha256": pair.reference.sha256, "shape": list(pair.reference.values.shape),
            "absoluteBound": metrics.max_abs, "maxAbs": metrics.max_abs, "rmse": metrics.rmse,
            "nmse": metrics.nmse, "absolutePercentiles": metrics.percentiles}
    return encoded({"schemaVersion": 1, "status": "candidate", "configurationHash": corpus.configuration_hash,
        "auditSha256": json.loads(corpus.bindings_json)["auditSha256"], "envelopes": envelopes})


def trusted_approval(candidate_source: str) -> str:
    """Root-owned tracked configuration is authority; policy fields never select it."""
    anchor = ROOT / "fixtures/numerical-authorization.json"
    review = ROOT / "fixtures/numerical-review.json"
    require(anchor.resolve() == anchor and review.resolve() == review, "authorization.path")
    require(anchor.is_file(), "authorization.missing")
    authorization = provenance.read_json(anchor)
    require(authorization["schemaVersion"] == 1 and authorization["status"] == "approved", "authorization.status")
    require(review.is_file(), "authorization.reviewMissing")
    review_source = review.read_bytes()
    require(hashlib.sha256(review_source).hexdigest() == authorization["reviewSha256"], "authorization.reviewHash")
    verdict, candidate = json.loads(review_source), json.loads(candidate_source)
    require(candidate["schemaVersion"] == 1 and candidate["status"] == "candidate", "authorization.candidateStatus")
    require(hashlib.sha256(candidate_source.encode()).hexdigest() == authorization["candidateSha256"], "authorization.candidateHash")
    require(all(candidate[k] == authorization[k] for k in ("configurationHash", "auditSha256")), "authorization.binding")
    require(verdict["decision"] == "approved" and type(authorization["reviewerId"]) is str and bool(authorization["reviewerId"]), "authorization.reviewDecision")
    require(all(verdict[k] == authorization[k] for k in ("candidateSha256", "configurationHash", "auditSha256", "reviewerId")), "authorization.reviewBinding")
    return encoded({"receipt": str(review), "receiptSha256": authorization["reviewSha256"], "candidateSha256": authorization["candidateSha256"]})


def freeze_policy(candidate: Path, review: Path, destination: Path) -> None:
    """Copy measured envelopes only when the root's fixed authorization allows it."""
    source = candidate.read_text()
    approval = json.loads(trusted_approval(source))
    require(review.absolute() == Path(approval["receipt"]), "authorization.reviewPath")
    require(not destination.exists(), "policy.alreadyFrozen")
    policy = json.loads(source)
    policy.update(status="approved", approval=approval)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        stream.write(encoded(policy))


@dataclass(frozen=True, slots=True)
class Assessment:
    metrics: Metrics
    envelope_violations: tuple[str, ...]

    @property
    def accepted(self) -> bool:
        return self.metrics.accepted and not self.envelope_violations


def assess(pair: Comparison, specification: str | None) -> Assessment:
    """Exercise an envelope independently of, and never as, release authorization."""
    spec = json.loads(specification) if specification is not None else None
    bound = None
    if spec is not None:
        require(spec["referenceSha256"] == pair.reference.sha256 and spec["shape"] == list(pair.reference.values.shape), "policy.reference." + pair.name)
        require(all(type(spec[k]) in (int, float) and np.isfinite(spec[k]) and spec[k] >= 0 for k in ("absoluteBound", "maxAbs", "rmse", "nmse")), "policy.envelope")
        percentiles = spec["absolutePercentiles"]
        require(type(percentiles) is list and len(percentiles) == 3 and all(type(x) in (int, float) and np.isfinite(x) and x >= 0 for x in percentiles), "policy.percentiles")
        require(percentiles == sorted(percentiles), "policy.percentileOrder")
        bound = np.full(pair.reference.values.shape, spec["absoluteBound"], dtype=np.float64)
    result = compare(pair.reference.values, pair.candidate.values, bound)
    violations = []
    if spec is not None:
        for key, value in (("maxAbs", result.max_abs), ("rmse", result.rmse), ("nmse", result.nmse)):
            if value == "infinity" or value > spec[key]:
                violations.append(key)
        violations.extend(f"p{p}" for p, value, ceiling in zip((50, 95, 99), result.percentiles, spec["absolutePercentiles"]) if value > ceiling)
    return Assessment(result, tuple(violations))


def evaluate(corpus: Corpus, policy_source: str | None) -> str:
    """Check reviewed reference-bound component and aggregate envelopes, fail closed."""
    policy = json.loads(policy_source) if policy_source is not None else {}
    approval = policy.get("approval", {})
    approved = policy.get("status") == "approved"
    if approved:
        original = {**{k: v for k, v in policy.items() if k != "approval"}, "status": "candidate"}
        require(approval == json.loads(trusted_approval(encoded(original))), "policy.authorization")
        require(policy["configurationHash"] == corpus.configuration_hash, "policy.configuration")
        require(policy["auditSha256"] == json.loads(corpus.bindings_json)["auditSha256"], "policy.audit")
        require(set(policy["envelopes"]) == {p.name for p in corpus.comparisons}, "policy.inventory")
    metrics, failures = {}, []
    for pair in corpus.comparisons:
        spec = policy["envelopes"][pair.name] if approved else None
        assessment = assess(pair, encoded(spec) if spec is not None else None)
        result = assessment.metrics
        record = {"maxAbs": result.max_abs, "rmse": result.rmse, "nmse": result.nmse,
            "absolutePercentiles": result.percentiles, "componentViolations": result.violations,
            "referenceSha256": pair.reference.sha256, "candidateSha256": pair.candidate.sha256,
            "shape": list(pair.reference.values.shape), "absoluteBound": spec["absoluteBound"] if spec else None,
            "envelopeViolations": assessment.envelope_violations}
        if spec is not None and not assessment.accepted:
            failures.append(pair.name)
        if pair.name.endswith("/logits"):
            r, c = pair.reference.values.astype(np.float64), pair.candidate.values.astype(np.float64)
            ri, ci = np.lexsort((np.arange(r.size), -r)), np.lexsort((np.arange(c.size), -c))
            k = min(10, r.size)
            norm = float(np.linalg.norm(r) * np.linalg.norm(c))
            record["rankingDiagnostics"] = {"referenceTop1": int(ri[0]), "candidateTop1": int(ci[0]),
                "referenceMargin": float(r[ri[0]]-r[ri[1]]), "candidateMargin": float(c[ci[0]]-c[ci[1]]),
                "k": k, "topKOverlap": len(set(ri[:k]) & set(ci[:k])), "cosine": float(np.dot(r, c)/norm) if norm else None}
        metrics[pair.name] = record
    status = "unvalidated" if not approved else ("failed" if failures else "pass")
    return encoded({"schemaVersion": 1, "status": status, "configurationHash": corpus.configuration_hash,
        "bindings": json.loads(corpus.bindings_json), "metrics": metrics, "failures": failures,
        "policySha256": hashlib.sha256(policy_source.encode()).hexdigest() if policy_source is not None else None,
        "reviewSha256": approval.get("receiptSha256") if approved else None,
        "implementation": {"validator": provenance.sha256(Path(__file__)), "ingestion": provenance.sha256(Path(data.__file__)), "graph": provenance.sha256(Path(web_graph.__file__))}})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "captures", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    parser.add_argument("--policy", type=Path, default=ROOT / "fixtures/numerical-policy.json")
    parser.add_argument("--characterize", action="store_true")
    parser.add_argument("--candidate-policy", type=Path)
    args = parser.parse_args()
    try:
        require(not args.policy.resolve().is_relative_to(args.captures.resolve()), "policy.candidateControlled")
        inputs = export.verify_inputs(args.manifest, args.audit)
        corpus = collect(inputs, args.captures, provenance.sha256(args.audit))
        source = args.policy.read_text() if args.policy.is_file() and not args.characterize else None
        report = json.loads(evaluate(corpus, source))
        if args.characterize:
            require(args.candidate_policy is not None and not args.candidate_policy.exists(), "candidate.destination")
            require(not args.candidate_policy.resolve().is_relative_to(ROOT / "public"), "candidate.publicBoundary")
            args.candidate_policy.parent.mkdir(parents=True, exist_ok=True)
            with args.candidate_policy.open("x") as stream:
                stream.write(candidate_policy(corpus))
            provenance.atomic_json(args.out.with_name(args.out.stem + "-goldens.json"), {run.directory.name: {"captureId": run.capture_id, "arrays": {key: value.sha256 for key, value in run.arrays.items()}} for run in corpus.captures})
        provenance.atomic_json(args.out, report)
    except (ValidationError, ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(json.dumps({"status": "failed", "field": str(error)}), file=sys.stderr)
        return 1
    print(json.dumps({"status": report["status"], "reportSha256": provenance.sha256(args.out), "comparisons": len(corpus.comparisons)}))
    return 0 if report["status"] == "pass" else (2 if report["status"] == "unvalidated" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
