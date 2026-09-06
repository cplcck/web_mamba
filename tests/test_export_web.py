"""Evidence ingestion and publication rejection tests, using tiny local fixtures."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from tools import capture_data as data
from tests.validation_fixtures import array_fixture


class ReleasePolicy(unittest.TestCase):
    def test_remains_unvalidated_when_policy_is_missing_or_unreviewed(self) -> None:
        import json
        from tools.validate import evaluate
        from tests.validation_fixtures import corpus_fixture
        # Given: finite observations, but no independently approved policy.
        for policy in (None, '{"status":"candidate","absoluteBound":1e30}'):
            with self.subTest(policy=policy):
                # When: evaluate the release gate, not just component metrics.
                result = json.loads(evaluate(corpus_fixture(), policy))
                # Then: a candidate cannot authorize or widen its own acceptance.
                self.assertEqual(result["status"], "unvalidated")


class ReportFreshness(unittest.TestCase):
    def test_rejects_a_changed_metric_when_report_is_checked_against_current_arrays(self) -> None:
        import json
        from tools.validate import evaluate
        from tools.export_web import checked_report
        from tests.validation_fixtures import corpus_fixture
        # Given: a report whose displayed error was altered after comparison.
        corpus = corpus_fixture()
        stale = json.loads(evaluate(corpus, None))
        stale["metrics"][corpus.comparisons[0].name]["maxAbs"] = 0
        # When / Then: recomputation rejects the stale report even without approval.
        with self.assertRaises(data.ValidationError) as caught:
            checked_report(corpus, None, json.dumps(stale))
        self.assertEqual(caught.exception.field, "export.staleReport")


class EnvelopeRejections(unittest.TestCase):
    def test_rejects_one_ulp_outside_when_the_component_and_rmse_bounds_are_frozen(self) -> None:
        from dataclasses import replace
        import hashlib
        import numpy as np
        from tools.validate import assess
        from tests.validation_fixtures import corpus_fixture, envelope_fixture
        # Given: an explicitly bounded tiny array; no simulated numerical approval.
        pair = corpus_fixture().comparisons[0]
        self.assertTrue(assess(pair, envelope_fixture()).accepted)
        values = pair.candidate.values.copy()
        values[1] = np.nextafter(values[1], np.float32(np.inf))
        payload = values.tobytes()
        changed = replace(pair, candidate=data.Observation(np.frombuffer(payload, dtype="<f4"), hashlib.sha256(payload).hexdigest(), 0))
        # When: one finite component moves one ULP beyond the explicit envelope.
        result = assess(changed, envelope_fixture())
        # Then: finite values and unchanged top-1 cannot excuse the component error.
        self.assertFalse(result.accepted)
        self.assertEqual(result.metrics.violations, 1)
        self.assertIn("rmse", result.envelope_violations)

    def test_rejects_invalid_envelopes_when_reference_or_percentile_fields_are_corrupt(self) -> None:
        import json
        from tools.validate import assess
        from tests.validation_fixtures import corpus_fixture, envelope_fixture
        # Given / When / Then: malformed or differently bound specifications never authorize arrays.
        for field, value in (("referenceSha256", "0" * 64), ("absolutePercentiles", []),
                             ("absolutePercentiles", [1, 1, float("inf")]), ("absoluteBound", -1)):
            with self.subTest(field=field, value=value):
                spec = json.loads(envelope_fixture())
                spec[field] = value
                with self.assertRaises(data.ValidationError):
                    assess(corpus_fixture().comparisons[0], json.dumps(spec))


class CorruptionRejections(unittest.TestCase):
    def test_rejects_bad_array_descriptors_when_disk_and_metadata_disagree(self) -> None:
        import hashlib
        import numpy as np
        # Given / When / Then: actual disk bytes are checked, including rehashed NaNs.
        with tempfile.TemporaryDirectory(prefix="array-rejections-") as temporary:
            directory = Path(temporary)
            for field, value in (("sha256", "0" * 64), ("shape", [5]), ("file", "../arrays.bin"),
                                 ("offsetBytes", 16), ("epoch", -1), ("dtype", "f64"), ("axisLabels", [])):
                with self.subTest(field=field):
                    descriptor = array_fixture(directory)
                    descriptor[field] = value
                    with self.assertRaises(data.ValidationError):
                        data.read_array(directory, descriptor)
            descriptor = array_fixture(directory)
            payload = np.array([1, np.nan, 4, 8], dtype="<f4").tobytes()
            (directory / "arrays.bin").write_bytes(payload)
            descriptor["sha256"] = hashlib.sha256(payload).hexdigest()
            with self.assertRaises(data.ValidationError) as caught:
                data.read_array(directory, descriptor)
            self.assertEqual(caught.exception.field, "array.finite")

    def test_rejects_graph_corruption_when_events_nodes_aliases_or_edges_change(self) -> None:
        import json
        from tools.web_graph import parse_graph
        from tests.validation_fixtures import graph_fixture
        # Given: valid arithmetic bytes; corruption affects actual graph metadata.
        with tempfile.TemporaryDirectory(prefix="graph-rejections-") as temporary:
            directory = Path(temporary)
            descriptor = array_fixture(directory)
            descriptor.update(shape=[1, 1, 1, 4], axisLabels=["ne3", "ne2", "ne1", "ne0"], epoch=4)
            arrays = {"t1": data.read_array(directory, descriptor)}
            for mutation in ("dropped-node", "dropped-event", "self-cycle", "short-sources", "view-range", "stale-epoch"):
                with self.subTest(mutation=mutation):
                    source = json.loads(graph_fixture())
                    if mutation == "dropped-node":
                        source["tensors"].pop()
                    elif mutation == "dropped-event":
                        source["events"].pop()
                    elif mutation == "self-cycle":
                        source["tensors"][1]["src"][0] = "t1"
                    elif mutation == "short-sources":
                        source["tensors"][1]["src"] = ["t0"]
                    elif mutation == "view-range":
                        source["tensors"][2]["viewOffsetBytes"] = 20
                    elif mutation == "stale-epoch":
                        source["events"][1]["epoch"] = 5
                    # When / Then: no valid payload can hide incomplete or impossible dependencies.
                    with self.assertRaises(data.ValidationError):
                        parse_graph(json.dumps(source), arrays)

    def test_rejects_state_mapping_when_an_active_row_or_layer_is_wrong(self) -> None:
        import json
        from tests.validation_fixtures import state_fixture
        # Given / When / Then: explicit mapping and full layer inventory cannot be omitted or shifted.
        with tempfile.TemporaryDirectory(prefix="mapping-rejections-") as temporary:
            directory = Path(temporary)
            for field in ("activeRow", "sourceRow", "layer", "sequenceIds"):
                with self.subTest(field=field):
                    states = json.loads(state_fixture(directory))
                    if field == "activeRow":
                        states[1][field] = 1
                    elif field in ("sourceRow", "layer"):
                        states[1]["arrays"][0][field] = 1
                    elif field == "sequenceIds":
                        states[1]["cells"][0][field] = []
                    with self.assertRaises(data.ValidationError):
                        data.parse_states(directory, json.dumps(states))

    def test_rejects_shifted_logits_when_row_or_vocabulary_metadata_changes(self) -> None:
        import json
        from tests.validation_fixtures import call_fixture
        # Given: an explicit final-input row, not generated text or a top-1 substitute.
        data.check_rows(call_fixture(), tuple(range(17)), (0, 17))
        for field, value in (("inputIndex", 15), ("batchTokenIndex", 15), ("vocabStart", 1),
                             ("vocabEndExclusive", 50279), ("outputRow", False)):
            with self.subTest(field=field):
                source = json.loads(call_fixture())
                source["logits"][field] = value
                # When / Then: reject alignment errors before any numerical tolerance.
                with self.assertRaises(data.ValidationError):
                    data.check_rows(json.dumps(source), tuple(range(17)), (0, 17))


class PublicationGate(unittest.TestCase):
    def test_rejects_self_approval_and_rehashed_policy_through_real_cli(self) -> None:
        from tests.validation_fixtures import approval_attack_results
        import json
        # Given: candidate-owned receipts and unchanged real capture archives.
        with tempfile.TemporaryDirectory(prefix="approval-attacks-") as temporary:
            # When: drive both attacks that independently obtained unauthorized PASS.
            results = approval_attack_results(Path(temporary))
            print(json.dumps({"authorizationAttacks": results}))
            # Then: changing candidate approval bytes never grants release authority.
            for result in results:
                with self.subTest(mutation=result["mutation"]):
                    self.assertNotEqual(result["exit"], 0, result)
                    self.assertEqual(result["stdout"], "")
                    self.assertIn(json.loads(result["stderr"])["field"],
                                  ("authorization.missing", "authorization.status", "authorization.candidateHash", "policy.authorization"))

    def test_refuses_publication_when_a_complete_bundle_is_unvalidated(self) -> None:
        import json
        from tools.export_web import write_bundle
        # Given: a complete metadata-only inventory without numerical approval.
        files = {name + ".json": json.dumps({"numericalStatus": "unvalidated"})
                 for name in ("manifest", "model", "provenance", "validation", "prefill", "decode")}
        with tempfile.TemporaryDirectory(prefix="publication-gate-") as temporary:
            destination = Path(temporary) / "out"
            # When / Then: neither ordinary export nor an out-of-scope draft writes files.
            for draft, field in ((False, "export.unvalidated"), (True, "export.draftBoundary")):
                with self.subTest(draft=draft), self.assertRaises(data.ValidationError) as caught:
                    write_bundle(files, destination, draft)
                self.assertEqual(caught.exception.field, field)
            self.assertFalse(destination.exists())

    def test_freezes_only_when_fixed_authority_review_is_approved(self) -> None:
        import json
        from tests.validation_fixtures import authority_fixture
        for case, field in (("approved", None), ("rejected", "authorization.reviewDecision"), ("missing", "authorization.reviewMissing")):
            with self.subTest(case=case), tempfile.TemporaryDirectory(prefix="freeze-gate-") as temporary:
                # Given: an unchanged copied validator and otherwise-valid synthetic fixed authority.
                root = Path(temporary)
                numeric, candidate, review = authority_fixture(root, "rejected" if case == "rejected" else "approved")
                destination = root / "synthetic-frozen.json"
                if case == "missing":
                    review.unlink()
                # When / Then: approved control freezes; each negative reaches its intended gate.
                if field is None:
                    numeric.freeze_policy(candidate, review, destination)
                    frozen = json.loads(destination.read_text())
                    self.assertEqual(frozen["status"], "approved")
                    self.assertEqual(frozen["envelopes"], json.loads(candidate.read_text())["envelopes"])
                else:
                    with self.assertRaises(numeric.ValidationError) as caught:
                        numeric.freeze_policy(candidate, review, destination)
                    self.assertEqual(caught.exception.field, field)
                    self.assertFalse(destination.exists())


class PublicationInputs(unittest.TestCase):
    def test_rejects_stale_manifest_when_its_identity_no_longer_matches(self) -> None:
        import json
        from tools.export_web import verify_inputs
        # Given: a disk manifest claiming an identity unrelated to its actual content.
        with tempfile.TemporaryDirectory(prefix="manifest-input-") as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"schemaVersion": 1, "captureId": "prepare-" + "0" * 64}))
            # When / Then: fail at identity before trusting any paths or audit claim.
            with self.assertRaises(data.ValidationError) as caught:
                verify_inputs(manifest, root / "audit.json")
            self.assertEqual(caught.exception.field, "manifest.identity")


class ArrayIngestion(unittest.TestCase):
    def test_loads_exact_immutable_values_when_descriptor_matches_disk(self) -> None:
        # Given: an independently specified tiny binary array on real disk.
        with tempfile.TemporaryDirectory(prefix="web-array-") as temporary:
            directory = Path(temporary)
            descriptor = array_fixture(directory)
            # When: the ingestion boundary consumes its descriptor and bytes.
            observation = data.read_array(directory, descriptor)
            # Then: correct immutable values, not merely a file-exists success.
            self.assertEqual(observation.values.tolist(), [1, 2, 4, 8])
            self.assertFalse(observation.values.flags.writeable)


class GraphIngestion(unittest.TestCase):
    def test_preserves_empty_view_when_every_actual_node_is_observed(self) -> None:
        from tools.web_graph import parse_graph
        from tests.validation_fixtures import graph_fixture
        # Given: independently constructed complete graph and its arithmetic payload.
        with tempfile.TemporaryDirectory(prefix="graph-input-") as temporary:
            directory = Path(temporary)
            descriptor = array_fixture(directory)
            descriptor.update(shape=[1, 1, 1, 4], axisLabels=["ne3", "ne2", "ne1", "ne0"], epoch=4)
            array = data.read_array(directory, descriptor)
            # When: parse the scheduler graph, including its empty view.
            graph = parse_graph(graph_fixture(), {"t1": array})
            # Then: leaves and zero-element nodes remain present and distinguishable.
            self.assertEqual(len(graph.nodes), 3)
            self.assertEqual(graph.nodes[2].numel, 0)
            self.assertEqual(graph.nodes[2].viewSourceId, "t1")


class DocumentExport(unittest.TestCase):
    def test_exports_reciprocal_sections_when_tensors_are_serialized(self) -> None:
        import json
        from tools.web_graph import DocumentIdentity, document
        from tests.validation_fixtures import hierarchy_fixture
        # Given: an explicit tiny graph with independent GGUF payload ranges.
        graph = hierarchy_fixture()
        weights = {n.name: (i * 4, 4) for i, n in enumerate(graph.nodes) if n.role == "weight"}
        # When: export a complete scenario document, not a drawing substitute.
        result = json.loads(document(graph, DocumentIdentity("test", "a" * 64, "prefill"), weights))
        # Then: every original tensor and separate section survives serialization.
        self.assertEqual(len(result["tensors"]), len(graph.nodes))
        self.assertTrue(all(set(("inputTensorIds", "outputTensorIds", "weightTensorIds")) <= e.keys() for e in result["entities"]))
        self.assertTrue(all(t["classification"] == "observed" for t in result["tensors"]))


class StateIngestion(unittest.TestCase):
    def test_selects_the_observed_active_row_when_fresh_state_is_parsed(self) -> None:
        from tests.validation_fixtures import state_fixture
        # Given: source-shaped zero initialization and explicit active-cell selection.
        with tempfile.TemporaryDirectory(prefix="state-input-") as temporary:
            directory = Path(temporary)
            source = state_fixture(directory)
            # When: ingest the actual mapping contract, not raw capacity buffers.
            snapshots = data.parse_states(directory, source)
            # Then: all 24 R/S histories retain the selected active row.
            self.assertIsNone(snapshots[0].active_row)
            self.assertEqual(snapshots[1].active_row, 0)
            self.assertEqual(len(snapshots[1].arrays), 48)


class HierarchyExport(unittest.TestCase):
    def test_maps_every_actual_operator_when_a_complete_block_is_exported(self) -> None:
        from tools.web_graph import hierarchy
        from tests.validation_fixtures import hierarchy_fixture
        # Given: a small explicit real-operator topology with residual and state edges.
        graph = hierarchy_fixture()
        # When: derive semantic ownership from its actual dependencies.
        entities = hierarchy(graph)
        # Then: no operator is substituted, duplicated, or dropped.
        operators = [e for e in entities if e.kind == "operator"]
        self.assertEqual(len(operators), sum(n.schedulerObserved for n in graph.nodes))
        self.assertEqual(len([e for e in entities if e.kind == "stage" and e.parentId == "block.0"]), 8)
        self.assertEqual(len({e.id for e in operators}), len(operators))


if __name__ == "__main__":
    unittest.main()
