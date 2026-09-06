"""Public provenance privacy regression with isolated export inputs."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from tools import capture_data as data


class PublicProvenance(unittest.TestCase):
    def test_omits_private_source_status_when_rendering_public_provenance(self) -> None:
        import json
        from unittest.mock import patch
        from tools import export_web
        from tools.validate import Corpus
        from tests.validation_fixtures import hierarchy_fixture
        # Given: tiny disk-backed export inputs with private raw source status.
        with tempfile.TemporaryDirectory(prefix="source-projection-") as temporary:
            root = Path(temporary)
            identity = {
                "commit": "a" * 40, "runtimeCommit": "a" * 40,
                "converterSha256": "b" * 64, "mambaConverterSha256": "c" * 64,
                "productPatchSha256": "d" * 64, "worktreeDiffSha256": "e" * 64,
                "tree": "f" * 40,
            }
            private_source = {
                **identity, "path": "/home/private/source",
                "status": "?? .omo/senpi-task/private.rawstatus\n",
            }
            manifest = root / "manifest.json"
            payload = root / "private.gguf"
            payload.write_bytes(b"fixture")
            manifest.write_text(json.dumps({
                "source": private_source, "environment": {"pythonExecutable": "/private/python"},
                "repoId": "state-spaces/mamba-130m-hf", "revision": "1" * 40,
                "config": {}, "ggufPath": str(payload),
            }))
            directory = root / "canonical-1-llama-split"
            directory.mkdir()
            directory.joinpath("capture.json").write_text(json.dumps({
                "identity": {
                    "build": {"compilerId": "fixture", "compilerVersion": "1",
                              "compilerSha256": "2" * 64, "files": {}, "cmakeFlags": {}},
                    "environment": {}, "systemInfo": "fixture",
                },
                "run": {"configured": {}, "effective": {}},
            }))
            graph = hierarchy_fixture()
            weights = [node for node in graph.nodes if node.role == "weight"]
            audit = root / "audit.json"
            audit.write_text(json.dumps({"audit": {
                "tensors": [
                    {"targetName": node.name, "targetDtype": node.dtype,
                     "nativeShape": list(node.nativeShape), "payloadOffset": index * 4,
                     "payloadBytes": 4}
                    for index, node in enumerate(weights)
                ],
                "blockCount": 1, "vocabularySize": 1, "tokenizerVocabularySize": 1,
                "paddedTokenIds": [], "uniquePayloadBytes": len(weights) * 4,
                "targetTensorCount": len(weights), "outputBinding": {},
            }}))
            snapshots = tuple(data.Snapshot(label, position, epoch, 0, "{}", ())
                              for label, position, epoch in (
                                  ("before-prefill", -1, 1), ("after-prefill", 15, 2),
                                  ("before-decode", 15, 3), ("after-decode", 16, 4)))
            capture = data.Capture(directory, "receipt", "native", "archive", "runtime",
                                   "fixture", (), {}, (graph, graph), snapshots)
            corpus = Corpus((), (capture,), "configuration", "{}")
            inputs = data.Inputs(manifest, "manifest", "prepare", "3" * 64,
                                 "environment", "config", "source", "native-source", {})
            report = json.dumps({"status": "unvalidated", "policySha256": "4" * 64,
                                 "reviewSha256": "5" * 64, "bindings": {"auditSha256": "6" * 64}})
            # When: render the real publication projection against the isolated audit.
            with patch.object(export_web, "AUDIT", audit):
                files = export_web.render(corpus, inputs, report)
            # Then: retain every source identity field, omit paths/status, preserve input.
            self.assertEqual(json.loads(files["provenance.json"])["source"], identity)
            self.assertNotIn("private.rawstatus", files["provenance.json"])
            self.assertEqual(json.loads(manifest.read_text())["source"], private_source)
