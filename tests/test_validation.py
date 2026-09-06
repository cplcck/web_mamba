"""HF harness tests; task 7 adds numerical-policy comparator cases separately."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Final
import unittest
from unittest.mock import patch

from tools import reference_hf as hf

ROOT: Final = Path(__file__).resolve().parents[1]
FOUNDATION: Final = ROOT.parent / "mamba1-130m-visualizer-foundation"
MANIFEST: Final = FOUNDATION / ".artifacts/manifests/model.json"
TOKENS: Final = FOUNDATION / "fixtures/token-ids.json"


class HFReferenceCLI(unittest.TestCase):
    def test_exports_aligned_f32_rows_when_canonical_fixture_is_supplied(self) -> None:
        # Given: immutable real model and the predeclared 17 synthetic IDs.
        with tempfile.TemporaryDirectory(prefix="hf-test-") as temporary:
            out = Path(temporary) / "capture"
            command = [sys.executable, "-B", str(ROOT / "tools/reference_hf.py"),
                       "--manifest", str(MANIFEST), "--tokens", str(TOKENS), "--out", str(out)]
            # When: the actual reference CLI runs both schedules offline.
            result = subprocess.run(command, capture_output=True, text=True, timeout=120,
                                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
            # Then: rows identify the actual input and the full vocabulary axis.
            self.assertEqual(result.returncode, 0, result.stderr)
            receipt = json.loads((out / "reference.json").read_text())
            self.assertEqual(receipt["numericalStatus"], "unvalidated")
            self.assertEqual(receipt["fixtureSha256"], hf.prepare.sha256(TOKENS))
            for name, index, token in (("split.prefill.logits", 15, 31),
                                       ("split.final.logits", 16, 64),
                                       ("fresh.final.logits", 16, 64)):
                row = receipt["arrays"][name]
                self.assertEqual((row["inputIndex"], row["tokenId"], row["shape"], row["dtype"]),
                                 (index, token, [50280], "float32"))
                self.assertEqual(row["axes"], ["vocabularyTokenId"])
            with hf.np.load(out / "arrays.npz", allow_pickle=False) as arrays:
                for name in arrays.files:
                    array = arrays[name]
                    self.assertTrue(hf.np.isfinite(array).all(), name)
                    self.assertEqual(hashlib.sha256(array.tobytes()).hexdigest(), receipt["arrays"][name]["sha256"])
                    if name.endswith(".conv"):
                        hf.np.testing.assert_array_equal(array[..., 1:], arrays[name.removesuffix("conv") + "R"])
                for kind in ("conv", "R", "S"):
                    self.assertEqual(arrays["split.after_prefill." + kind].tobytes(),
                                     arrays["split.before_final." + kind].tobytes())
            self.assertTrue(receipt["prefillSnapshotImmutable"])

    def test_exits_nonzero_without_success_receipt_when_tokens_are_malformed(self) -> None:
        # Given: real immutable inputs, but one invalid fixture field.
        with tempfile.TemporaryDirectory(prefix="hf-invalid-") as temporary:
            fixture, out = Path(temporary) / "tokens.json", Path(temporary) / "out"
            document = json.loads(TOKENS.read_text())
            document["outputsPerCall"] = 2
            fixture.write_text(json.dumps(document))
            # When: drive the same real CLI failure surface.
            result = subprocess.run([sys.executable, "-B", str(ROOT / "tools/reference_hf.py"),
                "--manifest", str(MANIFEST), "--tokens", str(fixture), "--out", str(out)],
                capture_output=True, text=True, timeout=120)
            # Then: no misleading success or completed output accompanies the named error.
            self.assertEqual(result.returncode, 1)
            self.assertEqual(json.loads(result.stderr)["error"], "tokens.outputsPerCall")
            self.assertEqual(result.stdout, "")
            self.assertFalse((out / "reference.json").exists())


class HFReferenceBoundaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        hf.torch.set_num_threads(1)
        hf.torch.set_num_interop_threads(1)

    def test_rejects_schedule_corruption_when_an_otherwise_valid_fixture_changes(self) -> None:
        for field, value, invariant in (("positions", [False, *range(1, 17)], "positions"),
                ("positions", list(range(1, 18)), "positions"), ("prefillLength", 15, "prefillLength"),
                ("implicitSpecialTokens", True, "implicitSpecialTokens"), ("sequenceId", 1, "sequenceId"),
                ("tokenIds", [42] * 16, "tokenIds.length"), ("tokenIds", [50280] * 17, "tokenIds.vocabulary"),
                ("tokenIds", [True] * 17, "tokenIds.vocabulary")):
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory(prefix="hf-boundary-") as temporary:
                # Given: a known-valid control, mutated at exactly one field.
                hf.Schedule.parse(TOKENS, 50280)
                document = json.loads(TOKENS.read_text())
                document[field] = value
                path = Path(temporary) / "tokens.json"
                path.write_text(json.dumps(document))
                # When / Then: parse rejects the named schedule invariant.
                with self.assertRaises(hf.ReferenceError) as caught:
                    hf.Schedule.parse(path, 50280)
                self.assertEqual(caught.exception.field, "tokens." + invariant)

    def test_selects_last_vocabulary_row_when_each_row_is_distinct(self) -> None:
        for span in (range(17), range(16), range(16, 17)):
            # Given: deterministic distinct values on every input/vocabulary coordinate.
            logits = hf.torch.arange(len(span) * 7, dtype=hf.torch.float32).reshape(1, len(span), 7)
            schedule = hf.Schedule((1,) * 17, 7, json.dumps({**json.loads(TOKENS.read_text()), "tokenIds": [1] * 17}).encode())
            # When: select the output row from a valid schedule call.
            selected = hf.row(logits, span, schedule)
            # Then: vocabulary order is unchanged and the last local row is selected.
            self.assertEqual(selected.tolist(), list(range((len(span) - 1) * 7, len(span) * 7)))

    def test_rejects_row_corruption_when_shape_dtype_or_index_is_wrong(self) -> None:
        schedule = hf.Schedule((1,) * 17, 7, json.dumps({**json.loads(TOKENS.read_text()), "tokenIds": [1] * 17}).encode())
        good = hf.torch.arange(112, dtype=hf.torch.float32).reshape(1, 16, 7)
        for tensor, span, field in ((good[:, :, :-1], range(16), "vocabularyShape"),
                (good.transpose(1, 2), range(16), "vocabularyShape"),
                (good.double(), range(16), "dtypeDevice"), (good, range(1, 17), "inputIndex"),
                (good * float("nan"), range(16), "finite"), (good * float("inf"), range(16), "finite")):
            # Given: the control satisfies the same row contract.
            hf.row(good, range(16), schedule)
            # When / Then: corruption is rejected at the output boundary.
            with self.subTest(field=field), self.assertRaises(hf.ReferenceError) as caught:
                hf.row(tensor, span, schedule)
            self.assertEqual(caught.exception.field, "logits." + field)

    def test_copies_history_when_the_real_cache_is_mutated_in_place(self) -> None:
        # Given: real installed cache APIs with deterministic, nonzero initial contents.
        config = hf.mamba.MambaConfig(**json.loads(MANIFEST.read_text())["config"])
        cache = hf.mamba.MambaCache(config, 1, dtype=hf.torch.float32, device="cpu")
        for conv, ssm in zip(cache.conv_states, cache.ssm_states):
            conv.copy_(hf.torch.arange(4, dtype=hf.torch.float32).expand_as(conv))
            ssm.fill_(7)
        retained = hf.snapshot(cache)
        # When: the installed mutable cache API updates history and SSM storage.
        for layer in range(24):
            cache.update_conv_state(layer, hf.torch.full((1, 1536, 1), 9.0), hf.torch.tensor([16]))
            cache.update_ssm_state(layer, hf.torch.full((1, 1536, 16), 11.0))
        # Then: the retained snapshot has old bytes despite live-cache changes.
        self.assertEqual(retained[0][0, 0, 0].tolist(), [0, 1, 2, 3])
        self.assertTrue(bool((retained[1] == 7).all()))
        self.assertEqual(cache.conv_states[0][0, 0].tolist(), [1, 2, 3, 9])
        self.assertTrue(bool((cache.ssm_states[0] == 11).all()))

    def test_rejects_model_policy_corruption_when_dtype_or_eval_changes(self) -> None:
        for mutate, field in ((lambda model: model.double(), "execution.dtype.backbone.embeddings.weight"),
                              (lambda model: model.train(), "execution.eval")):
            # Given: each independent case has its own valid real model control.
            config = hf.mamba.MambaConfig(hidden_size=8, num_hidden_layers=1, vocab_size=16, state_size=2)
            model = hf.mamba.MambaForCausalLM(config).eval()
            with self.subTest(field=field), hf.torch.inference_mode():
                hf.verify_execution(model)
                mutate(model)
                # When / Then: the execution gate rejects this case-specific corruption.
                with self.assertRaises(hf.ReferenceError) as caught:
                    hf.verify_execution(model)
                self.assertEqual(caught.exception.field, field)

    def test_rejects_optional_kernel_when_an_optional_implementation_is_available(self) -> None:
        # Given: absence of optional implementations is the otherwise-valid control.
        hf.eager_only()
        with patch.object(hf.importlib.util, "find_spec", return_value=True):
            # When / Then: availability is rejected before model construction.
            with self.assertRaises(hf.ReferenceError) as caught:
                hf.eager_only()
            self.assertEqual(caught.exception.field, "optionalKernel.mamba_ssm")

    def test_rejects_installed_source_when_cache_implementation_hash_changes(self) -> None:
        # Given: a real valid model under the locked implementation.
        config = hf.mamba.MambaConfig(hidden_size=8, num_hidden_layers=1, vocab_size=16, state_size=2)
        model = hf.mamba.MambaForCausalLM(config).eval()
        with hf.torch.inference_mode():
            hf.verify_execution(model)
            with patch.object(hf.prepare, "sha256", return_value="changed-installed-source"):
                # When / Then: changed cache implementation cannot produce a canonical-layout claim.
                with self.assertRaises(hf.ReferenceError) as caught:
                    hf.verify_execution(model)
                self.assertEqual(caught.exception.field, "runtime.modelingSha256")

    def test_rejects_missing_snapshot_when_manifest_is_otherwise_valid(self) -> None:
        # Given: the successful CLI control's manifest, with only an absent snapshot path.
        document = json.loads(MANIFEST.read_text())
        with tempfile.TemporaryDirectory(prefix="hf-missing-") as temporary:
            document["snapshotPath"] = str(Path(temporary) / "absent")
            document["captureId"] = "prepare-" + hf.prepare.json_hash(
                {key: value for key, value in document.items() if key != "captureId"})
            with patch.object(hf.prepare, "read_json", return_value=document):
                # When / Then: the loader fails before any model load or comparison.
                with self.assertRaises(hf.ReferenceError) as caught:
                    hf.run(MANIFEST, TOKENS, Path(temporary) / "out")
                self.assertEqual(caught.exception.field, "manifest.snapshotPath")


class HFScheduleIntegration(unittest.TestCase):
    def test_binds_supplied_bytes_when_equivalent_json_uses_different_whitespace(self) -> None:
        # Given: equivalent valid schedules with different source bytes.
        with tempfile.TemporaryDirectory(prefix="hf-fixture-hash-") as temporary:
            path = Path(temporary) / "tokens.json"
            payload = json.dumps(json.loads(TOKENS.read_text()), separators=(",", ":")).encode()
            path.write_bytes(payload)
            # When: the fixture boundary consumes those exact bytes.
            parsed = hf.Schedule.parse(path, 50280)
            # Then: identity is the supplied artifact, not the canonical filename or reserialization.
            self.assertEqual(parsed.fixture_sha256, hashlib.sha256(payload).hexdigest())
            self.assertNotEqual(parsed.fixture_sha256, hf.prepare.sha256(TOKENS))

    def test_calls_real_model_with_exact_tokens_when_fresh_and_split_are_captured(self) -> None:
        # Given: the actual local model, with a transparent spy around its real forward.
        document = json.loads(MANIFEST.read_text())
        hf.torch.set_num_threads(1)
        if hf.torch.get_num_interop_threads() != 1:
            hf.torch.set_num_interop_threads(1)
        model = hf.mamba.MambaForCausalLM.from_pretrained(document["snapshotPath"],
            local_files_only=True, dtype=hf.torch.float32, use_safetensors=True).eval()
        schedule = hf.Schedule.parse(TOKENS, 50280)
        with tempfile.TemporaryDirectory(prefix="hf-real-calls-") as temporary:
            with hf.torch.inference_mode(), patch.object(model, "forward", wraps=model.forward) as calls:
                # When: the harness executes both schedules against real HF.
                hf.capture(model, schedule, Path(temporary) / "out")
            # Then: exactly 17, 16 and 1 supplied IDs enter forward, with split-only cache reuse.
            self.assertEqual([call.args[0].tolist() for call in calls.call_args_list],
                [[list(schedule.token_ids)], [list(schedule.token_ids[:16])], [[64]]])
            first, prefill, decode = calls.call_args_list
            self.assertIsNot(first.kwargs["cache_params"], prefill.kwargs["cache_params"])
            self.assertIs(prefill.kwargs["cache_params"], decode.kwargs["cache_params"])
            self.assertEqual([call.kwargs["cache_position"].tolist() for call in calls.call_args_list],
                             [[0, 1, 2, 3], [0, 1, 2, 3], [16]])


if __name__ == "__main__":
    unittest.main()
