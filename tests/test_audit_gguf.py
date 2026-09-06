"""Independent synthetic safetensors/GGUF pairs; no converter calls."""
from __future__ import annotations
import importlib
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
import numpy as np
from safetensors.numpy import save_file
from tokenizers import AddedToken, Tokenizer, decoders, models, normalizers, pre_tokenizers, processors
from transformers import MambaConfig
sys.path.insert(0, "/home/cplcck/llama.cpp-ssm/gguf-py")
from gguf import GGUFWriter


class AuditTests(unittest.TestCase):
    def setUp(self) -> None:
        # Given: source-valid one-block model, with explicit tied output.
        artifacts = Path(__file__).resolve().parents[1] / ".artifacts"
        artifacts.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="mamba-audit-", dir=artifacts)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.snapshot = self.root / "snapshot"
        self.snapshot.mkdir()
        MambaConfig(vocab_size=8, hidden_size=2, intermediate_size=4,
                    num_hidden_layers=1, state_size=2, conv_kernel=3,
                    time_step_rank=1).save_pretrained(self.snapshot)
        tokenizer = Tokenizer(models.BPE({"<|endoftext|>": 0, "a": 1, "b": 2, "ab": 3}, [("a", "b")]))
        tokenizer.normalizer = normalizers.NFC()
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.add_special_tokens([AddedToken("<|endoftext|>", normalized=False, special=True)])
        tokenizer.add_tokens([AddedToken("  ", normalized=True)])
        tokenizer.save(str(self.snapshot / "tokenizer.json"))
        (self.snapshot / "tokenizer_config.json").write_text(json.dumps({
            "tokenizer_class": "GPTNeoXTokenizer", "bos_token": "<|endoftext|>",
            "eos_token": "<|endoftext|>", "unk_token": "<|endoftext|>",
            "pad_token": "<|endoftext|>", "add_prefix_space": False,
        }))
        shapes = {
            "backbone.embeddings.weight": (8, 2), "backbone.norm_f.weight": (2,),
            "backbone.layers.0.norm.weight": (2,),
            "backbone.layers.0.mixer.in_proj.weight": (8, 2),
            "backbone.layers.0.mixer.conv1d.weight": (4, 1, 3),
            "backbone.layers.0.mixer.conv1d.bias": (4,),
            "backbone.layers.0.mixer.x_proj.weight": (5, 4),
            "backbone.layers.0.mixer.dt_proj.weight": (4, 1),
            "backbone.layers.0.mixer.dt_proj.bias": (4,),
            "backbone.layers.0.mixer.A_log": (4, 2),
            "backbone.layers.0.mixer.D": (4,),
            "backbone.layers.0.mixer.out_proj.weight": (2, 4),
        }
        self.source = {name: np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 16
                       for name, shape in shapes.items()}
        self.source["lm_head.weight"] = self.source["backbone.embeddings.weight"].copy()
        names = ["token_embd.weight", "output_norm.weight", "blk.0.attn_norm.weight",
                 "blk.0.ssm_in.weight", "blk.0.ssm_conv1d.weight", "blk.0.ssm_conv1d.bias",
                 "blk.0.ssm_x.weight", "blk.0.ssm_dt.weight", "blk.0.ssm_dt.bias",
                 "blk.0.ssm_a", "blk.0.ssm_d", "blk.0.ssm_out.weight"]
        self.target = {target: self.source[source].copy() for source, target in zip(shapes, names)}
        self.target["blk.0.ssm_conv1d.weight"] = self.target["blk.0.ssm_conv1d.weight"][:, 0, :]
        self.target["blk.0.ssm_a"] = np.array([
            -1., -1.0644944906234741, -1.133148431777954, -1.2062302827835083,
            -1.2840254306793213, -1.366837978363037, -1.4549914598464966,
            -1.548830270767212], dtype=np.float32).reshape(4, 2)
        self.tokens = ["<|endoftext|>", "a", "b", "ab", "  ", "[PAD5]", "[PAD6]", "[PAD7]"]
        self.types = [3, 1, 1, 1, 4, 5, 5, 5]
        self.metadata_overrides: dict[str, str | int | float | bool | list[str]] = {}

    def write_pair(self) -> Path:
        save_file(self.source, self.snapshot / "model.safetensors", metadata={"format": "pt"})
        path = self.root / "model.gguf"
        writer = GGUFWriter(path, "mamba")
        for method, value in {
            "type": "model", "name": "snapshot", "finetune": "snapshot", "size_label": "0.10K",
            "context_length": 2**20, "embedding_length": 2, "feed_forward_length": 0,
            "head_count": 0, "block_count": 1, "ssm_conv_kernel": 3, "ssm_inner_size": 4,
            "ssm_state_size": 2, "ssm_time_step_rank": 1, "layer_norm_rms_eps": 1e-5,
            "ssm_dt_b_c_rms": False, "file_type": 0, "quantization_version": 2,
            "tokenizer_model": "gpt2", "tokenizer_pre": "olmo", "token_list": self.tokens,
            "token_types": self.types, "token_merges": ["a b"], "bos_token_id": 0,
            "eos_token_id": 0, "unk_token_id": 0, "pad_token_id": 0,
            **self.metadata_overrides,
        }.items():
            getattr(writer, "add_" + method)(value)
        for name, data in self.target.items():
            writer.add_tensor(name, data)
        try:
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
        finally:
            writer.close()
        return path

    def test_accounts_for_every_tensor_when_pair_is_valid(self) -> None:
        path = self.write_pair()
        # When: import the new auditor at the actual integration seam.
        audit = importlib.import_module("tools.audit_gguf")
        result = audit.audit_pair(self.snapshot, path)
        # Then
        self.assertEqual((result.sourceTensorCount, result.targetTensorCount), (13, 12))

    def test_rejects_exact_field_when_pair_is_corrupted(self) -> None:
        audit = importlib.import_module("tools.audit_gguf")
        cases = {
            "target.blk.0.ssm_d.presence": lambda: self.target.pop("blk.0.ssm_d"),
            "target.blk.0.unexpected.weight.presence": lambda: self.target.update({"blk.0.unexpected.weight": np.zeros(2, dtype=np.float32)}),
            "target.blk.0.ssm_conv1d.weight.logicalShape": lambda: self.target.update({"blk.0.ssm_conv1d.weight": self.target["blk.0.ssm_conv1d.weight"].T.copy()}),
            "target.blk.0.ssm_a.values": lambda: self.target.update({"blk.0.ssm_a": self.source["backbone.layers.0.mixer.A_log"].copy()}),
            "source.lm_head.weight.tiedEquality": lambda: self.source["lm_head.weight"].fill(9),
            "source.backbone.embeddings.weight.presence": lambda: self.source.pop("backbone.embeddings.weight"),
            "source.unexpected.weight.presence": lambda: self.source.update({"unexpected.weight": np.zeros(2, dtype=np.float32)}),
            "source.backbone.layers.0.mixer.conv1d.weight.shape": lambda: self.source.update({"backbone.layers.0.mixer.conv1d.weight": self.source["backbone.layers.0.mixer.conv1d.weight"].transpose(1, 0, 2).copy()}),
            "source.backbone.norm_f.weight.dtype": lambda: self.source.update({"backbone.norm_f.weight": self.source["backbone.norm_f.weight"].astype(np.float16)}),
            "target.output_norm.weight.dtype": lambda: self.target.update({"output_norm.weight": self.target["output_norm.weight"].astype(np.float16)}),
            "target.blk.0.ssm_a.finite": lambda: self.target["blk.0.ssm_a"].fill(float("nan")),
            "source.backbone.layers.0.mixer.A_log.finite": lambda: self.source["backbone.layers.0.mixer.A_log"].fill(float("inf")),
            "target.output_norm.weight.values": lambda: self.target["output_norm.weight"].__setitem__(0, -0.0),
            "metadata.tokenizer.ggml.tokens.value": lambda: self.tokens.__setitem__(1, "b"),
            "metadata.tokenizer.ggml.token_type.value": lambda: self.types.__setitem__(7, 1),
            "metadata.tokenizer.ggml.merges.value": lambda: self.metadata_overrides.update({"token_merges": ["b a"]}),
            "metadata.tokenizer.ggml.pre.value": lambda: self.metadata_overrides.update({"tokenizer_pre": "mpt"}),
            "metadata.mamba.ssm.state_size.value": lambda: self.metadata_overrides.update({"ssm_state_size": 3}),
            "metadata.tokenizer.ggml.bos_token_id.value": lambda: self.metadata_overrides.update({"bos_token_id": 1}),
            "metadata.tokenizer.ggml.add_bos_token.presence": lambda: self.metadata_overrides.update({"add_bos_token": True}),
        }
        for field, corrupt in cases.items():
            with self.subTest(field=field):
                # Given: prove an otherwise-valid control before each corruption.
                audit.audit_pair(self.snapshot, self.write_pair())
                tokens, types = self.tokens.copy(), self.types.copy()
                source = {k: v.copy() for k, v in self.source.items()}
                target = {k: v.copy() for k, v in self.target.items()}
                corrupt()
                # When / Then
                try:
                    with self.assertRaises(audit.AuditError) as raised:
                        audit.audit_pair(self.snapshot, self.write_pair())
                    self.assertEqual(raised.exception.field, field)
                finally:
                    self.source, self.target = source, target
                    self.tokens, self.types = tokens, types
                    self.metadata_overrides.clear()

    def test_rejects_extra_file_when_otherwise_valid_source_has_unaccounted_weights(self) -> None:
        audit = importlib.import_module("tools.audit_gguf")
        # Given
        target = self.write_pair()
        audit.audit_pair(self.snapshot, target)
        save_file({"extra": np.zeros(2, dtype=np.float32)}, self.snapshot / "extra.safetensors")
        # When / Then
        with self.assertRaises(audit.AuditError) as raised:
            audit.audit_pair(self.snapshot, target)
        self.assertEqual(raised.exception.field, "source.files")

    def test_binds_embedding_when_hf_serializes_no_output(self) -> None:
        # Given
        self.source.pop("lm_head.weight")
        audit = importlib.import_module("tools.audit_gguf")
        # When
        result = audit.audit_pair(self.snapshot, self.write_pair())
        # Then
        self.assertEqual((result.sourceTensorCount, result.outputBinding), (12, "hf-serialized-tied-omission"))

    def test_rejects_omission_when_hf_tying_is_disabled(self) -> None:
        # Given
        self.source.pop("lm_head.weight")
        audit = importlib.import_module("tools.audit_gguf")
        target = self.write_pair()
        audit.audit_pair(self.snapshot, target)
        config = MambaConfig.from_pretrained(self.snapshot)
        config.tie_word_embeddings = False
        config.save_pretrained(self.snapshot)
        # When / Then
        with self.assertRaises(audit.AuditError) as raised:
            audit.audit_pair(self.snapshot, target)
        self.assertEqual(raised.exception.field, "source.lm_head.weight.tiedContract")

    def test_checks_projection_when_explicit_output_is_untied(self) -> None:
        # Given
        config = MambaConfig.from_pretrained(self.snapshot)
        config.tie_word_embeddings = False
        config.save_pretrained(self.snapshot)
        self.source["lm_head.weight"].fill(2)
        self.target["output.weight"] = self.source["lm_head.weight"].copy()
        self.metadata_overrides["size_label"] = "0.12K"
        audit = importlib.import_module("tools.audit_gguf")
        # When
        result = audit.audit_pair(self.snapshot, self.write_pair())
        # Then
        self.assertEqual((result.targetTensorCount, result.outputBinding), (13, "explicit-output"))

    def test_fails_closed_when_cli_manifest_identity_is_invalid(self) -> None:
        # Given: real canonical CLI control, generated output remains test-local.
        root = Path(__file__).resolve().parents[1]
        canonical = root.parent / "mamba1-130m-visualizer-foundation/.artifacts/manifests/model.json"
        command = [sys.executable, "-B", str(root / "tools/audit_gguf.py"),
                   "--manifest", str(canonical), "--out", str(self.root / "report.json")]
        control = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
        self.assertEqual(control.returncode, 0, control.stderr)
        malformed = self.root / "malformed.json"
        malformed.write_text("[]")
        command[4] = str(malformed)
        # When
        result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
        # Then: machine-readable failure, never stale success output.
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr.strip(), json.dumps({"status": "failed", "field": "manifest.type"}))
        audit = importlib.import_module("tools.audit_gguf")
        cases = {
            "manifest.captureId": lambda m: m.update({"captureId": "stale"}),
            "manifest.ggufSha256": lambda m: m.update({"ggufSha256": "0" * 64}),
            "manifest.ggufPath.sha256": lambda m: m.update({"ggufPath": str(malformed)}),
            "manifest.fixtures.token-ids.json.sha256": lambda m: m["fixtures"]["token-ids.json"].update({"sha256": "0" * 64}),
            "manifest.environment": lambda m: m["environment"]["packages"].update({"torch": "stale"}),
            "manifest.source": lambda m: m["source"].update({"productPatchSha256": "0" * 64}),
        }
        for field, corrupt in cases.items():
            with self.subTest(field=field):
                # Given: the same real control; only the named identity is wrong.
                document = json.loads(canonical.read_text())
                corrupt(document)
                if field != "manifest.captureId":
                    document["captureId"] = "prepare-" + audit.provenance.json_hash({k: v for k, v in document.items() if k != "captureId"})
                changed = self.root / "corrupted-manifest.json"
                changed.write_text(json.dumps(document))
                command[4] = str(changed)
                # When
                result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
                # Then
                self.assertEqual((result.returncode, result.stdout), (1, ""))
                self.assertEqual(json.loads(result.stderr)["field"], field)
