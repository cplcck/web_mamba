#!/usr/bin/env python3
"""Download a complete immutable HF snapshot and convert it with pinned CPU code.

An ordinary rerun verifies the stored artifacts without consulting moving main.
Only --refresh-revision resolves main again. Paths in model.json are absolute.
The preparation captureId is an input identity, not a CPU graph capture claim.
"""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile

REPO_ID = "state-spaces/mamba-130m-hf"
BASELINE = "8144f3192e5a3131cd043f284525e6ceebf82d0f"
ROOT = Path(__file__).resolve().parents[1]
CANONICAL = [1, 42, 314, 2718, 7, 99, 1024, 17, 2048, 13, 512, 9, 4096, 23, 128, 31, 64]
EXPECTED_CONFIG = {
    "model_type": "mamba", "architectures": ["MambaForCausalLM"],
    "num_hidden_layers": 24, "hidden_size": 768, "intermediate_size": 1536,
    "state_size": 16, "conv_kernel": 4, "time_step_rank": 48,
}
CONVERSION_ENV = {
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def require_sha(revision):
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a full immutable lowercase commit SHA")
    return revision


def select_revision(hub, previous=None, refresh=False):
    if previous is not None and not refresh:
        return require_sha(previous)
    return require_sha(hub.model_info(REPO_ID, revision="main", timeout=30).sha)


def payload_hashes(snapshot):
    result = {}
    for path in sorted(snapshot.rglob("*")):
        relative = path.relative_to(snapshot)
        if relative.parts[:2] == (".cache", "huggingface"):
            continue
        if path.is_file():
            result[relative.as_posix()] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    return result


def verify_payloads(snapshot, expected):
    actual = payload_hashes(snapshot)
    if actual != expected:
        differences = sorted(name for name in actual.keys() | expected.keys()
                             if actual.get(name) != expected.get(name))
        raise ValueError("snapshot payload mismatch: " + ", ".join(differences))
    return actual


def hub_tree(info, revision):
    if info.sha != revision:
        raise ValueError("Hub returned metadata for a different revision")
    tree = {}
    for entry in info.siblings:
        name = entry.rfilename
        if not name or Path(name).is_absolute() or ".." in Path(name).parts or name.startswith(".cache/"):
            raise ValueError("unsafe Hub payload path")
        lfs = entry.lfs
        tree[name] = {"bytes": entry.size, "gitBlob": entry.blob_id,
                      "sha256": lfs.sha256 if lfs else None}
        if not isinstance(entry.size, int) or entry.size < 0:
            raise ValueError("Hub payload metadata lacks size")
        digest = tree[name]["sha256"] or entry.blob_id
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", digest):
            raise ValueError("Hub payload metadata lacks content hash")
    if not {"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"} <= tree.keys():
        raise ValueError("incomplete Mamba snapshot tree")
    return tree


def verify_hub_tree(snapshot, tree):
    payloads = payload_hashes(snapshot)
    if payloads.keys() != tree.keys():
        raise ValueError("snapshot does not contain the complete immutable Hub tree")
    for name, expected in tree.items():
        actual = payloads[name]
        if actual["bytes"] != expected["bytes"]:
            raise ValueError("Hub payload size mismatch: " + name)
        if expected["sha256"]:
            matches = actual["sha256"] == expected["sha256"]
        else:
            data = (snapshot / name).read_bytes()
            matches = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == expected["gitBlob"]
        if not matches:
            raise ValueError("Hub payload hash mismatch: " + name)
    return payloads


def validate_config(config):
    for key, expected in EXPECTED_CONFIG.items():
        if config.get(key) != expected or type(config.get(key)) is not type(expected):
            raise ValueError("unexpected Mamba configuration: " + key)
    for key, expected in {"n_layer": 24, "d_model": 768, "d_inner": 1536, "expand": 2}.items():
        if key in config and config[key] != expected:
            raise ValueError("conflicting Mamba configuration: " + key)
    vocab = config.get("vocab_size")
    if type(vocab) is not int or vocab <= 0:
        raise ValueError("invalid vocabulary size")
    return vocab


def validate_fixtures(directory, vocab):
    canonical_path = directory / "token-ids.json"
    calibration_path = directory / "calibration-ids.json"
    canonical, calibration = read_json(canonical_path), read_json(calibration_path)
    expected_calibration = [
        {"id": "reversed", "tokenIds": list(reversed(CANONICAL))},
        {"id": "repeated-42", "tokenIds": [42] * 17},
        {"id": "range-100-116", "tokenIds": list(range(100, 117))},
    ]
    for document in (canonical, calibration):
        if document.get("schemaVersion") != 1 or document.get("kind") != "synthetic-token-ids":
            raise ValueError("invalid fixture schema")
    if canonical.get("tokenIds") != CANONICAL or calibration.get("fixtures") != expected_calibration:
        raise ValueError("fixture IDs differ from the predeclared inputs")
    settings = {"sequenceId": 0, "prefillLength": 16, "positions": list(range(17)),
                "outputsPerCall": 1, "implicitSpecialTokens": False}
    if any(canonical.get(key) != value or type(canonical.get(key)) is not type(value)
           for key, value in settings.items()):
        raise ValueError("invalid canonical fixture schedule")
    for ids in [canonical["tokenIds"]] + [f["tokenIds"] for f in calibration["fixtures"]]:
        if any(type(token) is not int or not 0 <= token < vocab for token in ids):
            raise ValueError("fixture token ID outside vocabulary or not an integer")
    return {path.name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in (canonical_path, calibration_path)}


def source_identity(source):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(source), *args], timeout=30)
    commit = git("rev-parse", "HEAD").decode().strip()
    if commit != BASELINE:
        raise ValueError("llama.cpp source is not the required baseline")
    status = git("status", "--porcelain=v1", "--untracked-files=all").decode()
    relevant = ["convert_hf_to_gguf.py", "conversion", "gguf-py", "src", "include", "ggml",
                "CMakeLists.txt", "cmake", "vendor"]
    if git("diff", "HEAD", "--", *relevant) or git("ls-files", "--others", "--exclude-standard", "--", *relevant):
        raise ValueError("converter/runtime source has unapproved modifications")
    return {"path": str(source), "commit": commit, "runtimeCommit": commit,
            "tree": git("rev-parse", "HEAD^{tree}").decode().strip(),
            "status": status, "productPatchSha256": hashlib.sha256(b"").hexdigest(),
            "worktreeDiffSha256": hashlib.sha256(git("diff", "HEAD", "--binary")).hexdigest(),
            "converterSha256": sha256(source / "convert_hf_to_gguf.py"),
            "mambaConverterSha256": sha256(source / "conversion/mamba.py")}


def environment_receipt():
    packages = dict(sorted((d.metadata["Name"].lower().replace("_", "-"), d.version)
                           for d in importlib.metadata.distributions()))
    lock = ROOT / "tools/requirements.lock"
    text = lock.read_text()
    locked = dict(re.findall(r"^([A-Za-z0-9_.-]+)==([^\s]+)", text, re.MULTILINE))
    for name, version in locked.items():
        if packages.get(name.lower().replace("_", "-")) != version:
            raise ValueError("installed dependency differs from lock: " + name)
    if not locked:
        raise ValueError("empty dependency lock")
    cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")), platform.machine())
    return {"python": platform.python_version(), "pythonExecutable": sys.executable,
            "pythonBinarySha256": sha256(Path(sys.executable).resolve()),
            "platform": platform.platform(), "cpu": cpu, "packages": packages,
            "requirementsSha256": sha256(ROOT / "tools/requirements.txt"),
            "lockSha256": sha256(lock), "conversionEnvironment": CONVERSION_ENV,
            "runtimeBuild": None, "runtimeBuildReason": "CPU binary, compiler flags and runtime settings belong to capture task 4"}


async def run_command(command, *, env, timeout, log):
    """Bound the subprocess and reap it on cancellation as well as failure."""
    with log.open("w") as stream:
        process = await asyncio.create_subprocess_exec(*command, env=env, stdout=stream,
                                                       stderr=asyncio.subprocess.STDOUT)
        try:
            code = await asyncio.wait_for(process.wait(), timeout)
            if code:
                raise subprocess.CalledProcessError(code, command)
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()


def run_converter(source, snapshot, output, log, timeout):
    command = [sys.executable, "-B", str(source / "convert_hf_to_gguf.py"), str(snapshot),
               "--outtype", "f32", "--outfile", str(output)]
    env = {**os.environ, **CONVERSION_ENV}
    env.pop("NO_LOCAL_GGUF", None)
    env.pop("PYTHONPATH", None)
    asyncio.run(run_command(command, env=env, timeout=timeout, log=log))
    with output.open("rb") as stream:
        if stream.read(4) != b"GGUF":
            raise ValueError("converter did not produce GGUF")
    return command


def prepare_model(source, artifact_dir, *, fixture_dir=ROOT / "fixtures", refresh=False,
                  hub=None, downloader=None, converter=run_converter, timeout=600):
    source, artifact_dir = Path(source).resolve(), Path(artifact_dir).resolve()
    manifests = artifact_dir / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    with (manifests / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _prepare(source, artifact_dir, fixture_dir, refresh, hub, downloader, converter, timeout)


def _prepare(source, artifacts, fixture_dir, refresh, hub, downloader, converter, timeout):
    source_info = source_identity(source)
    environment = environment_receipt()
    manifests = artifacts / "manifests"
    manifest_path, resolution_path = manifests / "model.json", manifests / "resolution.json"
    previous = read_json(manifest_path) if manifest_path.exists() else None
    resolution = read_json(resolution_path) if resolution_path.exists() else None
    if previous:
        if previous.get("schemaVersion") != 1 or previous.get("repoId") != REPO_ID:
            raise ValueError("invalid model manifest")
        identity = {key: value for key, value in previous.items() if key != "captureId"}
        if previous.get("captureId") != "prepare-" + json_hash(identity):
            raise ValueError("manifest identity mismatch")
        require_sha(previous["revision"])
        verify_payloads(Path(previous["snapshotPath"]), previous["payloads"])
        if sha256(previous["ggufPath"]) != previous["ggufSha256"]:
            raise ValueError("GGUF hash mismatch")
        fixtures = validate_fixtures(fixture_dir, validate_config(previous["config"]))
        if fixtures != previous["fixtures"]:
            raise ValueError("fixture hash mismatch")
        if source_info != previous["source"] or environment != previous["environment"]:
            raise ValueError("source/environment identity differs from recorded conversion")
        if not refresh:
            return previous
    if hub is None:
        from huggingface_hub import HfApi
        hub = HfApi()
    revision = select_revision(hub, resolution["revision"] if resolution else None, refresh)
    if not resolution or resolution["revision"] != revision:
        resolution = {"repoId": REPO_ID, "revision": revision, "sourceAtResolution": source_info}
        atomic_json(resolution_path, resolution)
    if "tree" not in resolution:
        resolution["tree"] = hub_tree(hub.model_info(REPO_ID, revision=revision,
                                                    files_metadata=True, timeout=30), revision)
        atomic_json(resolution_path, resolution)
    snapshot = artifacts / "hf" / revision
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if not snapshot.exists():
        if downloader is None:
            from huggingface_hub import snapshot_download
            downloader = snapshot_download
        # Default Hub cache reuses verified local blobs; no filtering and no --remote.
        cached = Path(downloader(repo_id=REPO_ID, revision=revision, max_workers=4, etag_timeout=30))
        verify_hub_tree(cached, resolution["tree"])
        with tempfile.TemporaryDirectory(prefix=".prepare-", dir=snapshot.parent) as temporary:
            stage = Path(temporary) / "snapshot"
            shutil.copytree(cached, stage, ignore=shutil.ignore_patterns(".cache"))
            verify_hub_tree(stage, resolution["tree"])
            stage.rename(snapshot)
    payloads = verify_hub_tree(snapshot, resolution["tree"])
    config = read_json(snapshot / "config.json")
    fixtures = validate_fixtures(fixture_dir, validate_config(config))
    gguf = artifacts / "gguf" / (revision + "-f32.gguf")
    gguf.parent.mkdir(parents=True, exist_ok=True)
    if previous and previous["revision"] == revision:
        return previous
    if gguf.exists():
        raise ValueError("unmanifested GGUF exists; preserve it and investigate before retry")
    temporary = gguf.with_suffix(".gguf.tmp")
    try:
        command = converter(source, snapshot, temporary, manifests / (revision + "-conversion.log"), timeout)
        verify_payloads(snapshot, payloads)
        if source_identity(source) != source_info:
            raise ValueError("source changed during conversion")
        digest = sha256(temporary)
        temporary.replace(gguf)
    finally:
        temporary.unlink(missing_ok=True)
    result = {"schemaVersion": 1, "repoId": REPO_ID, "revision": revision,
              "snapshotPath": str(snapshot), "ggufPath": str(gguf), "ggufSha256": digest,
              "outtype": "f32", "payloads": payloads, "hubTree": resolution["tree"],
              "config": config, "fixtures": fixtures, "source": source_info,
              "environment": environment, "converterCommand": command,
              "prepareScriptSha256": sha256(Path(__file__)),
              "conversionLogPath": str(manifests / (revision + "-conversion.log"))}
    result["captureId"] = "prepare-" + json_hash(result)
    atomic_json(manifest_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--refresh-revision", action="store_true")
    parser.add_argument("--timeout", type=int, default=600, help="converter timeout in seconds")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    # SIGTERM follows the same cleanup path as Ctrl-C, including the child process.
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    try:
        result = prepare_model(args.source, args.artifact_dir, refresh=args.refresh_revision, timeout=args.timeout)
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError, TimeoutError) as error:
        print("prepare-model failed: " + str(error), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("prepare-model interrupted; immutable inputs retained", file=sys.stderr)
        return 130
    print(json.dumps({"manifest": str(args.artifact_dir.resolve() / "manifests/model.json"),
                      "revision": result["revision"], "ggufSha256": result["ggufSha256"]}))
    return 0


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
