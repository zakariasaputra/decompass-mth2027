"""Clean-room reproducibility and integrity verification test.

Verifies:
1. Zero private match data or ground-truth leaks in git tracking.
2. Necessary offline model weights (TCN, HGB, Nova PathCRF variants) & ensemble config are present.
3. Deterministic fresh-clone execution across two independent cold --rebuild-cache runs
   (cache cleared before each run, < 600s each) plus one warm cached run (< 180s).
4. Exact bit-for-bit SHA-256 match against expected submission hash across all three runs.
"""

import glob
import hashlib
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPECTED_SHA = "61d792693de164b87f9b810df03c3699ccb15ea4d9437900ae735c1242b973fb"
MAX_COLD_RUNTIME_SECONDS = 600.0
MAX_WARM_RUNTIME_SECONDS = 180.0


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def check_git_leaks():
    print("[1/4] Checking git tracking for data leaks...")
    res = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        print("Warning: git status returned non-zero code.")
        return

    # Check git ls-files
    res = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    tracked = res.stdout.splitlines()

    prohibited_prefixes = ("data/", "external/", "organizers/")
    allowed_exceptions = {
        "data/README.md",
        "external/LICENSES.md",
        "external/MANIFEST.json",
    }
    prohibited_exts = (".mp4", ".pdf", ".zip", ".tar", ".gz", ".pl3container")

    for f in tracked:
        f_norm = f.replace("\\", "/")
        if f_norm.startswith(prohibited_prefixes) and f_norm not in allowed_exceptions:
            raise AssertionError(f"PROHIBITED DATA LEAK DETECTED: {f} is tracked in git!")
        if any(f_norm.endswith(ext) for ext in prohibited_exts):
            raise AssertionError(f"PROHIBITED LARGE/BINARY FILE DETECTED: {f} is tracked in git!")
        if (("gt/" in f_norm or "data/" in f_norm) and "_gt" in f_norm and "_public" not in f_norm) or "reference.csv" in f_norm:
            raise AssertionError(f"PROHIBITED PRIVATE ASSET DETECTED: {f} is tracked in git!")

    print("      Git leak check PASSED (0 private assets tracked).")


def check_model_weights():
    print("[2/4] Checking offline model weights...")
    required_models = [
        "tcn_variant_a.pt",
        "tcn_variant_b.pt",
        "hgb_variant_a.pkl",
        "hgb_variant_b.pkl",
        "nova_pathcrf_best_team.pt",
        "nova_pathcrf_args.json",
    ]
    for m in required_models:
        model_path = os.path.join(ROOT, "models", m)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Missing required model weights: {model_path}")
        sz = os.path.getsize(model_path)
        print(f"      models/{m} present ({sz / 1024:.1f} KB).")
    ens_cfg = os.path.join(ROOT, "config", "ensemble_weights.json")
    if not os.path.isfile(ens_cfg):
        raise FileNotFoundError(f"Missing ensemble weight configuration: {ens_cfg}")
    print("      config/ensemble_weights.json present.")


def _clear_pipeline_cache():
    """Remove generated pipeline .npz caches to simulate a fresh clone."""
    data_dir = os.environ.get("MTAH_DATA_DIR", os.path.join(ROOT, "data"))
    cache_dir = os.path.join(data_dir, "cache")
    if not os.path.isdir(cache_dir):
        return 0
    removed = 0
    for npz_path in glob.glob(os.path.join(cache_dir, "*.npz")):
        os.remove(npz_path)
        removed += 1
    return removed


def _run_once(tag, out_path, rebuild_cache=False, max_seconds=MAX_COLD_RUNTIME_SECONDS):
    cmd = [sys.executable, os.path.join(ROOT, "run_pipeline.py")]
    if rebuild_cache:
        cmd.append("--rebuild-cache")
    cmd.extend(["--out", out_path])
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"{tag} failed:\n{res.stderr}")
        sys.exit(1)
    print(f"      {tag} completed in {elapsed:.2f}s (limit < {max_seconds:.0f}s).")
    assert elapsed < max_seconds, f"{tag} exceeded time limit ({elapsed:.2f}s >= {max_seconds:.0f}s)"
    return sha256_file(out_path)


def check_pipeline_reproducibility():
    print("[3/4] Running clean-room fresh-cache cold (x2) + warm (x1) executions...")
    tmp_a = os.path.join(ROOT, "submission", "tmp_run_a.csv")
    tmp_b = os.path.join(ROOT, "submission", "tmp_run_b.csv")
    tmp_c = os.path.join(ROOT, "submission", "tmp_run_c.csv")
    os.makedirs(os.path.dirname(tmp_a), exist_ok=True)

    try:
        removed_a = _clear_pipeline_cache()
        print(f"      Cleared {removed_a} cache files before Cold Run 1.")
        sha_a = _run_once("Cold Run 1 (--rebuild-cache)", tmp_a, rebuild_cache=True, max_seconds=MAX_COLD_RUNTIME_SECONDS)

        removed_b = _clear_pipeline_cache()
        print(f"      Cleared {removed_b} cache files before Cold Run 2.")
        sha_b = _run_once("Cold Run 2 (--rebuild-cache)", tmp_b, rebuild_cache=True, max_seconds=MAX_COLD_RUNTIME_SECONDS)

        sha_c = _run_once("Warm Run 3 (cached)", tmp_c, rebuild_cache=False, max_seconds=MAX_WARM_RUNTIME_SECONDS)

        print("[4/4] Verifying bit-for-bit SHA-256 hashes...")
        print(f"      Cold Run 1: {sha_a}")
        print(f"      Cold Run 2: {sha_b}")
        print(f"      Warm Run 3: {sha_c}")

        assert sha_a == sha_b, "Determinism error: SHA-256 mismatch between two independent cold runs!"
        assert sha_a == sha_c, "Cache consistency error: warm run SHA-256 differs from cold runs!"
        assert sha_a == EXPECTED_SHA, f"Hash mismatch: expected {EXPECTED_SHA}, got {sha_a}"
        print("      Verification SUCCESS: 100% bit-for-bit fresh-clone reproduction of expected submission hash!")
    finally:
        for p in (tmp_a, tmp_b, tmp_c):
            if os.path.isfile(p):
                os.remove(p)


def main():
    print("=== Clean-Room Verification Suite ===")
    check_git_leaks()
    check_model_weights()
    check_pipeline_reproducibility()
    print("=====================================")


if __name__ == "__main__":
    main()
