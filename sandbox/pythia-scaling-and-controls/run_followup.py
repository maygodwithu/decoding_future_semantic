"""
Orchestrator for the full follow-up study (priorities 1-3 + reporting).
Idempotent: re-running skips any step whose output file already exists.
Cleans the HF hub cache for a scaling-study model after that model's steps
finish (keeps disk usage bounded; see du/df checks in run()).
"""
import json
import os
import shutil
import subprocess
import sys
import time

PY = sys.executable
ROOT = os.path.dirname(os.path.abspath(__file__))

SCALING_MODELS = [
    ("EleutherAI/pythia-410m", "pythia-410m"),
    ("EleutherAI/pythia-1b", "pythia-1.0b"),
    ("EleutherAI/pythia-1.4b", "pythia-1.4b"),  # already extracted by build_usable_and_split.py
    ("EleutherAI/pythia-2.8b", "pythia-2.8b"),
]
STEER_N_PROMPTS = 60


def run(cmd, **kw):
    print(f"[run_followup] $ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True, cwd=ROOT, **kw)
    print(f"[run_followup] done in {time.time() - t0:.1f}s", flush=True)


def exists(path):
    return os.path.exists(os.path.join(ROOT, path))


def disk_free_gb():
    total, used, free = shutil.disk_usage(ROOT)
    return free / 1e9


def clean_model_cache(model_name):
    cache_root = os.path.expanduser("~/.cache/huggingface/hub")
    safe = "models--" + model_name.replace("/", "--")
    p = os.path.join(cache_root, safe)
    if os.path.isdir(p):
        shutil.rmtree(p, ignore_errors=True)
        print(f"[run_followup] cleaned HF cache for {model_name}", flush=True)


def step_scaling(model_name, tag):
    out_dir = f"artifacts/{tag}/greedy"
    print(f"[run_followup] === scaling model {model_name} ({tag}) === free_disk={disk_free_gb():.1f}GB", flush=True)

    if not exists(f"{out_dir}/hidden_last_token.npz"):
        run([PY, "extract_hidden_and_generations.py", "--model_name", model_name, "--decode", "greedy",
             "--out_dir", out_dir])
    if not exists(f"{out_dir}/continuation_embeddings.npy"):
        run([PY, "embed_continuations.py", "--dir", out_dir])
    if not exists(f"{out_dir}/probe_results.json"):
        run([PY, "probe_train_generic.py", "--dir", out_dir])

    probe_results = json.load(open(os.path.join(ROOT, out_dir, "probe_results.json")))
    best_layer = probe_results["best_layer_by_val"]

    if not exists(f"{out_dir}/steering_summary.json"):
        run([PY, "steering_generic.py", "--dir", out_dir, "--model_name", model_name,
             "--layer", str(best_layer), "--n_prompts", str(STEER_N_PROMPTS),
             "--out_json", f"{out_dir}/steering_summary.json"])

    if model_name != "EleutherAI/pythia-1.4b":  # keep 1.4b cached; needed again for priority 1/3
        clean_model_cache(model_name)


def step_priority1():
    out_dir = "artifacts/pythia-1.4b/greedy"
    if not exists(f"{out_dir}/token_control_results.csv"):
        run([PY, "token_identity_control.py", "--dir", out_dir, "--model_name", "EleutherAI/pythia-1.4b",
             "--layer", "20", "--probe_layer", "20", "--out_csv", f"{out_dir}/token_control_results.csv"])


def step_priority3():
    out_dir = "artifacts/pythia-1.4b/sampled"
    if not exists(f"{out_dir}/hidden_last_token.npz"):
        run([PY, "extract_hidden_and_generations.py", "--model_name", "EleutherAI/pythia-1.4b",
             "--decode", "sampled", "--temperature", "0.8", "--top_p", "0.95", "--seed", "42",
             "--out_dir", out_dir])
    if not exists(f"{out_dir}/continuation_embeddings.npy"):
        run([PY, "embed_continuations.py", "--dir", out_dir])
    if not exists(f"{out_dir}/probe_results.json"):
        run([PY, "probe_train_generic.py", "--dir", out_dir])
    probe_results = json.load(open(os.path.join(ROOT, out_dir, "probe_results.json")))
    best_layer = probe_results["best_layer_by_val"]
    if not exists(f"{out_dir}/token_control_results.csv"):
        run([PY, "token_identity_control.py", "--dir", out_dir, "--model_name", "EleutherAI/pythia-1.4b",
             "--layer", "20", "--probe_layer", str(best_layer), "--out_csv", f"{out_dir}/token_control_results.csv"])


def main():
    os.chdir(ROOT)
    if not exists("artifacts/usable_prompts.jsonl") or not exists("artifacts/split.json"):
        run([PY, "build_usable_and_split.py"])

    step_priority1()

    for model_name, tag in SCALING_MODELS:
        step_scaling(model_name, tag)

    step_priority3()

    run([PY, "build_summaries.py"])
    run([PY, "plot_scaling.py"])
    run([PY, "write_report.py"])
    print("[run_followup] ALL DONE")


if __name__ == "__main__":
    main()
