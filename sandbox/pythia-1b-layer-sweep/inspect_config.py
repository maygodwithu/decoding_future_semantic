"""
Step 2 of protocol: inspect EleutherAI/pythia-1b's config programmatically to
get the actual transformer block count (n_layers) and hidden_size, rather
than assuming values. Saves artifacts/pythia_1b_layer_sweep/model_config.json.
"""
import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from transformers import AutoConfig

MODEL_NAME = "EleutherAI/pythia-1b"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
ART_DIR = os.path.join(OUT_DIR, "artifacts", "pythia_1b_layer_sweep")
os.makedirs(ART_DIR, exist_ok=True)


def main():
    cfg = AutoConfig.from_pretrained(MODEL_NAME)
    n_layers = cfg.num_hidden_layers
    hidden_size = cfg.hidden_size
    meta = {
        "model_name": MODEL_NAME,
        "n_layers": n_layers,
        "hidden_size": hidden_size,
        "num_attention_heads": getattr(cfg, "num_attention_heads", None),
        "architectures": getattr(cfg, "architectures", None),
        "note": (
            "n_layers = cfg.num_hidden_layers (number of gpt_neox transformer "
            "blocks, 0-indexed 0..n_layers-1). hidden_size = cfg.hidden_size."
        ),
    }
    with open(f"{ART_DIR}/model_config.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
