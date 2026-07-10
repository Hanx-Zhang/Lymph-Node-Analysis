from pathlib import Path


MODEL_DIR = Path(__file__).resolve().parent

config = {
    "in_channels": 1,
    "out_channels": 1,
    "weight_path": str(MODEL_DIR / "azygos_weights0_235.pth"),
    "device": "cuda:0",
    "roi_size": (128, 128, 128),
    "sw_batch_size": 1,
    "overlap": 0.50
}


