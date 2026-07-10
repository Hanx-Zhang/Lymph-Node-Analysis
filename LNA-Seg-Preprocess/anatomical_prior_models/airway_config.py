from pathlib import Path


MODEL_DIR = Path(__file__).resolve().parent

config = {
    "in_channels": 1,
    "out_channels": 2,
    "finalsigmoid": 1,
    "fmaps_degree": 16,
    "fmaps_layer_number": 4,
    "layer_order": "cpi",
    "GroupNormNumber": 4,
    "device": "cuda:0",
    "weight_path": str(MODEL_DIR / "airway_model2.pth"),
    "roi_size": (128, 224, 304),
    "sw_batch_size": 1,
    "overlap": 0.75,
    "mode": 'gaussian',
    "sigma_scale": 0.25,
    "use_HU_window": True,
}

