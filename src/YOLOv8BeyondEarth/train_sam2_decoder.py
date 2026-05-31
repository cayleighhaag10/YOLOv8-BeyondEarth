import sys
sys.path.insert(0, "/scratch/users/cayleigh/YOLOv8-BeyondEarth/src")

import typing_extensions
if not hasattr(typing_extensions, "TypeIs"):
    typing_extensions.TypeIs = typing_extensions.TypeGuard

import torch
import torch.nn.functional as F
import numpy as np
import geopandas as gpd
import rasterio
import rasterio.mask
from pathlib import Path
from tqdm import tqdm

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

SAM2_CHECKPOINT = Path("/scratch/users/cayleigh/checkpoints/sam2.1_hiera_small.pt")
DATASET_DIR     = Path("/scratch/users/cayleigh/Apr2023-Mars-Moon-Earth-mask-5px/preprocessing")
CKPT_OUT_DIR    = Path("/scratch/users/cayleigh/sam2_finetuned")
CKPT_OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda:0"
EPOCHS = 10
LR     = 1e-4
MIN_AREA_PX = 36   # 6×6 px minimum instance area


# Fine-tune SAM2 mask decoder using Prieur et al. boulder dataset.

# Freeze image encoder

# Train prompt encoder + msk decoder
