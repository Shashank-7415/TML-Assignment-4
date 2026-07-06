import os
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

# ==========================================================
# CONFIGURATION
# ==========================================================

ZIP_FILE = "/Users/shashankmbp/tml_assignment4/Dataset.zip"
DATASET_DIR = Path("Dataset")

TEMP_OUT_DIR = Path("submission_temp")
FILE_PATH = "submission.zip"

# ==========================================================
# UNZIP DATASET IF NECESSARY
# ==========================================================

if not DATASET_DIR.exists():
    if not os.path.exists(ZIP_FILE):
        raise FileNotFoundError(
            f"Could not find dataset zip at {ZIP_FILE}"
        )

    print(f"Extracting {ZIP_FILE}...")
    with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
        zip_ref.extractall(".")
else:
    print("Dataset already extracted.")

# Create output folder
TEMP_OUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# DATASET MAPPING
# ==========================================================

CATEGORIES = [
    ("WM_1", 1, 25),
    ("WM_2", 26, 50),
    ("WM_3", 51, 75),
    ("WM_4", 76, 100),
    ("WM_5", 101, 125),
    ("WM_6", 126, 150),
    ("WM_7", 151, 175),
    ("WM_8", 176, 200),
]

# ==========================================================
# WATERMARK STRENGTH (calibrated per group locally via detection-proxy /
# LPIPS sweep — see calibrate.py). The frequency-domain component was
# tested and found to mostly damage quality without adding real detection
# signal for these 8 schemes, so it's left out here. If you want to
# re-test it, do so via calibrate.py with alpha_freq=0 as a baseline
# comparison first.
# ==========================================================

ALPHA_OVERRIDES = {
    
    "WM_1": 1.4,
    "WM_2": 1.0,
    "WM_3": 1.4,
    "WM_4": 1.0,
    "WM_5": 2.0,
    "WM_6": 1.0,
    "WM_7": 1.0,
    "WM_8": 1.0,

}
DEFAULT_ALPHA = 3.0  # fallback if a group isn't listed above

BLUR_SIGMA = 8.0  # controls the high-pass cutoff when isolating the watermark


# ==========================================================
# WATERMARK ESTIMATION + INJECTION HELPERS
# ==========================================================

def load_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32)


def estimate_spatial_residual(source_images) -> np.ndarray:
    """Average ALL 25 watermarked images in a group (not just one), then
    high-pass the result. Averaging cancels out the random photo content
    across the 25 images while the shared watermark reinforces, since it's
    the one thing common to all of them. High-passing then strips out the
    leftover low-frequency 'average scene' structure, leaving mostly the
    watermark's own pattern.
    """
    imgs = [load_image(p) for p in source_images]
    stack = np.stack(imgs, axis=0)
    avg = stack.mean(axis=0)
    low_freq = gaussian_filter(avg, sigma=(BLUR_SIGMA, BLUR_SIGMA, 0))
    residual = avg - low_freq
    return residual


def texture_mask(img: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Weight injected in busier/textured regions more than flat regions,
    since perceptual distance (LPIPS) is far less sensitive there. Keeps
    a floor so flat regions still get some signal.
    """
    gray = img.mean(axis=2)
    local_mean = gaussian_filter(gray, sigma=sigma)
    local_var = gaussian_filter((gray - local_mean) ** 2, sigma=sigma)
    mask = local_var / (local_var.max() + 1e-8)
    mask = 0.4 + 0.6 * mask
    return mask[..., None]


def inject_residual(target: np.ndarray, residual: np.ndarray, alpha: float) -> np.ndarray:
    masked_residual = residual * texture_mask(target)
    forged = target + alpha * masked_residual
    return np.clip(forged, 0, 255)


# ==========================================================
# MAIN LOOP
# ==========================================================

print("Generating forged images...")

total_processed = 0

for source_wm, target_start, target_stop in CATEGORIES:

    print(
        f"Processing {source_wm} -> "
        f"{target_start}.png to {target_stop}.png"
    )

    source_dir = DATASET_DIR / "watermarked_sources" / source_wm
    source_images = sorted(source_dir.glob("*.png"))

    if len(source_images) != 25:
        print(
            f"Warning: expected 25 source images in "
            f"{source_dir}, found {len(source_images)}"
        )

    # Estimate the watermark ONCE per group, using all 25 source images
    # together, instead of pairing one target with one arbitrary source.
    residual = estimate_spatial_residual(source_images)
    alpha = ALPHA_OVERRIDES.get(source_wm, DEFAULT_ALPHA)

    target_dir = DATASET_DIR / "clean_targets"

    target_images = [
        target_dir / f"{i}.png"
        for i in range(target_start, target_stop + 1)
    ]

    for target_path in target_images:

        target_img = load_image(target_path)

        forged_img = inject_residual(target_img, residual, alpha)
        forged_img = forged_img.astype(np.uint8)

        # Save using exact filename
        out_path = TEMP_OUT_DIR / target_path.name
        Image.fromarray(forged_img).save(out_path)

        total_processed += 1

print(f"Generated {total_processed} images.")

if total_processed != 200:
    print(
        f"WARNING: expected 200 images "
        f"but generated {total_processed}"
    )

# ==========================================================
# CREATE SUBMISSION ZIP
# ==========================================================

print("Creating submission zip...")

with zipfile.ZipFile(
    FILE_PATH,
    "w",
    zipfile.ZIP_DEFLATED
) as zipf:

    for img_path in sorted(TEMP_OUT_DIR.glob("*.png")):
        zipf.write(
            img_path,
            arcname=img_path.name
        )

print(f"Submission file saved as {FILE_PATH}")
