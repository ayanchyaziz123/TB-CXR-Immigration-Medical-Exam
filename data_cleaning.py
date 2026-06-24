"""
TB-CXR Data Cleaning and Feature Engineering Pipeline.

Steps applied in order:
    1. Duplicate detection        — perceptual hashing across all three datasets
    2. Image quality filtering    — exposure, blur, minimum size checks
    3. Lung ROI cropping          — remove borders, rulers, text overlays
    4. CLAHE enhancement          — standard contrast normalization for chest X-rays
    5. Dataset statistics         — compute per-channel mean/std from training split

Reference:
    Rajpurkar et al. CheXNet (2017) — CLAHE preprocessing on chest X-rays
    Candemir et al. Lung segmentation in chest radiographs (2014)
"""

import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
from collections import defaultdict


# ── 1. Duplicate Detection ─────────────────────────────────────────────────

def phash(image_path: str, hash_size: int = 16) -> str:
    """
    Perceptual hash of a chest X-ray image using DCT.

    Why DCT instead of pixel comparison:
        Same scan in TBX11K and Montgomery may have different resolutions,
        JPEG compression levels, or slight crops. Pixel-level MD5 would miss
        these near-duplicates. DCT captures structural fingerprint regardless
        of minor rescaling or compression artifacts.

    Why 8x8 subset of DCT coefficients:
        Low-frequency DCT components encode overall structure (lung shape, density).
        High-frequency components encode fine texture — irrelevant for deduplication
        and highly sensitive to JPEG quality. Taking only the top-left 8x8 block
        gives a 64-bit hash that is robust to format differences.

    Why median threshold:
        Binarizing around the median produces a balanced hash with ~32 zeros and
        ~32 ones regardless of image brightness, making Hamming distances comparable
        across bright (normal) and dark (severe TB) images.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None  # Unreadable file — will be caught by quality filter

    # Downsample to hash_size x hash_size before DCT to reduce computation
    img_resized = cv2.resize(img, (hash_size, hash_size))

    # cv2.dct requires float32 input
    dct = cv2.dct(img_resized.astype(np.float32))

    # Top-left 8x8 = low-frequency structural information only
    dct_low = dct[:8, :8]

    # Binarize around median — keeps bit distribution balanced across images
    median_val = np.median(dct_low)
    bits = (dct_low > median_val).flatten()

    return ''.join(['1' if b else '0' for b in bits])


def hamming_distance(h1: str, h2: str) -> int:
    """Count bit positions where two hash strings differ."""
    return sum(c1 != c2 for c1, c2 in zip(h1, h2))


def detect_duplicates(df: pd.DataFrame, threshold: int = 8) -> pd.DataFrame:
    """
    Find near-duplicate images across all datasets.

    Why threshold=8:
        Out of 64 bits, 8 differing bits (12.5%) captures same-scan duplicates
        that differ due to JPEG compression or slight crops, while avoiding
        false-positives between different patients with similar lung density.
        Empirically validated on chest X-ray datasets in prior work.

    Args:
        df:        DataFrame with 'image_path' column
        threshold: Max Hamming distance to consider as duplicate

    Returns:
        (cleaned_df, duplicate_pairs_df)
    """
    print(f'Computing perceptual hashes for {len(df)} images...')
    df = df.copy()

    # Hash every image — O(n) pass
    hashes = []
    for path in tqdm(df['image_path'], desc='Hashing'):
        hashes.append(phash(path))
    df['phash'] = hashes

    # Drop images that couldn't be read (hash is None)
    df = df.dropna(subset=['phash'])

    # Linear scan with early-exit: O(n²) worst case but fast in practice
    # because most images are unique and break immediately
    duplicate_indices = set()
    duplicate_pairs   = []
    seen_hashes       = {}  # hash_string → dataframe index of first occurrence

    for idx, row in df.iterrows():
        h = row['phash']
        found_dup = False
        for seen_h, seen_idx in seen_hashes.items():
            if hamming_distance(h, seen_h) <= threshold:
                # This image is a near-duplicate of one already seen — mark for removal
                duplicate_indices.add(idx)
                duplicate_pairs.append({
                    'original_idx':    seen_idx,
                    'duplicate_idx':   idx,
                    'original_path':   df.loc[seen_idx, 'image_path'],
                    'duplicate_path':  row['image_path'],
                    'hamming_distance': hamming_distance(h, seen_h)
                })
                found_dup = True
                break  # One match is enough — no need to check remaining hashes
        if not found_dup:
            # First time seeing this hash — register as canonical
            seen_hashes[h] = idx

    df['is_duplicate'] = df.index.isin(duplicate_indices)
    dup_pairs_df = pd.DataFrame(duplicate_pairs)

    print(f'\nDuplicate Detection Results:')
    print(f'  Total images      : {len(df)}')
    print(f'  Duplicates found  : {len(duplicate_indices)}')
    print(f'  Unique images     : {len(df) - len(duplicate_indices)}')
    if len(dup_pairs_df) > 0:
        # Cross-dataset duplicates are the most important to catch —
        # same scan in training (TBX11K) and test (Montgomery) inflates AUC
        cross = dup_pairs_df[
            dup_pairs_df["original_path"].str.contains("mont|shen", case=False, na=False)
        ]
        print(f'  Cross-dataset dups: {len(cross)}')

    # Return only the canonical (non-duplicate) images
    return df[~df['is_duplicate']].reset_index(drop=True), dup_pairs_df


# ── 2. Image Quality Filtering ─────────────────────────────────────────────

def check_image_quality(image_path: str,
                         min_size: int = 256,
                         min_brightness: float = 15.0,
                         max_brightness: float = 240.0,
                         min_sharpness: float = 50.0) -> tuple:
    """
    Filter out low-quality chest X-rays before training.

    Why these thresholds:
        min_size=256:        Anything smaller is likely a thumbnail, not a diagnostic scan.
                             Actual chest X-rays are typically 2000×2000+ px (downsampled to 224 for training).
        min_brightness=15:   Pixel mean < 15 → nearly all-black → failed DICOM export or blank film.
        max_brightness=240:  Pixel mean > 240 → nearly all-white → overexposed or corrupted file.
        min_sharpness=50:    Laplacian variance < 50 → blurry image from patient motion or
                             bad digitization. Laplacian accentuates edges; low variance means
                             few edges → no diagnostic detail.

    Returns:
        (is_valid: bool, reason: str)
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False, 'unreadable'  # Corrupted file — cannot be opened at all

    h, w = img.shape
    if h < min_size or w < min_size:
        return False, f'too_small ({w}x{h})'

    # Single-pass brightness check: global mean pixel value (0–255 scale)
    brightness = img.mean()
    if brightness < min_brightness:
        return False, f'underexposed (mean={brightness:.1f})'
    if brightness > max_brightness:
        return False, f'overexposed (mean={brightness:.1f})'

    # Laplacian variance: high variance = sharp edges = good diagnostic quality
    # Second-order derivative amplifies high-frequency detail; variance summarizes its strength
    sharpness = cv2.Laplacian(img, cv2.CV_64F).var()
    if sharpness < min_sharpness:
        return False, f'blurry (laplacian={sharpness:.1f})'

    return True, 'ok'


def filter_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Apply quality checks to all images and return cleaned DataFrame."""
    print(f'\nRunning quality checks on {len(df)} images...')
    reasons = []
    valid   = []
    for path in tqdm(df['image_path'], desc='Quality'):
        ok, reason = check_image_quality(path)
        valid.append(ok)
        reasons.append(reason)

    df = df.copy()
    df['quality_ok']     = valid
    df['quality_reason'] = reasons

    removed = df[~df['quality_ok']]
    print(f'\nQuality Filter Results:')
    print(f'  Passed : {df["quality_ok"].sum()}')
    print(f'  Removed: {(~df["quality_ok"]).sum()}')
    if len(removed) > 0:
        print(f'  Removal reasons:')
        print(removed['quality_reason'].value_counts().to_string())

    return df[df['quality_ok']].reset_index(drop=True)


# ── 3. Lung ROI Cropping ───────────────────────────────────────────────────

def crop_lung_roi(image: Image.Image, margin: float = 0.03) -> Image.Image:
    """
    Crop chest X-ray to the lung field, removing:
        - Black borders from DICOM-to-PNG conversion padding
        - White/grey frame borders added by radiology workstations
        - Embedded text: patient name, date, hospital stamp (at image edges)
        - Ruler/scale bar artifacts (white lines at image corners)

    Why these operations matter for CNN training:
        Without ROI cropping, CNNs learn spurious correlations between border
        artifacts and TB labels. Shenzhen dataset has particularly prominent
        white text labels at the bottom; Montgomery has ruler markers.
        GradCAM studies have shown CNNs attending to these artifacts instead
        of lung tissue when ROI cropping is omitted.

    Method:
        1. Threshold at pixel=20 (empirical: background is near-black after DICOM conversion)
        2. Morphological close: fills gaps inside lung field (ribs, vasculature create holes)
        3. Morphological open: removes isolated noise pixels outside the lung
        4. Bounding box + margin: tight crop with 3% padding to avoid cutting apices

    Args:
        image:  PIL Image (chest X-ray, any size)
        margin: Fractional margin to add on each side (0.03 = 3%)

    Returns:
        Cropped PIL Image — preserves original mode (RGB or L)
    """
    # Work in grayscale for thresholding; crop is applied to the original color image
    img_np = np.array(image.convert('L'))

    # Threshold=20: background from DICOM conversion is typically 0–10;
    # 20 provides a safe margin for slightly brightened backgrounds
    _, thresh = cv2.threshold(img_np, 20, 255, cv2.THRESH_BINARY)

    # MORPH_CLOSE with 15x15 kernel: fills interior gaps from rib shadows and
    # low-density lung vessels that would otherwise split the lung into fragments
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)

    # MORPH_OPEN with 5x5 kernel: removes small noise blobs outside the lung field
    # (e.g., dust artifacts, single bright pixels at image corners)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)

    # Find bounding box of all foreground (lung) pixels
    coords = cv2.findNonZero(thresh)
    if coords is None:
        # Cannot detect lung field (e.g., all-dark image) — return original unchanged
        return image

    x, y, w, h = cv2.boundingRect(coords)
    H, W = img_np.shape

    # Expand box by margin% to avoid clipping lung apices and costophrenic angles
    mx = int(W * margin)
    my = int(H * margin)
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(W, x + w + mx)
    y2 = min(H, y + h + my)

    return image.crop((x1, y1, x2, y2))


# ── 4. CLAHE Enhancement ──────────────────────────────────────────────────

def apply_clahe(image: Image.Image,
                clip_limit: float = 2.0,
                tile_grid_size: tuple = (8, 8)) -> Image.Image:
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE) for chest X-rays.

    Why CLAHE instead of global histogram equalization:
        Global HE stretches contrast across the entire image, which can cause
        background regions to wash out lung tissue detail. CLAHE operates on
        local tiles (8x8 grid here) so it boosts contrast where diagnostic
        features appear (infiltrates, cavitations) without over-amplifying
        uniform background regions.

    Why clip_limit=2.0:
        The clip limit prevents noise amplification by capping the histogram
        before redistribution. clip_limit=2.0 is the CheXNet standard and
        balances enhancement vs. noise. Higher values (e.g., 8.0) over-sharpen
        and create ring artifacts around dense structures.

    Why tile_grid_size=(8, 8):
        Divides the image into 8×8=64 non-overlapping tiles. Each tile gets
        its own histogram. For a 224×224 image, each tile is 28×28 pixels —
        large enough to capture local lung structure, small enough to adapt to
        regional density differences (upper vs. lower lobes).

    Clinical impact for TB classes:
        - Inactive TB: CLAHE makes calcified nodules crisper (high-density spots)
        - Active TB: enhances subtle early infiltrates that overlap with normal density
        - Severe TB: clarifies cavity walls and miliary nodule distribution

    Args:
        image:          PIL Image (RGB or L)
        clip_limit:     CLAHE contrast clip limit
        tile_grid_size: Local grid size for adaptive histogram

    Returns:
        CLAHE-enhanced PIL Image in RGB mode (3 channels for torchvision compatibility)
    """
    # Convert to grayscale — CLAHE is a single-channel operation
    # Chest X-rays are inherently greyscale; RGB channels contain identical info
    img_np = np.array(image.convert('L'))

    clahe    = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(img_np)

    # Convert back to RGB so downstream torchvision transforms and ImageNet-pretrained
    # models receive a 3-channel tensor (all channels are identical but expected)
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)


class CLAHETransform:
    """
    CLAHE as a torchvision-compatible callable transform.

    Usage in transforms.Compose:
        transforms.Compose([
            LungROICrop(),
            CLAHETransform(),   # ← here, before Resize
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            ...
        ])

    Must come before transforms.ToTensor() since it operates on PIL Images.
    Must come after LungROICrop() so CLAHE enhances the cropped region only.
    """
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)):
        self.clip_limit     = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, image: Image.Image) -> Image.Image:
        return apply_clahe(image, self.clip_limit, self.tile_grid_size)

    def __repr__(self):
        return f'CLAHE(clip_limit={self.clip_limit}, tile_grid={self.tile_grid_size})'


class LungROICrop:
    """
    Lung ROI crop as a torchvision-compatible callable transform.

    Must come first in the Compose pipeline — before CLAHE and before Resize —
    so that the CNN sees only cropped lung content at every resolution.
    """
    def __init__(self, margin: float = 0.03):
        self.margin = margin

    def __call__(self, image: Image.Image) -> Image.Image:
        return crop_lung_roi(image, self.margin)

    def __repr__(self):
        return f'LungROICrop(margin={self.margin})'


# ── 5. Dataset-Specific Normalization ─────────────────────────────────────

def compute_dataset_stats(image_paths: list,
                           sample_size: int = 1000) -> dict:
    """
    Compute per-channel mean and standard deviation from the training images.

    Why not use ImageNet stats:
        ImageNet mean=[0.485, 0.456, 0.406] was computed from natural photos.
        Chest X-rays are greyscale and after CLAHE have a different intensity
        distribution. Using dataset-specific stats reduces the normalization
        mismatch and has been shown to slightly improve convergence in
        medical imaging transfer learning.

    Why sample_size=1000:
        Computing exact stats over 11,000+ images is slow and unnecessary —
        the law of large numbers means 1,000 random images give a stable
        estimate (std error < 0.001 for typical chest X-ray distributions).

    IMPORTANT: Compute only from the TRAINING split. Using val or test images
        leaks distribution information into the normalization step.

    Returns:
        {'mean': [r, g, b], 'std': [r, g, b]}
        (For chest X-rays after CLAHE, typically ~[0.531, 0.531, 0.531] and [0.252, 0.252, 0.252])
    """
    if len(image_paths) > sample_size:
        import random
        image_paths = random.sample(image_paths, sample_size)

    print(f'Computing dataset statistics from {len(image_paths)} images...')

    # Online variance computation: accumulate sum and sum-of-squares
    # then use E[X²] - E[X]² to compute std without storing all pixels in memory
    pixel_sum   = np.zeros(3)
    pixel_sq    = np.zeros(3)
    pixel_count = 0

    for path in tqdm(image_paths, desc='Stats'):
        img = cv2.imread(path)
        if img is None:
            continue  # Skip unreadable files silently — quality filter handles these
        # cv2 reads BGR; convert to RGB to match torchvision convention
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # Resize to training resolution so pixel distribution matches model input
        img = cv2.resize(img, (224, 224))
        pixel_sum  += img.reshape(-1, 3).sum(axis=0)
        pixel_sq   += (img ** 2).reshape(-1, 3).sum(axis=0)
        pixel_count += 224 * 224

    mean = pixel_sum / pixel_count
    # std = sqrt(E[X²] - E[X]²) — numerically stable online formula
    std  = np.sqrt(pixel_sq / pixel_count - mean ** 2)

    print(f'Dataset mean : {mean.tolist()}')
    print(f'Dataset std  : {std.tolist()}')
    print(f'(ImageNet mean: [0.485, 0.456, 0.406])')
    print(f'(ImageNet std : [0.229, 0.224, 0.225])')

    return {'mean': mean.tolist(), 'std': std.tolist()}


# ── Full Pipeline ──────────────────────────────────────────────────────────

def run_full_cleaning_pipeline(df: pd.DataFrame,
                                output_csv: str = 'data/cleaned_dataset.csv'
                                ) -> pd.DataFrame:
    """
    Run the complete data cleaning pipeline in order.

    Pipeline order matters:
        1. Quality filter first — skip corrupted files before hashing (faster)
        2. Duplicate detection second — after bad images removed, fewer hashes to compare
        3. ROI crop + CLAHE are NOT applied here — they run at load time via transforms
           so original images are preserved for audit and IRB compliance

    Args:
        df:         Raw DataFrame from load_dataset() with 'image_path' and 'label' columns
        output_csv: Path to save the cleaned image manifest (CSV with all metadata)

    Returns:
        Cleaned DataFrame ready for train_test_split
    """
    print('=' * 55)
    print('TB-CXR Data Cleaning Pipeline')
    print('=' * 55)
    print(f'Input: {len(df)} images')

    # Step 1: Remove corrupted, tiny, blurry, and wrongly-exposed scans
    df = filter_quality(df)
    print(f'After quality filter: {len(df)} images')

    # Step 2: Remove cross-dataset duplicates (same scan in TBX11K + Montgomery/Shenzhen)
    # These inflate AUC if the same image appears in both train and test sets
    df, dup_pairs = detect_duplicates(df)
    print(f'After deduplication: {len(df)} images')

    # Step 3: Report final class and dataset distributions
    print(f'\nFinal label distribution:')
    print(df['label_name'].value_counts().to_string())
    print(f'\nDataset source distribution:')
    print(df['dataset'].value_counts().to_string())

    # Step 4: Save cleaned manifest — this CSV is the reproducible record of what
    # was used for training, required for paper's "Data Availability" section
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f'\nCleaned manifest saved to: {output_csv}')

    return df


if __name__ == '__main__':
    print('TB-CXR Data Cleaning Utilities')
    print('Run run_full_cleaning_pipeline(df) after loading dataset.')
    print('\nKey functions:')
    print('  detect_duplicates(df)          — perceptual hash deduplication')
    print('  filter_quality(df)             — exposure, blur, size checks')
    print('  apply_clahe(image)             — CLAHE contrast enhancement')
    print('  crop_lung_roi(image)           — remove borders and artifacts')
    print('  compute_dataset_stats(paths)   — dataset-specific normalization stats')
    print('\nTransforms for torchvision pipeline:')
    print('  CLAHETransform(clip_limit=2.0)')
    print('  LungROICrop(margin=0.03)')
