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
    Perceptual hash of a chest X-ray image.
    Resize to hash_size x hash_size, convert to grayscale, compare DCT frequencies.
    More robust than MD5 for near-duplicate detection (same scan, different format/crop).
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img_resized = cv2.resize(img, (hash_size, hash_size))
    dct         = cv2.dct(img_resized.astype(np.float32))
    dct_low     = dct[:8, :8]
    median_val  = np.median(dct_low)
    bits        = (dct_low > median_val).flatten()
    return ''.join(['1' if b else '0' for b in bits])


def hamming_distance(h1: str, h2: str) -> int:
    return sum(c1 != c2 for c1, c2 in zip(h1, h2))


def detect_duplicates(df: pd.DataFrame, threshold: int = 8) -> pd.DataFrame:
    """
    Find near-duplicate images across all datasets.
    threshold: max hamming distance to consider as duplicate (0=exact, 8=near-duplicate).

    Returns a DataFrame with duplicate pairs flagged.
    """
    print(f'Computing perceptual hashes for {len(df)} images...')
    df = df.copy()
    hashes = []
    for path in tqdm(df['image_path'], desc='Hashing'):
        hashes.append(phash(path))
    df['phash'] = hashes
    df = df.dropna(subset=['phash'])

    # Group by hash bucket for efficiency
    buckets = defaultdict(list)
    for idx, row in df.iterrows():
        key = row['phash'][:8]  # First 8 bits as bucket key
        buckets[key].append((idx, row['phash']))

    duplicate_indices = set()
    duplicate_pairs   = []

    seen_hashes = {}
    for idx, row in df.iterrows():
        h = row['phash']
        found_dup = False
        for seen_h, seen_idx in seen_hashes.items():
            if hamming_distance(h, seen_h) <= threshold:
                duplicate_indices.add(idx)
                duplicate_pairs.append({
                    'original_idx': seen_idx,
                    'duplicate_idx': idx,
                    'original_path': df.loc[seen_idx, 'image_path'],
                    'duplicate_path': row['image_path'],
                    'hamming_distance': hamming_distance(h, seen_h)
                })
                found_dup = True
                break
        if not found_dup:
            seen_hashes[h] = idx

    df['is_duplicate'] = df.index.isin(duplicate_indices)
    dup_pairs_df = pd.DataFrame(duplicate_pairs)

    print(f'\nDuplicate Detection Results:')
    print(f'  Total images      : {len(df)}')
    print(f'  Duplicates found  : {len(duplicate_indices)}')
    print(f'  Unique images     : {len(df) - len(duplicate_indices)}')
    if len(dup_pairs_df) > 0:
        print(f'  Cross-dataset dups: {len(dup_pairs_df[dup_pairs_df["original_path"].str.contains("mont|shen", case=False, na=False)])}')

    return df[~df['is_duplicate']].reset_index(drop=True), dup_pairs_df


# ── 2. Image Quality Filtering ─────────────────────────────────────────────

def check_image_quality(image_path: str,
                         min_size: int = 256,
                         min_brightness: float = 15.0,
                         max_brightness: float = 240.0,
                         min_sharpness: float = 50.0) -> tuple:
    """
    Filter out low-quality chest X-rays.

    Checks:
        - Minimum resolution (corrupted/thumbnail images)
        - Brightness range (overexposed: all white; underexposed: all black)
        - Sharpness via Laplacian variance (blurry scans)

    Returns: (is_valid: bool, reason: str)
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False, 'unreadable'

    h, w = img.shape
    if h < min_size or w < min_size:
        return False, f'too_small ({w}x{h})'

    brightness = img.mean()
    if brightness < min_brightness:
        return False, f'underexposed (mean={brightness:.1f})'
    if brightness > max_brightness:
        return False, f'overexposed (mean={brightness:.1f})'

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
        - Black borders from DICOM conversion
        - White/grey frame borders
        - Embedded text and patient labels (typically at image edges)
        - Ruler artifacts

    Method:
        1. Threshold to find non-background pixels
        2. Find largest bounding box
        3. Add small margin to avoid cutting lung edges

    Args:
        image:  PIL Image (chest X-ray)
        margin: Fractional margin to add around detected lung field

    Returns:
        Cropped PIL Image
    """
    img_np = np.array(image.convert('L'))  # Grayscale

    # Threshold: keep pixels brighter than near-black background
    _, thresh = cv2.threshold(img_np, 20, 255, cv2.THRESH_BINARY)

    # Morphological operations to fill gaps and remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

    # Find bounding box of all non-zero pixels
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return image  # Cannot detect — return original

    x, y, w, h = cv2.boundingRect(coords)
    H, W = img_np.shape

    # Add margin
    mx = int(W * margin)
    my = int(H * margin)
    x1 = max(0,     x - mx)
    y1 = max(0,     y - my)
    x2 = min(W, x + w + mx)
    y2 = min(H, y + h + my)

    return image.crop((x1, y1, x2, y2))


# ── 4. CLAHE Enhancement ──────────────────────────────────────────────────

def apply_clahe(image: Image.Image,
                clip_limit: float = 2.0,
                tile_grid_size: tuple = (8, 8)) -> Image.Image:
    """
    Contrast Limited Adaptive Histogram Equalization for chest X-rays.

    Standard in published chest X-ray AI papers (CheXNet, CheXpert baseline).
    Enhances local contrast, especially important for:
        - Early infiltrates (subtle opacity changes)
        - Calcified nodules in Inactive TB
        - Miliary pattern in Severe TB

    Args:
        image:         PIL Image
        clip_limit:    CLAHE clip limit (2.0 = standard for chest X-rays)
        tile_grid_size: Local window size (8x8 = standard)

    Returns:
        CLAHE-enhanced PIL Image (RGB, same size as input)
    """
    img_np   = np.array(image.convert('L'))  # Convert to grayscale
    clahe    = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(img_np)
    rgb      = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)


class CLAHETransform:
    """Callable transform compatible with torchvision.transforms.Compose."""
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit     = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, image: Image.Image) -> Image.Image:
        return apply_clahe(image, self.clip_limit, self.tile_grid_size)

    def __repr__(self):
        return f'CLAHE(clip_limit={self.clip_limit}, tile_grid={self.tile_grid_size})'


class LungROICrop:
    """Callable transform: crop to lung field before resizing."""
    def __init__(self, margin=0.03):
        self.margin = margin

    def __call__(self, image: Image.Image) -> Image.Image:
        return crop_lung_roi(image, self.margin)

    def __repr__(self):
        return f'LungROICrop(margin={self.margin})'


# ── 5. Dataset-Specific Normalization ─────────────────────────────────────

def compute_dataset_stats(image_paths: list,
                           sample_size: int = 1000) -> dict:
    """
    Compute per-channel mean and std from training images.
    Uses a random sample for efficiency.

    Standard practice: compute from training set only (no val/test leakage).
    Many chest X-ray papers show ImageNet stats are suboptimal;
    dataset-specific stats improve convergence.

    Returns:
        {'mean': [r, g, b], 'std': [r, g, b]}
    """
    if len(image_paths) > sample_size:
        import random
        image_paths = random.sample(image_paths, sample_size)

    print(f'Computing dataset statistics from {len(image_paths)} images...')
    pixel_sum  = np.zeros(3)
    pixel_sq   = np.zeros(3)
    pixel_count = 0

    for path in tqdm(image_paths, desc='Stats'):
        img = cv2.imread(path)
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = cv2.resize(img, (224, 224))
        pixel_sum  += img.reshape(-1, 3).sum(axis=0)
        pixel_sq   += (img ** 2).reshape(-1, 3).sum(axis=0)
        pixel_count += 224 * 224

    mean = pixel_sum / pixel_count
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
    Run the complete data cleaning pipeline in order:
        1. Quality filter
        2. Duplicate detection
        3. Save cleaned manifest

    ROI cropping and CLAHE are applied at load time via transforms (not saved to disk)
    to preserve original images for audit purposes.
    """
    print('=' * 55)
    print('TB-CXR Data Cleaning Pipeline')
    print('=' * 55)
    print(f'Input: {len(df)} images')

    # Step 1: Quality filter
    df = filter_quality(df)
    print(f'After quality filter: {len(df)} images')

    # Step 2: Duplicate detection
    df, dup_pairs = detect_duplicates(df)
    print(f'After deduplication: {len(df)} images')

    # Step 3: Class distribution report
    print(f'\nFinal label distribution:')
    print(df['label_name'].value_counts().to_string())
    print(f'\nDataset source distribution:')
    print(df['dataset'].value_counts().to_string())

    # Step 4: Save cleaned manifest
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
