"""
Label preparation script for 4-class TB severity classification.
Maps radiology report text to CDC immigration medical exam classes.

Classes:
    0 - Normal        (USCIS: Cleared)
    1 - Inactive TB   (USCIS: Class B2 - cleared with follow-up)
    2 - Active TB     (USCIS: Class B1 - needs sputum test, evaluation)
    3 - Severe TB     (USCIS: Class A  - not cleared, treatment required)
"""

import os
import re
import pandas as pd

# ── Keyword rules for severity grading from radiology reports ──────────────
SEVERE_KEYWORDS = [
    "cavit", "miliary", "bilateral extensive", "large effusion",
    "extensive consolidation", "disseminated", "progressive",
    "bilateral infiltrat", "massive"
]

ACTIVE_KEYWORDS = [
    "infiltrat", "consolidat", "ground glass", "opacity",
    "exudate", "airspace", "active", "pneumonic", "lesion"
]

INACTIVE_KEYWORDS = [
    "calcif", "fibro", "scar", "healed", "old", "chronic",
    "nodule", "granuloma", "pleural thicken", "residual"
]


def classify_from_report(report_text: str, has_tb: bool) -> int:
    if not has_tb or not isinstance(report_text, str):
        return 0  # Normal

    text = report_text.lower()

    if any(kw in text for kw in SEVERE_KEYWORDS):
        return 3  # Severe / Cavitary (Class A)

    if any(kw in text for kw in ACTIVE_KEYWORDS):
        return 2  # Active without cavitation (Class B1)

    if any(kw in text for kw in INACTIVE_KEYWORDS):
        return 1  # Inactive / Healed (Class B2)

    return 2  # Default TB positive to Active if report unclear


LABEL_NAMES = {
    0: "Normal",
    1: "Inactive_TB",
    2: "Active_TB",
    3: "Severe_TB"
}

USCIS_CLASS = {
    0: "Cleared",
    1: "Class B2 - Cleared with follow-up",
    2: "Class B1 - Further evaluation required",
    3: "Class A - Not cleared, treatment required"
}


def prepare_montgomery(data_dir: str) -> pd.DataFrame:
    """
    Montgomery County Chest X-ray dataset.
    Expects:
        data_dir/CXR_png/          - X-ray images
        data_dir/ClinicalReadings/ - radiology reports (.txt)
        data_dir/Montgomery_TB_and_NonTB.csv (if available)
    """
    rows = []
    cxr_dir = os.path.join(data_dir, "CXR_png")
    report_dir = os.path.join(data_dir, "ClinicalReadings")

    if not os.path.exists(cxr_dir):
        print(f"CXR directory not found: {cxr_dir}")
        return pd.DataFrame()

    for fname in sorted(os.listdir(cxr_dir)):
        if not fname.endswith(".png"):
            continue

        img_path = os.path.join(cxr_dir, fname)
        base = fname.replace(".png", "")
        has_tb = base.startswith("MCUCXR") and "1" in base[-1]

        report_text = ""
        report_path = os.path.join(report_dir, base + ".txt")
        if os.path.exists(report_path):
            with open(report_path, "r", errors="ignore") as f:
                report_text = f.read()

        label = classify_from_report(report_text, has_tb)
        rows.append({
            "image_path": img_path,
            "label": label,
            "label_name": LABEL_NAMES[label],
            "uscis_class": USCIS_CLASS[label],
            "dataset": "Montgomery",
            "report": report_text[:200]
        })

    return pd.DataFrame(rows)


def prepare_shenzhen(data_dir: str) -> pd.DataFrame:
    """
    Shenzhen Hospital Chest X-ray dataset.
    Expects:
        data_dir/CXR_png/          - X-ray images (0=normal, 1=TB in filename)
        data_dir/ClinicalReadings/ - radiology reports (.txt)
    """
    rows = []
    cxr_dir = os.path.join(data_dir, "CXR_png")
    report_dir = os.path.join(data_dir, "ClinicalReadings")

    if not os.path.exists(cxr_dir):
        print(f"CXR directory not found: {cxr_dir}")
        return pd.DataFrame()

    for fname in sorted(os.listdir(cxr_dir)):
        if not fname.endswith(".png"):
            continue

        img_path = os.path.join(cxr_dir, fname)
        has_tb = fname.split("_")[-1].replace(".png", "") == "1"

        report_text = ""
        base = fname.replace(".png", "")
        report_path = os.path.join(report_dir, base + ".txt")
        if os.path.exists(report_path):
            with open(report_path, "r", errors="ignore") as f:
                report_text = f.read()

        label = classify_from_report(report_text, has_tb)
        rows.append({
            "image_path": img_path,
            "label": label,
            "label_name": LABEL_NAMES[label],
            "uscis_class": USCIS_CLASS[label],
            "dataset": "Shenzhen",
            "report": report_text[:200]
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("TB Severity Label Preparation")
    print("=" * 50)
    print("\nClass definitions:")
    for k, v in LABEL_NAMES.items():
        print(f"  {k} - {v:20s} -> {USCIS_CLASS[k]}")
    print("\nTo use:")
    print("  from prepare_labels import prepare_montgomery, prepare_shenzhen")
    print("  df_mont = prepare_montgomery('/path/to/montgomery')")
    print("  df_shen = prepare_shenzhen('/path/to/shenzhen')")
    print("  combined = pd.concat([df_mont, df_shen])")
