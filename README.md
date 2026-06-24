# TB-CXR: Severity-Stratified Tuberculosis Classification for Immigration Medical Examination

**Deep Transfer Learning with GradCAM Explainability and Uncertainty Quantification**

> **Author:** Azizur Rahman  
> **Affiliation:** Indiana Wesleyan University · RadTH Technologies  
> **Target Venue:** Journal of Biomedical Informatics

---

## Overview

Every immigrant applying for a US green card must undergo a mandatory chest X-ray as part of the USCIS Form I-693 immigration medical examination. Civil surgeons classify these X-rays according to CDC tuberculosis guidelines. This system provides AI-assisted severity classification to support civil surgeons processing high-volume immigration medical exams.

---

## 4-Class CDC/USCIS Severity Mapping (Core Contribution)

| Class | Label | X-Ray Finding | USCIS Outcome |
|---|---|---|---|
| 0 | Normal | Clear lungs | Cleared |
| 1 | Inactive TB | Calcified nodules, fibrous scars | Class B2 — Cleared with 30-day follow-up |
| 2 | Active TB | Infiltrates, consolidation (no cavitation) | Class B1 — Further evaluation required |
| 3 | Severe TB | Cavitation, miliary pattern | Class A — Not cleared, treatment required |

---

## Key Contributions

1. **Novel label schema** — First CNN to map chest X-ray findings to USCIS immigration medical examination classes (Class A / B1 / B2 / Normal)
2. **Asymmetric Triage Loss** — Severe TB misclassification penalized 5× more than Normal
3. **Cross-dataset generalization** — Train on TBX11K, validate on Montgomery + Shenzhen (different hospitals, different countries)
4. **GradCAM explainability** — Highlights lung regions driving each classification for civil surgeon documentation
5. **Uncertainty quantification** — Monte Carlo Dropout flags low-confidence predictions for mandatory human review
6. **Cross-Demographic CTEI** — Fairness metric across age and sex subgroups (adapted from MiST framework)

---

## Project Structure

```
TB-CXR-Immigration-Medical-Exam/
├── TB_CXR_Immigration_Triage.ipynb   # Main notebook
├── README.md
├── dataset/
│   ├── prepare_labels.py              # 4-class label extraction from radiology reports
│   └── download_instructions.md
└── checkpoints/
    ├── densenet121/
    └── resnet50/
```

---

## Datasets

| Dataset | Images | Source | Use |
|---|---|---|---|
| TBX11K | 11,200 | Kaggle | Primary training |
| Montgomery County | 138 | NIH / Kaggle | External validation |
| Shenzhen Hospital | 662 | NIH / Kaggle | External validation |

### Download

```bash
# Kaggle CLI
kaggle datasets download -d usmanshams/tbx-11
kaggle datasets download -d raddar/tuberculosis-chest-xrays-montgomery
kaggle datasets download -d raddar/tuberculosis-chest-xrays-shenzhen
```

---

## Installation

```bash
pip install torch torchvision timm
pip install scikit-learn pandas numpy matplotlib seaborn tqdm
pip install grad-cam opencv-python pillow
```

---

## Models

| Model | Params | Role |
|---|---|---|
| DenseNet-121 | 8M | Primary (CheXNet standard) |
| ResNet-50 | 25M | Baseline comparison |

---

## NIW Statement

> *"This system directly assists the mandatory tuberculosis screening process required for all US immigration applicants under USCIS Form I-693, contributing to national public health protection by enabling more accurate and scalable AI-assisted triage of immigration medical examinations. Tuberculosis remains the leading infectious disease cause of death worldwide, and foreign-born individuals account for over 70% of US TB cases (CDC, 2023)."*

---

## Related Work

This paper is part of a research agenda on AI-powered clinical tools for immigrant populations:

| Paper | Task |
|---|---|
| MiST | Multilingual symptom triage for LEP patients |
| TB-CXR | Severity-stratified TB classification for immigration medical exams |

---

## Clinical Disclaimer

This system is a research prototype and decision support tool. All Class B1, Class B2, and Class A predictions must be reviewed by a licensed civil surgeon. Not approved for standalone clinical use.
