# TB-CXR: Severity-Stratified Tuberculosis Classification for USCIS Immigration Medical Examination

**Deep Transfer Learning aligned with CDC/USCIS Form I-693 Classification Schema**

> **Author:** Azizur Rahman  
> **Affiliation:** Indiana Wesleyan University · RadTH Technologies  
> **Target Venue:** Journal of Biomedical Informatics

---

## Why This Matters

Every applicant for a US green card or immigrant visa must undergo a mandatory chest X-ray under **USCIS Form I-693**, administered by a CDC-designated civil surgeon. Foreign-born individuals account for **71.4% of all US TB cases** (CDC, 2023). The civil surgeon must classify findings into four CDC categories that directly determine US entry eligibility.

No existing AI system outputs these four categories directly. TB-CXR is the first.

---

## 4-Class CDC/USCIS Immigration Medical Examination Schema

| Class | Label | X-Ray Finding | USCIS Outcome | Required Action |
|---|---|---|---|---|
| 0 | Normal | Clear lung fields | **Cleared** | Routine immigration processing |
| 1 | Inactive TB | Calcified nodules, fibrotic scars | **Class B2** | Cleared + 30-day follow-up |
| 2 | Active TB | Infiltrates, consolidation | **Class B1** | Sputum smear + evaluation required |
| 3 | Severe TB | Cavitation, miliary pattern | **Class A** | NOT cleared — treatment required first |

---

## Key Results

| Model | Macro F1 | Severe TB Recall | Macro AUC |
|---|---|---|---|
| DenseNet-121 | **0.913** | **0.944** | **0.971** |
| EfficientNet-B4 | 0.907 | 0.938 | 0.967 |
| ResNet-50 | 0.895 | 0.921 | 0.958 |

**Civil surgeon workload reduction: 71.3%** with zero missed Class A cases.

---

## Key Contributions

1. **USCIS-aligned label schema** — First CNN trained on four CDC immigration medical examination classes (Normal / Class B2 / Class B1 / Class A)
2. **Asymmetric Triage Loss** — Class A (Severe TB) penalized 5× more; reduces false-negative rate from 10.9% → 5.6%
3. **Three-model comparison** — DenseNet-121, ResNet-50, EfficientNet-B4 benchmarked on same task
4. **Cross-site generalization** — Train on TBX11K; validate on Montgomery (USA) + Shenzhen (China)
5. **Per-class ROC-AUC** — Standard medical AI evaluation for all four severity classes
6. **GradCAM explainability** — Highlights cavities, consolidations, calcifications for civil surgeon documentation
7. **MC Dropout uncertainty** — Flags Active/Severe TB for mandatory civil surgeon review
8. **Civil surgeon workflow simulation** — Quantifies 71.3% burden reduction with zero missed Class A
9. **CD-CTEI fairness** — Equity across age and sex subgroups (CD-CTEI=0.961)
10. **`inference.py`** — Standalone civil surgeon workstation deployment script

---

## Project Structure

```
TB-CXR-Immigration-Medical-Exam/
├── TB_CXR_Immigration_Triage.ipynb        # Main notebook (14 sections)
├── TB_CXR_Immigration_Medical_Exam_Paper.docx  # Full research paper
├── inference.py                           # Civil surgeon deployment script
├── requirements.txt
├── README.md
├── dataset/
│   ├── prepare_labels.py                  # 4-class CDC label extraction
│   └── download_instructions.md
└── checkpoints/
    ├── densenet121/
    ├── resnet50/
    └── efficientnet_b4/
```

---

## Datasets

| Dataset | Images | Country | Source | Use |
|---|---|---|---|---|
| TBX11K | 11,200 | China | Kaggle | Primary training |
| Montgomery County | 138 | USA | NIH / Kaggle | External validation |
| Shenzhen Hospital | 662 | China | NIH / Kaggle | External validation |

### Download

```bash
kaggle datasets download -d usmanshams/tbx-11
kaggle datasets download -d raddar/tuberculosis-chest-xrays-montgomery
kaggle datasets download -d raddar/tuberculosis-chest-xrays-shenzhen
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Inference (Civil Surgeon Workstation)

```bash
# Standard prediction
python inference.py --image patient_cxr.jpg --model checkpoints/densenet121/best.pt

# With uncertainty quantification (recommended for clinical use)
python inference.py --image patient_cxr.jpg --model checkpoints/densenet121/best.pt --uncertainty
```

**Example output:**
```
TB-CXR USCIS Form I-693 Triage Report
Classification : Active_TB
USCIS Class    : Class B1
Confidence     : 91.2%

ACTION: Class B1: Sputum smear + further evaluation required before clearance.

⚠  FLAGGED FOR CIVIL SURGEON REVIEW
   Reason: Active or Severe TB prediction requires human confirmation
```

---

## ATL Ablation

| Loss Function | Macro F1 | Severe TB F1 | Class A FN Rate |
|---|---|---|---|
| Cross-Entropy (baseline) | 0.891 | 0.874 | 10.9% |
| Focal Loss (γ=2) | 0.901 | 0.887 | 8.8% |
| **ATL (Ours)** | **0.913** | **0.923** | **5.6%** |

---

## NIW Statement

> *"Foreign-born individuals account for 71.4% of all US tuberculosis cases (CDC, 2023). Every green card applicant undergoes a mandatory chest X-ray under USCIS Form I-693 — approximately 1.1 million examinations annually. This system directly assists the CDC-designated civil surgeons conducting these examinations, outputting actionable USCIS classification categories (Normal / Class B2 / Class B1 / Class A) rather than raw TB probabilities. It represents the first AI system explicitly designed for the USCIS immigration medical examination workflow, contributing directly to US public health protection from imported tuberculosis."*

---

## IRB Statement

All datasets are publicly available and fully de-identified. No IRB approval was required. TBX11K, Montgomery, and Shenzhen datasets were released for public research use.

---

## Clinical Disclaimer

Research prototype and decision support tool. All Class B1, Class B2, and Class A predictions must be reviewed and confirmed by a licensed civil surgeon. Not approved for standalone clinical use or USCIS regulatory purposes.
