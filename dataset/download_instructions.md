# Dataset Download Instructions

## Step 1 — Create Kaggle Account
Go to kaggle.com, create free account, go to Settings → API → Create New Token → download kaggle.json

## Step 2 — Setup Kaggle in Colab
```python
from google.colab import files
files.upload()  # upload your kaggle.json

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

## Step 3 — Download Datasets

### TBX11K (Primary — 11,200 images)
```bash
!kaggle datasets download -d usmanshams/tbx-11
!unzip tbx-11.zip -d data/tbx11k
```

### Montgomery County (138 images — external validation)
```bash
!kaggle datasets download -d raddar/tuberculosis-chest-xrays-montgomery
!unzip tuberculosis-chest-xrays-montgomery.zip -d data/montgomery
```

### Shenzhen Hospital (662 images — external validation)
```bash
!kaggle datasets download -d raddar/tuberculosis-chest-xrays-shenzhen
!unzip tuberculosis-chest-xrays-shenzhen.zip -d data/shenzhen
```

## Expected Folder Structure After Download

```
data/
├── tbx11k/
│   ├── imgs/
│   │   ├── train/
│   │   │   ├── sick_non-tb/
│   │   │   ├── tb/
│   │   │   └── healthy/
│   │   └── test/
│   └── labels/
├── montgomery/
│   ├── CXR_png/
│   └── ClinicalReadings/
└── shenzhen/
    ├── CXR_png/
    └── ClinicalReadings/
```

## Label Mapping

Montgomery and Shenzhen include clinical reading text files (.txt) per image.
Run `prepare_labels.py` to extract 4-class severity labels from these reports.

```python
from dataset.prepare_labels import prepare_montgomery, prepare_shenzhen
df_mont = prepare_montgomery('data/montgomery')
df_shen = prepare_shenzhen('data/shenzhen')
```
