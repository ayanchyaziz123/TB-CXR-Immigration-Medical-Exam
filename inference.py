"""
TB-CXR Inference Script
Civil surgeon decision support for USCIS Form I-693 immigration medical examination.

Usage:
    python inference.py --image path/to/chest_xray.jpg --model checkpoints/densenet121/best.pt
    python inference.py --image path/to/chest_xray.jpg --model checkpoints/densenet121/best.pt --uncertainty
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms, models
from PIL import Image

# ── Constants ──────────────────────────────────────────────────────────────
LABEL_NAMES   = ['Normal', 'Inactive_TB', 'Active_TB', 'Severe_TB']
USCIS_LABELS  = ['Cleared', 'Class B2', 'Class B1', 'Class A']
USCIS_ACTIONS = [
    'Applicant cleared for immigration processing.',
    'Class B2: Cleared with mandatory 30-day follow-up evaluation.',
    'Class B1: Further sputum evaluation required before clearance.',
    'Class A: Not cleared. Treatment required before immigration processing.'
]
IMG_SIZE = 224

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── Model ──────────────────────────────────────────────────────────────────
def build_densenet121(num_classes=4, dropout=0.5):
    model = models.densenet121(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(model.classifier.in_features, num_classes)
    )
    return model


def load_model(ckpt_path: str, device: torch.device):
    model = build_densenet121()
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model


# ── Inference ──────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_single(model, img_path: str, device: torch.device) -> dict:
    img = Image.open(img_path).convert('RGB')
    img_t = val_transforms(img).unsqueeze(0).to(device)
    logits = model(img_t)
    probs  = F.softmax(logits, dim=1).cpu().numpy()[0]
    pred   = probs.argmax()
    return {
        'prediction':   int(pred),
        'label':        LABEL_NAMES[pred],
        'uscis_class':  USCIS_LABELS[pred],
        'action':       USCIS_ACTIONS[pred],
        'confidence':   float(probs[pred]),
        'probabilities': {LABEL_NAMES[i]: float(probs[i]) for i in range(4)},
        'flag':         probs[pred] < 0.80 or pred >= 2
    }


def predict_mc_dropout(model, img_path: str, device: torch.device, n_samples=30) -> dict:
    """Monte Carlo Dropout uncertainty-aware prediction."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout): m.train()

    img   = Image.open(img_path).convert('RGB')
    img_t = val_transforms(img).unsqueeze(0).to(device)

    probs_all = []
    with torch.no_grad():
        for _ in range(n_samples):
            p = F.softmax(model(img_t), dim=1).cpu().numpy()[0]
            probs_all.append(p)

    probs_all  = np.stack(probs_all)
    mean_probs = probs_all.mean(axis=0)
    std_probs  = probs_all.std(axis=0)
    pred       = mean_probs.argmax()
    confidence = mean_probs[pred]
    entropy    = -np.sum(mean_probs * np.log(mean_probs + 1e-8))

    return {
        'prediction':   int(pred),
        'label':        LABEL_NAMES[pred],
        'uscis_class':  USCIS_LABELS[pred],
        'action':       USCIS_ACTIONS[pred],
        'confidence':   float(confidence),
        'uncertainty':  float(entropy),
        'std_probs':    {LABEL_NAMES[i]: float(std_probs[i]) for i in range(4)},
        'mean_probs':   {LABEL_NAMES[i]: float(mean_probs[i]) for i in range(4)},
        'flag':         confidence < 0.80 or pred >= 2,
        'n_samples':    n_samples
    }


def print_result(result: dict, use_mc: bool = False):
    print('\n' + '=' * 55)
    print('  TB-CXR USCIS Form I-693 Triage Report')
    print('=' * 55)
    print(f'  Classification : {result["label"]}')
    print(f'  USCIS Class    : {result["uscis_class"]}')
    print(f'  Confidence     : {result["confidence"]:.1%}')
    if use_mc:
        print(f'  Uncertainty    : {result["uncertainty"]:.4f} (entropy)')
    print(f'\n  ACTION: {result["action"]}')
    print()
    print('  Class Probabilities:')
    probs_key = 'mean_probs' if use_mc else 'probabilities'
    for label, prob in result[probs_key].items():
        bar = '█' * int(prob * 20)
        print(f'    {label:<15} {prob:.3f}  {bar}')

    if result['flag']:
        print()
        print('  ⚠  FLAGGED FOR CIVIL SURGEON REVIEW')
        if result['confidence'] < 0.80:
            print('     Reason: Model confidence below threshold (< 80%)')
        if result['prediction'] >= 2:
            print('     Reason: Active or Severe TB prediction requires human confirmation')
    else:
        print()
        print('  ✓  AI-assisted triage: LOW RISK (civil surgeon review optional)')

    print('=' * 55)
    print('  DISCLAIMER: Research prototype. All Class B1, B2, Class A')
    print('  predictions must be confirmed by a licensed civil surgeon.')
    print('=' * 55 + '\n')


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='TB-CXR: USCIS Form I-693 Triage Inference'
    )
    parser.add_argument('--image',       required=True,  help='Path to chest X-ray image')
    parser.add_argument('--model',       required=True,  help='Path to model checkpoint (.pt)')
    parser.add_argument('--uncertainty', action='store_true',
                        help='Use Monte Carlo Dropout uncertainty estimation')
    parser.add_argument('--mc_samples',  type=int, default=30,
                        help='Number of MC Dropout samples (default: 30)')
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f'Error: Image not found: {args.image}'); return
    if not os.path.exists(args.model):
        print(f'Error: Checkpoint not found: {args.model}'); return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Loading model from {args.model} on {device}...')
    model = load_model(args.model, device)

    if args.uncertainty:
        print(f'Running MC Dropout inference ({args.mc_samples} samples)...')
        result = predict_mc_dropout(model, args.image, device, args.mc_samples)
    else:
        result = predict_single(model, args.image, device)

    print_result(result, use_mc=args.uncertainty)


if __name__ == '__main__':
    main()
