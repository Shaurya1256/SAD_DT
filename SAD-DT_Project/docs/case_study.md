
# Case Study: Saliency-aware Anomaly Detection for Digital Twins

## Problem Statement & Objectives
Detect and localize visual defects in industrial parts for Smart Digital Twins with
near-real-time inference and interpretable heatmaps.

## Data Preprocessing
- Resize to 256×256, normalize to [0,1].
- Optional synthetic anomalies (texture overlays, cutouts) for augmentation.
- Train on normal images only (unsupervised AD).

## Model Selection & Development
- Attention Conv-AE for reconstruction.
- ResNet18 features for semantic discrepancy.
- Saliency weighting (Laplacian-based) to prioritize critical regions.

## Visualizations & Insights
- Provide input, reconstruction, and anomaly heatmaps per sample.
- Observe concentration of activations on true defect areas.

## Recommendations
- Deploy on edge GPU; set threshold via validation ROC.
- For production, replace simple saliency with a pretrained saliency model.
