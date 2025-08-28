
# SAD-DT: Saliency-aware Anomaly Detection for Smart Digital Twins

## Abstract
We present SAD-DT, a lightweight anomaly detection framework that combines
saliency-weighted reconstruction with feature-space discrepancy to improve
localization and robustness under limited labeled data—typical in Smart Digital Twins (SDTs).

## 1. Introduction
- Motivation: SDTs require reliable visual monitoring with low latency.
- Problem: Scarce anomalies, domain gaps, and region-importance (saliency) are not fully addressed.
- Contribution: (i) saliency-weighted multi-term loss, (ii) fusion of pixel & feature residuals,
(iii) lightweight attention AE for real-time, (iv) optional synthetic (DT) augmentation path.

## 2. Literature Review (short)
(Expand to 25+ references with DOI; some seed areas listed below.)
- Industrial visual anomaly detection (reconstruction- and feature-based methods).
- Saliency and task-aware compression/attention for vision.
- Domain adaptation with synthetic-to-real alignment in industrial vision.
- Smart Digital Twins and image-based monitoring.

## 3. Research Questions & Objectives
- RQ1: Does saliency-weighted training improve pixel-level AUROC vs. vanilla AE?
- RQ2: Does fusing pixel and feature residuals improve detection robustness?
- RQ3: How does the method trade accuracy vs. latency for SDT deployment?
**Objectives:** develop, implement, and evaluate SAD-DT on MVTec AD; compare with baselines.

## 4. Method
### 4.1 Model
- Attention-enabled Conv-AE (encoder/decoder) + pretrained ResNet18 feature extractor.
### 4.2 Loss
- L = λ_rec * SalW-MSE + λ_feat * SalW-feature-L2 (+ optional sim→real alignment).
### 4.3 Scoring
- Pixel anomaly map = residual + feature discrepancy (saliency-weighted); image score via pooling.

## 5. Experimental Setup
- Dataset: MVTec AD (15 classes); train on normal, test on mixed.
- Metrics: AUROC (image/pixel), latency (ms), params (MB).
- Baselines: vanilla AE; feature-only method (PaDiM-style); lightweight attention AE.

## 6. Results & Analysis
- Expect improved AUROC and sharper localization in salient regions.
- Include heatmaps and reconstructions (Figures).

## 7. Conclusion & Future Work
- SAD-DT offers a cost-effective, interpretable pipeline for SDTs.
- Future: stronger saliency, federated updates, and realistic DT synthetic data.

## References (seed list; replace/expand to 25+ with DOI)
- (Add IEEE/SCI/Scopus-indexed papers with DOI here; avoid conference-only)
