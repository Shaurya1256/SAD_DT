
# SAD-DT: Saliency-aware Anomaly Detection for Digital Twins

A lightweight PyTorch pipeline for anomaly detection that fuses saliency-weighted reconstruction
error with feature-space discrepancy. Designed to fit Smart Digital Twin (SDT) use-cases where
real-time and interpretability matter.

## Quick Start

```bash
# create/activate env (example)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# install
pip install -r requirements.txt

# dataset: download MVTec AD and point --data-dir to the root
python src/sad_dt.py --data-dir /path/to/mvtec --category bottle --epochs 30 --batch-size 16
```

Artifacts (saved to working directory):
- `viz_epoch_*.png` : sample input / reconstruction / anomaly heatmap

## Project Layout
```
SAD-DT_Project/
  ├─ src/
  │   └─ sad_dt.py
  ├─ docs/
  │   ├─ paper_draft.md
  │   ├─ case_study.md
  │   ├─ presentation_script.md
  │   └─ journal_suggestions.md
  ├─ requirements.txt
  └─ README.md
```

## License
MIT (for the provided code). Check dataset licenses separately.
