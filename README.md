# Tay ↔ Vietnamese Translation

Bidirectional translation product for Tay and Vietnamese. Includes training code, batch inference, Flask API, and web interface.

## Features
- Translate Tay→Vietnamese and Vietnamese→Tay
- JSON API and web UI
- Train seq2seq models from source code
- Fast EDA for data
- Docker and CI

## Structure
```
api/                 Flask app + UI
config/model.toml    Model name and configuration
translate/           Translation/training code converted to Tay
data/original/       CSV samples src,tgt
scripts/             EDA, utilities
experiments/         mBART50 training code based on original pipeline
```

## Installation
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Web
```
gunicorn -w 2 -b 0.0.0.0:8000 api.app:app
```
Open `http://localhost:8000`.

## API
`POST /api/translate`
```json
{ "text": "…", "direction": "tay2vi" }
```

## Train Using Original Pipeline
```
python experiments/mbart50/train_tay_vi.py --csv data/original/tay_vi_train.csv --out runs/mbart50 --epochs 1
```

## Weights
Add checkpoints to the **Releases** section of GitHub and link them here.

## NMT Experiments
Available models
- MBART50
- NLLB-200 distilled 600M
- mT5 small
- viT5 base
- BARTPho syllable

Quick run
```
python experiments/nmt/train_mbart50.py
python experiments/nmt/train_nllb600m.py
python experiments/nmt/train_mt5.py
python experiments/nmt/train_vit5.py
python experiments/nmt/train_bartpho.py
```

Choose any model
```
python experiments/nmt/train_generic.py   --csv data/bitext/tay_vi.csv   --out runs/custom   --model facebook/mbart-large-50-many-to-many-mmt   --src_code en_XX --tgt_code vi_VN
```
