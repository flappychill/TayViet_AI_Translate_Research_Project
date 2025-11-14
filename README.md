# Tay ↔ Vietnamese Translation

Sản phẩm dịch hai chiều Tày và Việt. Gồm mã huấn luyện, suy luận hàng loạt, API Flask và giao diện web.

## Tính năng
- Dịch Tày→Việt và Việt→Tày
- API JSON và web UI
- Huấn luyện seq2seq theo mã gốc
- EDA nhanh cho dữ liệu
- Docker và CI

## Cấu trúc
```
api/                 Flask app + UI
config/model.toml    Tên model và cấu hình
translate/           Mã dịch/huấn luyện đã chuyển sang Tày
data/original/       CSV mẫu src,tgt
scripts/             EDA, tiện ích
experiments/         Mã train mBART50 theo pipeline gốc
```

## Cài đặt
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Chạy web
```
gunicorn -w 2 -b 0.0.0.0:8000 api.app:app
```
Mở `http://localhost:8000`.

## API
`POST /api/translate`
```json
{ "text": "…", "direction": "tay2vi" }
```

## Huấn luyện theo pipeline gốc
```
python experiments/mbart50/train_tay_vi.py --csv data/original/tay_vi_train.csv --out runs/mbart50 --epochs 1
```

## Weights
Đưa checkpoint vào phần **Releases** của GitHub và gắn liên kết tại đây.

## NMT Experiments
Các mô hình có sẵn
- MBART50
- NLLB-200 distilled 600M
- mT5 small
- viT5 base
- BARTPho syllable

Chạy nhanh
```
python experiments/nmt/train_mbart50.py
python experiments/nmt/train_nllb600m.py
python experiments/nmt/train_mt5.py
python experiments/nmt/train_vit5.py
python experiments/nmt/train_bartpho.py
```

Tự chọn model bất kỳ
```
python experiments/nmt/train_generic.py   --csv data/bitext/tay_vi.csv   --out runs/custom   --model facebook/mbart-large-50-many-to-many-mmt   --src_code en_XX --tgt_code vi_VN
```
