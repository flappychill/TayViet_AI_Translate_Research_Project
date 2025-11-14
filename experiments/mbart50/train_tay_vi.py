import os, random, gc, json, math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_scheduler, DataCollatorForSeq2Seq, GenerationConfig
import sacrebleu
import evaluate

def parse_args():
    import argparse
    a = argparse.ArgumentParser()
    a.add_argument("--csv", type=str, required=True)
    a.add_argument("--out", type=str, required=True)
    a.add_argument("--epochs", type=int, default=6)
    a.add_argument("--lr", type=float, default=2e-5)
    a.add_argument("--warmup", type=float, default=0.1)
    a.add_argument("--grad_accum", type=int, default=8)
    a.add_argument("--train_bs", type=int, default=1)
    a.add_argument("--eval_bs", type=int, default=2)
    a.add_argument("--max_src", type=int, default=128)
    a.add_argument("--max_new", type=int, default=64)
    a.add_argument("--patience", type=int, default=3)
    a.add_argument("--model", type=str, default="facebook/mbart-large-50-many-to-many-mmt")
    a.add_argument("--src_code", type=str, default="en_XX")
    a.add_argument("--tgt_code", type=str, default="vi_VN")
    return a.parse_args()

args = parse_args()
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("medium")

try:
    meteor_metric = evaluate.load("meteor")
except Exception:
    meteor_metric = None

def load_pairs(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8")
    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
    cols = set(df.columns)
    for a,b in [("tay","vietnamese"),("tay","viet"),("tày","việt"),("source","target"),("src","tgt")]:
        if {a,b}.issubset(cols):
            df = df[[a,b]].copy()
            df.columns = ["src","tgt"]
            break
    df["src"] = df["src"].astype(str).str.strip().str.strip('"').str.strip("'")
    df["tgt"] = df["tgt"].astype(str).str.strip().str.strip('"').str.strip("'")
    df = df[(df["src"]!="")&(df["tgt"]!="")].drop_duplicates().reset_index(drop=True)
    return df

df_all = load_pairs(args.csv)
train_df, test_df = train_test_split(df_all, test_size=0.05, random_state=SEED)
train_df, val_df  = train_test_split(train_df, test_size=0.10, random_state=SEED)

class PairDS(Dataset):
    def __init__(self, df):
        self.s = df["src"].tolist()
        self.t = df["tgt"].tolist()
    def __len__(self):
        return len(self.s)
    def __getitem__(self, i):
        return {"src": self.s[i], "tgt": self.t[i]}

tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
model     = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(DEVICE)
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()

gen_cfg = GenerationConfig.from_model_config(model.config)
gen_cfg.max_new_tokens = args.max_new
gen_cfg.early_stopping = False
gen_cfg.num_beams = 1
gen_cfg.no_repeat_ngram_size = 3
if getattr(model.config, "pad_token_id", None) is None:
    model.config.pad_token_id = tokenizer.pad_token_id

forced_bos = None
if hasattr(tokenizer, "lang_code_to_id") and args.tgt_code in tokenizer.lang_code_to_id:
    forced_bos = tokenizer.lang_code_to_id[args.tgt_code]

def set_langs(src_code, tgt_code):
    if hasattr(tokenizer, "src_lang"):
        tokenizer.src_lang = src_code
    if hasattr(tokenizer, "tgt_lang"):
        tokenizer.tgt_lang = tgt_code

collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest", label_pad_token_id=-100)
def collate(batch):
    set_langs(args.src_code, args.tgt_code)
    srcs = [b["src"] for b in batch]
    tgts = [b["tgt"] for b in batch]
    enc  = tokenizer(srcs, truncation=True, max_length=args.max_src, padding=False, text_target=tgts)
    return collator([{k:v[i] for k,v in enc.items()} for i in range(len(srcs))])

def build_loaders():
    tr, va, te = PairDS(train_df), PairDS(val_df), PairDS(test_df)
    def ld(ds, bs, sh):
        return DataLoader(ds, batch_size=bs, shuffle=sh, collate_fn=collate, pin_memory=True, num_workers=2, persistent_workers=True)
    return ld(tr,args.train_bs,True), ld(va,args.eval_bs,False), ld(te,args.eval_bs,False)

train_loader, val_loader, test_loader = build_loaders()
amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available() and amp_dtype==torch.float16)

if forced_bos is not None:
    model.config.forced_bos_token_id = forced_bos
    model.config.decoder_start_token_id = forced_bos

def generate_with_mode(input_ids, attn_mask, mode):
    kwargs = dict(attention_mask=attn_mask, max_new_tokens=args.max_new, no_repeat_ngram_size=3, forced_bos_token_id=forced_bos)
    if mode=="beam":
        return model.generate(input_ids, generation_config=gen_cfg, num_beams=4, early_stopping=True, **kwargs)
    if mode=="greedy":
        return model.generate(input_ids, generation_config=gen_cfg, do_sample=False, num_beams=1, **kwargs)
    if mode=="topk":
        return model.generate(input_ids, generation_config=gen_cfg, do_sample=True, top_k=50, temperature=0.7, num_beams=1, **kwargs)
    if mode=="topp":
        return model.generate(input_ids, generation_config=gen_cfg, do_sample=True, top_p=0.9, temperature=0.7, num_beams=1, **kwargs)
    return None

@torch.no_grad()
def eval_epoch(loader, mode):
    model.eval()
    preds, refs = [], []
    for batch in tqdm(loader, desc=f"eval-{mode}"):
        ids, msk = batch["input_ids"].to(DEVICE,non_blocking=True), batch["attention_mask"].to(DEVICE,non_blocking=True)
        gen = generate_with_mode(ids, msk, mode)
        preds.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
        labels = batch["labels"].masked_fill(batch["labels"]==-100, tokenizer.pad_token_id)
        refs.extend(tokenizer.batch_decode(labels, skip_special_tokens=True))
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    met = None
    if meteor_metric is not None:
        try:
            met = float(meteor_metric.compute(predictions=preds, references=refs)["meteor"])
        except Exception:
            met = None
    return bleu, met

def train_epoch(loader, optimizer, scheduler):
    model.train()
    total = 0.0
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(tqdm(loader, desc="train")):
        try:
            ids, msk, lab = batch["input_ids"].to(DEVICE,non_blocking=True), batch["attention_mask"].to(DEVICE,non_blocking=True), batch["labels"].to(DEVICE,non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
                out = model(input_ids=ids, attention_mask=msk, labels=lab)
                logits = out.logits.view(-1, out.logits.size(-1))
                loss = nn.functional.cross_entropy(logits, lab.view(-1), ignore_index=-100, label_smoothing=0.1) / args.grad_accum
            scaler.scale(loss).backward()
            if (step+1)%args.grad_accum==0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach().item())
        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache(); torch.cuda.ipc_collect()
    return total/max(1,len(loader))

steps = max(1, math.ceil(len(train_loader)*args.epochs/args.grad_accum))
optimizer = optim.AdamW(model.parameters(), lr=args.lr)
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=int(args.warmup*steps), num_training_steps=steps)

best_bleu, patience = -1.0, 0
os.makedirs(args.out, exist_ok=True)

for ep in range(args.epochs):
    tr = train_epoch(train_loader, optimizer, scheduler)
    vb, vm = eval_epoch(val_loader, "beam")
    if vb > best_bleu:
        best_bleu, patience = vb, 0
        model.save_pretrained(args.out); tokenizer.save_pretrained(args.out)
    else:
        patience += 1
        if patience >= args.patience:
            break
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.ipc_collect()

model = AutoModelForSeq2SeqLM.from_pretrained(args.out).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(args.out, use_fast=True)
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()
if forced_bos is not None:
    model.config.forced_bos_token_id = forced_bos
    model.config.decoder_start_token_id = forced_bos

for m in ["beam","greedy","topk","topp"]:
    b,_=eval_epoch(test_loader,m)
    print(m, round(b,2))
