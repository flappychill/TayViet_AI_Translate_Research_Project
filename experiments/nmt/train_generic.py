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

def args():
    import argparse
    a=argparse.ArgumentParser()
    a.add_argument("--csv",type=str,required=True)
    a.add_argument("--out",type=str,required=True)
    a.add_argument("--model",type=str,required=True)
    a.add_argument("--src_code",type=str,default="en_XX")
    a.add_argument("--tgt_code",type=str,default="vi_VN")
    a.add_argument("--epochs",type=int,default=6)
    a.add_argument("--lr",type=float,default=2e-5)
    a.add_argument("--warmup",type=float,default=0.1)
    a.add_argument("--grad_accum",type=int,default=8)
    a.add_argument("--train_bs",type=int,default=1)
    a.add_argument("--eval_bs",type=int,default=2)
    a.add_argument("--max_src",type=int,default=128)
    a.add_argument("--max_new",type=int,default=64)
    a.add_argument("--patience",type=int,default=3)
    return a.parse_args()

cfg=args()
SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("medium")
try:
    meteor=evaluate.load("meteor")
except Exception:
    meteor=None

def load_pairs(p):
    d=pd.read_csv(p,encoding="utf-8")
    d=d.rename(columns={c:str(c).strip().lower() for c in d.columns})
    cols=set(d.columns)
    for a,b in [("tay","vietnamese"),("tay","viet"),("tày","việt"),("src","tgt"),("source","target")]:
        if {a,b}.issubset(cols):
            d=d[[a,b]].copy();d.columns=["src","tgt"];break
    d["src"]=d["src"].astype(str).str.strip()
    d["tgt"]=d["tgt"].astype(str).str.strip()
    d=d[(d["src"]!="")&(d["tgt"]!="")].drop_duplicates().reset_index(drop=True)
    return d

all_df=load_pairs(cfg.csv)
tr_df,te_df=train_test_split(all_df,test_size=0.05,random_state=SEED)
tr_df,va_df=train_test_split(tr_df,test_size=0.10,random_state=SEED)

class DS(Dataset):
    def __init__(self,df):
        self.s=df["src"].tolist();self.t=df["tgt"].tolist()
    def __len__(self):
        return len(self.s)
    def __getitem__(self,i):
        return {"src":self.s[i],"tgt":self.t[i]}

tok=AutoTokenizer.from_pretrained(cfg.model,use_fast=True)
mdl=AutoModelForSeq2SeqLM.from_pretrained(cfg.model).to(DEVICE)
if hasattr(mdl,"gradient_checkpointing_enable"):
    mdl.gradient_checkpointing_enable()
gen=GenerationConfig.from_model_config(mdl.config)
gen.max_new_tokens=cfg.max_new
gen.early_stopping=False
gen.num_beams=1
gen.no_repeat_ngram_size=3
if getattr(mdl.config,"pad_token_id",None) is None:
    mdl.config.pad_token_id=tok.pad_token_id

forced=None
if hasattr(tok,"lang_code_to_id") and cfg.tgt_code in tok.lang_code_to_id:
    forced=tok.lang_code_to_id[cfg.tgt_code]

def set_lang(a,b):
    if hasattr(tok,"src_lang"):
        tok.src_lang=a
    if hasattr(tok,"tgt_lang"):
        tok.tgt_lang=b

col=DataCollatorForSeq2Seq(tok,model=mdl,padding="longest",label_pad_token_id=-100)
def pack(batch):
    set_lang(cfg.src_code,cfg.tgt_code)
    s=[b["src"] for b in batch]
    t=[b["tgt"] for b in batch]
    enc=tok(s,truncation=True,max_length=cfg.max_src,padding=False,text_target=t)
    return col([{k:v[i] for k,v in enc.items()} for i in range(len(s))])

def mk(ds,bs,sh):
    return DataLoader(ds,batch_size=bs,shuffle=sh,collate_fn=pack,pin_memory=True,num_workers=2,persistent_workers=True)

tr,va,te=DS(tr_df),DS(va_df),DS(te_df)
tr_loader=mk(tr,cfg.train_bs,True)
va_loader=mk(va,cfg.eval_bs,False)
te_loader=mk(te,cfg.eval_bs,False)

amp=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
scaler=torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available() and amp==torch.float16)

if forced is not None:
    mdl.config.forced_bos_token_id=forced
    mdl.config.decoder_start_token_id=forced

def gen_mode(x,a,m):
    kw=dict(attention_mask=a,max_new_tokens=cfg.max_new,no_repeat_ngram_size=3,forced_bos_token_id=forced)
    if m=="beam":
        return mdl.generate(x,generation_config=gen,num_beams=4,early_stopping=True,**kw)
    if m=="greedy":
        return mdl.generate(x,generation_config=gen,do_sample=False,num_beams=1,**kw)
    if m=="topk":
        return mdl.generate(x,generation_config=gen,do_sample=True,top_k=50,temperature=0.7,num_beams=1,**kw)
    if m=="topp":
        return mdl.generate(x,generation_config=gen,do_sample=True,top_p=0.9,temperature=0.7,num_beams=1,**kw)
    return None

@torch.no_grad()
def eval_epoch(loader,mode):
    mdl.eval();preds=[];refs=[]
    for b in tqdm(loader,desc=f"eval-{mode}"):
        ids=b["input_ids"].to(DEVICE,non_blocking=True)
        msk=b["attention_mask"].to(DEVICE,non_blocking=True)
        out=gen_mode(ids,msk,mode)
        preds.extend(tok.batch_decode(out,skip_special_tokens=True))
        lab=b["labels"].masked_fill(b["labels"]==-100,tok.pad_token_id)
        refs.extend(tok.batch_decode(lab,skip_special_tokens=True))
    bleu=sacrebleu.corpus_bleu(preds,[refs]).score
    met=None
    if meteor is not None:
        try:
            met=float(meteor.compute(predictions=preds,references=refs)["meteor"])
        except Exception:
            met=None
    return bleu,met

def train_epoch(loader,opt,sch):
    mdl.train();total=0.0
    opt.zero_grad(set_to_none=True)
    for i,b in enumerate(tqdm(loader,desc="train")):
        try:
            ids=b["input_ids"].to(DEVICE,non_blocking=True)
            msk=b["attention_mask"].to(DEVICE,non_blocking=True)
            lab=b["labels"].to(DEVICE,non_blocking=True)
            with torch.autocast(device_type="cuda",dtype=amp,enabled=torch.cuda.is_available()):
                o=mdl(input_ids=ids,attention_mask=msk,labels=lab)
                z=o.logits.view(-1,o.logits.size(-1))
                loss=nn.functional.cross_entropy(z,lab.view(-1),ignore_index=-100,label_smoothing=0.1)/cfg.grad_accum
            scaler.scale(loss).backward()
            if (i+1)%cfg.grad_accum==0:
                torch.nn.utils.clip_grad_norm_(mdl.parameters(),1.0)
                scaler.step(opt);scaler.update();sch.step()
                opt.zero_grad(set_to_none=True)
            total+=float(loss.detach().item())
        except torch.cuda.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache(); torch.cuda.ipc_collect()
    return total/max(1,len(loader))

steps=max(1,math.ceil(len(tr_loader)*cfg.epochs/cfg.grad_accum))
opt=optim.AdamW(mdl.parameters(),lr=cfg.lr)
sch=get_scheduler("linear",optimizer=opt,num_warmup_steps=int(cfg.warmup*steps),num_training_steps=steps)

best=-1.0;pat=0
os.makedirs(cfg.out,exist_ok=True)
for e in range(cfg.epochs):
    trl=train_epoch(tr_loader,opt,sch)
    vb,vm=eval_epoch(va_loader,"beam")
    if vb>best:
        best=vb;pat=0
        mdl.save_pretrained(cfg.out); tok.save_pretrained(cfg.out)
    else:
        pat+=1
        if pat>=cfg.patience:
            break
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.ipc_collect()

mdl=AutoModelForSeq2SeqLM.from_pretrained(cfg.out).to(DEVICE)
tok=AutoTokenizer.from_pretrained(cfg.out,use_fast=True)
if hasattr(mdl,"gradient_checkpointing_enable"):
    mdl.gradient_checkpointing_enable()
if forced is not None:
    mdl.config.forced_bos_token_id=forced
    mdl.config.decoder_start_token_id=forced

for m in ["beam","greedy","topk","topp"]:
    b,_=eval_epoch(te_loader,m)
    print(m,round(b,2))
