import pandas as pd, json, os, sys
src = sys.argv[1] if len(sys.argv)>1 else "translate/flow/data/final.csv"
out = "reports"
os.makedirs(out, exist_ok=True)
df = pd.read_csv(src)
df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
cols = set(df.columns)
for a,b in [("tay","vietnamese"),("tày","việt"),("src","tgt"),("source","target")]:
    if {a,b}.issubset(cols):
        df = df[[a,b]].copy()
        df.columns = ["src","tgt"]
        break
df = df.dropna().drop_duplicates()
df["src_len"]=df["src"].astype(str).str.len()
df["tgt_len"]=df["tgt"].astype(str).str.len()
stats = {
 "rows": int(len(df)),
 "src_chars_mean": float(df["src_len"].mean()),
 "tgt_chars_mean": float(df["tgt_len"].mean()),
 "src_chars_p95": float(df["src_len"].quantile(0.95)),
 "tgt_chars_p95": float(df["tgt_len"].quantile(0.95))
}
with open(os.path.join(out,"stats.json"),"w",encoding="utf-8") as f:
    json.dump(stats,f,ensure_ascii=False,indent=2)
df.sample(min(200,len(df))).to_csv(os.path.join(out,"preview.csv"),index=False)
print(json.dumps(stats,ensure_ascii=False))
