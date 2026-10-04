"""
多標的訓練資料準備: BTC + ETH + SOL + BNB + XRP + ADA (15m)
- 訓練時段每標的各自切片 → 上幣前的時段自動為空 (SOL pre-2020-08)
- 存到 finetune_btc/data_multi/{train,val,test}_data.pkl
- test 集只保留 BTC (純 BTC 評估，不污染)
"""
import pickle
from pathlib import Path
import pandas as pd

SYMBOLS = ["BTC_USDT", "ETH_USDT", "SOL_USDT", "BNB_USDT", "XRP_USDT", "ADA_USDT"]
SRC_DIR = Path("/home/anson/quant-btc-gold/user_data/data/binance")
OUT = Path("/home/anson/Kronos/finetune_btc/data_multi")
OUT.mkdir(parents=True, exist_ok=True)

TRAIN = ("2020-01-01", "2024-12-31")
VAL   = ("2024-09-01", "2025-09-30")
TEST  = ("2025-09-15", "2026-05-09")   # 只放 BTC


def load(sym: str) -> pd.DataFrame:
    df = pd.read_feather(SRC_DIR / f"{sym}-15m.feather")
    df["datetime"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)
    df = df.set_index("datetime").sort_index()
    df["vol"] = df["volume"]
    df["amt"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4 * df["vol"]
    return df[["open", "high", "low", "close", "vol", "amt"]].dropna()


loaded = {s: load(s) for s in SYMBOLS}
for s, df in loaded.items():
    print(f"  {s}: {len(df):>7} rows  {df.index[0]} → {df.index[-1]}")


def slice_set(name, start, end, only_btc=False):
    s = pd.Timestamp(start); e = pd.Timestamp(end)
    out = {}
    for sym, df in loaded.items():
        if only_btc and sym != "BTC_USDT":
            continue
        sub = df.loc[s:e]
        if len(sub) >= 269:   # lookback 256 + predict 12 + 1
            out[sym] = sub
    with open(OUT / f"{name}_data.pkl", "wb") as f:
        pickle.dump(out, f)
    print(f"\n[{name}] symbols={list(out.keys())}")
    for sym, sub in out.items():
        print(f"  {sym}: {len(sub):>7} rows ({sub.index[0]} → {sub.index[-1]})")


slice_set("train", *TRAIN)
slice_set("val",   *VAL)
slice_set("test",  *TEST, only_btc=True)
print(f"\n→ {OUT}")
