"""
把 BTC 15m feather → repo 預期格式: {symbol: pd.DataFrame[open,high,low,close,vol,amt]}
存到 train/val/test 三個 pickle。
"""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

SRC = "/home/anson/quant-btc-gold/user_data/data/binance/BTC_USDT-15m.feather"
OUT = Path("/home/anson/Kronos/finetune_btc/data")
OUT.mkdir(parents=True, exist_ok=True)

# 時間切分
TRAIN = ("2020-01-01", "2024-12-31")
VAL   = ("2024-09-01", "2025-09-30")  # 與 train 重疊以保留 lookback
TEST  = ("2025-09-15", "2026-05-09")  # 微調前未訓練過

df = pd.read_feather(SRC)
df["datetime"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)
df = df.set_index("datetime").sort_index()

# 加 vol / amt 欄位
df["vol"] = df["volume"]
df["amt"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4 * df["vol"]
df = df[["open", "high", "low", "close", "vol", "amt"]].dropna()

print(f"全資料 {len(df)} 根, {df.index[0]} ~ {df.index[-1]}")


def slice_save(name, start, end):
    s = pd.Timestamp(start); e = pd.Timestamp(end)
    sub = df.loc[s:e]
    out = {"BTC_USDT": sub}
    with open(OUT / f"{name}_data.pkl", "wb") as f:
        pickle.dump(out, f)
    print(f"  {name}: {len(sub)} 根 ({sub.index[0]} ~ {sub.index[-1]})")


slice_save("train", *TRAIN)
slice_save("val", *VAL)
slice_save("test", *TEST)
print(f"→ {OUT}")
