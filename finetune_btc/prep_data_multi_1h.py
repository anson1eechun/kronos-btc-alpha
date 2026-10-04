"""
多資產 1h pickle (for Kronos fine-tune)
- BTC_USDT (Binance, 6 年)
- GC=F / ES=F / NQ=F / SI=F (yfinance 1h, ~2.4 年)
- 切 train / val / test 三段
"""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

OUT = Path("/home/anson/Kronos/finetune_btc/data_multi")
OUT.mkdir(parents=True, exist_ok=True)

TRAIN = ("2020-01-01", "2024-12-31")
VAL   = ("2024-09-01", "2025-09-30")
TEST  = ("2025-09-15", "2026-05-09")


def df_to_kronos(df, name):
    """確保有 open/high/low/close/vol/amt + datetime index name='datetime', 無 NaN."""
    df = df.copy()
    df["vol"] = df["volume"]
    df["amt"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4 * df["vol"]
    df = df[["open", "high", "low", "close", "vol", "amt"]].dropna()
    df = df[(df[["open","high","low","close"]] > 0).all(axis=1)]
    df = df[df["vol"] > 0]
    df.index.name = "datetime"      # dataset.py 依賴此欄名
    print(f"  {name}: {len(df)} 根 ({df.index[0]} ~ {df.index[-1]})")
    return df


def load_btc():
    src = "/home/anson/quant-btc-gold/user_data/data/binance/BTC_USDT-1h.feather"
    d = pd.read_feather(src)
    d["datetime"] = pd.to_datetime(d["date"], utc=True).dt.tz_convert(None)
    d = d.set_index("datetime").sort_index()
    d = d[["open", "high", "low", "close", "volume"]]
    return df_to_kronos(d, "BTC_USDT")


def load_yf(sym):
    t = yf.Ticker(sym).history(period="730d", interval="1h", auto_adjust=False)
    if t.empty:
        raise RuntimeError(f"no data for {sym}")
    t.index = pd.to_datetime(t.index, utc=True).tz_convert(None)
    t = t.rename(columns={"Open": "open", "High": "high", "Low": "low",
                          "Close": "close", "Volume": "volume"})
    t = t[["open", "high", "low", "close", "volume"]]
    return df_to_kronos(t, sym)


print("===== 載入資料 =====")
data = {
    "BTC_USDT": load_btc(),
    "GC_F":     load_yf("GC=F"),
    "ES_F":     load_yf("ES=F"),
    "NQ_F":     load_yf("NQ=F"),
    "SI_F":     load_yf("SI=F"),
}


def slice_save(name, start, end):
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    out = {sym: d.loc[s:e] for sym, d in data.items()}
    out = {sym: d for sym, d in out.items() if len(d) > 280}   # 需 > lookback+predict
    with open(OUT / f"{name}_data.pkl", "wb") as f:
        pickle.dump(out, f)
    print(f"  [{name}] 標的數 {len(out)}")
    for sym, d in out.items():
        print(f"    {sym}: {len(d)} 根")


print("\n===== 切分 train / val / test =====")
slice_save("train", *TRAIN)
slice_save("val", *VAL)
slice_save("test", *TEST)
print(f"\n→ {OUT}")
