"""
Phase A: 收集 15m K 線、predict 12 根 (=3h) horizon、每點 20 採樣，300 點。
       存原始路徑供後續多角度分析。
Phase B: 從原始資料做：
    - 採樣數比較 (5/10/20)
    - 交易成本敏感度 (0/0.04%/0.08%)
    - Walk-forward (前 60% vs 後 40%)
    - Regime 拆解 (上漲/盤整/下跌 by 200-bar SMA slope)
"""

import sys, time, argparse, pickle
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append("/home/anson/Kronos/repo")
from model import Kronos, KronosTokenizer, KronosPredictor

DATA = "/home/anson/quant-btc-gold/user_data/data/binance/BTC_USDT-15m.feather"
TOK = "/home/anson/Kronos/finetune_btc/outputs/models/btc_multi_tokenizer/checkpoints/best_model"
MDL = "/home/anson/Kronos/finetune_btc/outputs/models/btc_multi_predictor/checkpoints/best_model"
OUT = Path("/home/anson/Kronos/alpha_v3_multi")
OUT.mkdir(exist_ok=True)

LOOKBACK = 400
PRED_LEN = 12          # 12 × 15min = 3h
HORIZON_BARS = 12      # 評估點 = 3h
SMA_WIN = 200          # regime 判定窗口（200×15m = 50h）


def collect(n_points=300, samples=20, max_years=2):
    df = pd.read_feather(DATA)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)
    df = df.sort_values("date").reset_index(drop=True)
    # 真正 out-of-sample: 微調 test 區間 (2025-09-15 起)
    # 為了 lookback=400 (100h)，從更早一點開始載入
    test_start = pd.Timestamp("2025-09-15") - pd.Timedelta(hours=120)
    df = df[df["date"] >= test_start].reset_index(drop=True)

    # 預計算 SMA 與其斜率（用線性回歸 over SMA_WIN 根上的 log price）
    df["log_close"] = np.log(df["close"])
    df["sma"] = df["close"].rolling(SMA_WIN).mean()

    min_idx = LOOKBACK
    max_idx = len(df) - PRED_LEN - 1
    test_idx = np.linspace(min_idx, max_idx, n_points, dtype=int)

    tok = KronosTokenizer.from_pretrained(TOK)
    mdl = Kronos.from_pretrained(MDL)
    predictor = KronosPredictor(mdl, tok, device="cuda:0", max_context=512)

    paths = np.zeros((n_points, samples, PRED_LEN), dtype=np.float32)
    meta = []
    t0 = time.time()
    for k, i in enumerate(test_idx):
        x_df = df.iloc[i - LOOKBACK:i][["open", "high", "low", "close", "volume"]].reset_index(drop=True)
        x_ts = df.iloc[i - LOOKBACK:i]["date"].reset_index(drop=True)
        y_ts = df.iloc[i:i + PRED_LEN]["date"].reset_index(drop=True)
        cur_close = float(df.iloc[i - 1]["close"])
        actual_close = float(df.iloc[i - 1 + HORIZON_BARS]["close"])

        # regime: 用 i 之前 SMA_WIN 根 log price 線性回歸斜率
        seg = df["log_close"].iloc[i - SMA_WIN:i].values
        x = np.arange(SMA_WIN)
        slope = np.polyfit(x, seg, 1)[0]   # 每根 log return
        slope_annual = slope * 4 * 24 * 365  # 年化 log return ≈ trend strength

        for s in range(samples):
            pred_df = predictor.predict(
                df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
                pred_len=PRED_LEN, T=1.0, top_p=0.9, sample_count=1, verbose=False,
            )
            paths[k, s] = pred_df["close"].values

        meta.append({
            "idx": int(i),
            "time": df.iloc[i - 1]["date"],
            "cur_close": cur_close,
            "actual_close_h": actual_close,
            "actual_ret_h": (actual_close - cur_close) / cur_close,
            "slope_annual": slope_annual,
        })

        elapsed = time.time() - t0
        eta = elapsed / (k + 1) * (n_points - k - 1)
        if (k + 1) % 20 == 0 or k == 0:
            print(f"  {k+1}/{n_points} elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

    meta_df = pd.DataFrame(meta)
    with open(OUT / "raw.pkl", "wb") as f:
        pickle.dump({"paths": paths, "meta": meta_df,
                     "lookback": LOOKBACK, "pred_len": PRED_LEN,
                     "horizon_bars": HORIZON_BARS, "samples": samples}, f)
    print(f"saved → {OUT/'raw.pkl'} ({paths.nbytes/1e6:.1f} MB)")
    return paths, meta_df


def metrics(rets: np.ndarray, periods_per_year: float):
    if len(rets) == 0 or rets.std() == 0:
        return dict(n=len(rets), mean=float("nan"), std=float("nan"),
                    sharpe=float("nan"), cum_cmp=float("nan"), maxdd=float("nan"),
                    win_rate=float("nan"))
    equity = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return dict(
        n=len(rets), mean=float(rets.mean()), std=float(rets.std()),
        sharpe=float(np.sqrt(periods_per_year) * rets.mean() / rets.std()),
        cum_cmp=float(equity[-1] - 1), maxdd=float(dd.min()),
        win_rate=float((rets > 0).mean()),
    )


def analyze():
    with open(OUT / "raw.pkl", "rb") as f:
        d = pickle.load(f)
    paths = d["paths"]              # (N, S, P)
    meta = d["meta"]
    actual_ret = meta["actual_ret_h"].values
    cur_close = meta["cur_close"].values
    slope = meta["slope_annual"].values
    N, S_max, P = paths.shape
    # close at horizon for each (n, s)
    close_h = paths[:, :, HORIZON_BARS - 1]                # (N, S)
    pred_rets_full = (close_h - cur_close[:, None]) / cur_close[:, None]  # (N, S)

    # 15m K-line, horizon=3h → 每筆獨立回報 (periods_per_year for sharpe annualization)
    horizon_h = HORIZON_BARS * 0.25
    periods_per_year = 24 * 365 / horizon_h   # 2920

    # =========== 1. 採樣數比較 ===========
    print("\n=== 採樣數比較 (cost=0%) ===")
    rows = []
    for S in [5, 10, 20]:
        pr = pred_rets_full[:, :S]
        prob_up = (pr > 0).mean(axis=1)
        sig = np.where(prob_up > 0.5, 1, -1)
        rets = sig * actual_ret
        acc = ((prob_up > 0.5) == (actual_ret > 0)).mean()
        m = metrics(rets, periods_per_year)
        m.update(samples=S, acc=acc, ic=float(pd.Series(pr.mean(axis=1)).corr(pd.Series(actual_ret), method="spearman")))
        rows.append(m)
    print(pd.DataFrame(rows)[["samples", "n", "acc", "ic", "win_rate", "sharpe", "cum_cmp", "maxdd"]].to_string(
        index=False, formatters={"acc": "{:.3f}".format, "ic": "{:+.3f}".format,
                                  "win_rate": "{:.3f}".format, "sharpe": "{:+.2f}".format,
                                  "cum_cmp": "{:+.2%}".format, "maxdd": "{:.2%}".format}))

    # === 用 samples=20 作後續分析 ===
    pr = pred_rets_full
    prob_up = (pr > 0).mean(axis=1)

    # =========== 2. 交易成本敏感度 ===========
    print("\n=== 交易成本敏感度 (samples=20, 假設每筆獨立 round-trip) ===")
    rows = []
    for cost_bp in [0, 4, 8, 16]:  # 0 / 0.04% / 0.08% / 0.16%
        c = cost_bp / 10000
        sig = np.where(prob_up > 0.5, 1, -1)
        rets = sig * actual_ret - c  # 每筆一次 round-trip 成本
        m = metrics(rets, periods_per_year); m["cost_bp"] = cost_bp
        rows.append(m)
    print(pd.DataFrame(rows)[["cost_bp", "n", "win_rate", "sharpe", "cum_cmp", "maxdd"]].to_string(
        index=False, formatters={"win_rate": "{:.3f}".format, "sharpe": "{:+.2f}".format,
                                  "cum_cmp": "{:+.2%}".format, "maxdd": "{:.2%}".format}))

    # =========== 3. Walk-forward 穩定性 ===========
    print("\n=== Walk-forward (前 60% vs 後 40%, samples=20, cost=0.08%) ===")
    cut = int(N * 0.6)
    cost = 8 / 10000
    sig = np.where(prob_up > 0.5, 1, -1)
    rets = sig * actual_ret - cost
    rows = []
    for name, sl in [("前 60%", slice(0, cut)), ("後 40%", slice(cut, N)),
                      ("全部", slice(0, N))]:
        m = metrics(rets[sl], periods_per_year); m["seg"] = name
        m["acc"] = ((prob_up[sl] > 0.5) == (actual_ret[sl] > 0)).mean()
        rows.append(m)
    print(pd.DataFrame(rows)[["seg", "n", "acc", "win_rate", "sharpe", "cum_cmp", "maxdd"]].to_string(
        index=False, formatters={"acc": "{:.3f}".format, "win_rate": "{:.3f}".format,
                                  "sharpe": "{:+.2f}".format, "cum_cmp": "{:+.2%}".format,
                                  "maxdd": "{:.2%}".format}))

    # =========== 4. Regime 拆解 ===========
    print("\n=== Regime 拆解 by 200-bar (50h) log-price slope, 年化 ===")
    # 用 slope 分三檔: < -0.2 跌, -0.2~0.2 盤整, > 0.2 漲 (年化 log-return)
    regime = np.where(slope > 0.2, "UP",
              np.where(slope < -0.2, "DOWN", "SIDE"))
    rows = []
    for r in ["UP", "SIDE", "DOWN", "ALL"]:
        mask = np.ones(N, dtype=bool) if r == "ALL" else (regime == r)
        if mask.sum() == 0:
            continue
        m = metrics(rets[mask], periods_per_year)
        m["regime"] = r
        m["acc"] = ((prob_up[mask] > 0.5) == (actual_ret[mask] > 0)).mean()
        m["long_share"] = float((prob_up[mask] > 0.5).mean())
        rows.append(m)
    print(pd.DataFrame(rows)[["regime", "n", "acc", "long_share", "win_rate", "sharpe", "cum_cmp", "maxdd"]].to_string(
        index=False, formatters={"acc": "{:.3f}".format, "long_share": "{:.2f}".format,
                                  "win_rate": "{:.3f}".format, "sharpe": "{:+.2f}".format,
                                  "cum_cmp": "{:+.2%}".format, "maxdd": "{:.2%}".format}))

    # =========== 5. 高信心 gate 在各 regime ===========
    print("\n=== |prob-0.5|>0.2 高信心 gate × Regime (cost=0.08%) ===")
    hi_mask = (prob_up > 0.7) | (prob_up < 0.3)
    rows = []
    for r in ["UP", "SIDE", "DOWN", "ALL"]:
        rmask = np.ones(N, dtype=bool) if r == "ALL" else (regime == r)
        m_all = rmask & hi_mask
        if m_all.sum() == 0:
            rows.append({"regime": r, "n_hi": 0}); continue
        m = metrics(rets[m_all], periods_per_year)
        m["regime"] = r; m["n_hi"] = int(m_all.sum())
        m["acc_hi"] = ((prob_up[m_all] > 0.5) == (actual_ret[m_all] > 0)).mean()
        rows.append(m)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["collect", "analyze", "both"], default="both")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--samples", type=int, default=20)
    args = ap.parse_args()

    if args.phase in ("collect", "both"):
        collect(n_points=args.n, samples=args.samples)
    if args.phase in ("analyze", "both"):
        analyze()
