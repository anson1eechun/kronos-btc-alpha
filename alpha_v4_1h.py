"""
1h timeframe alpha validation (多資產微調後評估)
- 預測 BTC 1h, horizon = 12 根 (= 12h)
- 300 點 × 20 採樣, OOS test 區間 2025-09-15+
- 同時支援 baseline / finetuned 模型 (用 --model 切換)
"""
import sys, time, argparse, pickle
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append("/home/anson/Kronos/repo")
from model import Kronos, KronosTokenizer, KronosPredictor

DATA = "/home/anson/quant-btc-gold/user_data/data/binance/BTC_USDT-1h.feather"

MODELS = {
    "base": (
        "/home/anson/Kronos/models/Kronos-Tokenizer-base",
        "/home/anson/Kronos/models/Kronos-base",
    ),
    "ft_multi": (
        "/home/anson/Kronos/finetune_btc/outputs/models/btc_multi_tokenizer/checkpoints/best_model",
        "/home/anson/Kronos/finetune_btc/outputs/models/btc_multi_predictor/checkpoints/best_model",
    ),
}

LOOKBACK = 256
PRED_LEN = 12
HORIZON_BARS = 12
SMA_WIN = 200


def collect(model_key, n_points=300, samples=20, test_start="2025-09-15"):
    tok_path, mdl_path = MODELS[model_key]
    out = Path(f"/home/anson/Kronos/alpha_v4_{model_key}")
    out.mkdir(exist_ok=True)

    df = pd.read_feather(DATA)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)
    df = df.sort_values("date").reset_index(drop=True)
    # 從 test_start - lookback 開始載入
    start = pd.Timestamp(test_start) - pd.Timedelta(hours=LOOKBACK + 5)
    df = df[df["date"] >= start].reset_index(drop=True)

    df["log_close"] = np.log(df["close"])

    min_idx = LOOKBACK
    max_idx = len(df) - PRED_LEN - 1
    test_idx = np.linspace(min_idx, max_idx, n_points, dtype=int)

    tok = KronosTokenizer.from_pretrained(tok_path)
    mdl = Kronos.from_pretrained(mdl_path)
    predictor = KronosPredictor(mdl, tok, device="cuda:0", max_context=512)

    paths = np.zeros((n_points, samples, PRED_LEN), dtype=np.float32)
    meta = []
    t0 = time.time()
    for k, i in enumerate(test_idx):
        x_df = df.iloc[i - LOOKBACK:i][["open","high","low","close","volume"]].reset_index(drop=True)
        x_ts = df.iloc[i - LOOKBACK:i]["date"].reset_index(drop=True)
        y_ts = df.iloc[i:i + PRED_LEN]["date"].reset_index(drop=True)
        cur_close = float(df.iloc[i - 1]["close"])
        actual_close = float(df.iloc[i - 1 + HORIZON_BARS]["close"])

        seg = df["log_close"].iloc[i - SMA_WIN:i].values
        x = np.arange(SMA_WIN)
        slope = np.polyfit(x, seg, 1)[0]
        slope_annual = slope * 24 * 365

        for s in range(samples):
            pred_df = predictor.predict(
                df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
                pred_len=PRED_LEN, T=1.0, top_p=0.9, sample_count=1, verbose=False,
            )
            paths[k, s] = pred_df["close"].values

        meta.append({
            "idx": int(i), "time": df.iloc[i - 1]["date"],
            "cur_close": cur_close, "actual_close_h": actual_close,
            "actual_ret_h": (actual_close - cur_close) / cur_close,
            "slope_annual": slope_annual,
        })

        elapsed = time.time() - t0
        eta = elapsed / (k + 1) * (n_points - k - 1)
        if (k + 1) % 20 == 0 or k == 0:
            print(f"  [{model_key}] {k+1}/{n_points} elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

    meta_df = pd.DataFrame(meta)
    with open(out / "raw.pkl", "wb") as f:
        pickle.dump({"paths": paths, "meta": meta_df,
                     "lookback": LOOKBACK, "pred_len": PRED_LEN,
                     "horizon_bars": HORIZON_BARS, "samples": samples}, f)
    print(f"saved → {out/'raw.pkl'}")
    return paths, meta_df, out


def metrics(rets, periods_per_year):
    if len(rets) == 0 or rets.std() == 0:
        return dict(n=len(rets), mean=float("nan"), sharpe=float("nan"),
                    cum_cmp=float("nan"), maxdd=float("nan"), win_rate=float("nan"))
    equity = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return dict(n=len(rets), mean=float(rets.mean()),
                sharpe=float(np.sqrt(periods_per_year) * rets.mean() / rets.std()),
                cum_cmp=float(equity[-1] - 1), maxdd=float(dd.min()),
                win_rate=float((rets > 0).mean()))


def analyze(model_key):
    out = Path(f"/home/anson/Kronos/alpha_v4_{model_key}")
    with open(out / "raw.pkl", "rb") as f:
        d = pickle.load(f)
    paths = d["paths"]
    meta = d["meta"]
    actual = meta["actual_ret_h"].values
    cur_close = meta["cur_close"].values
    slope = meta["slope_annual"].values
    N, S, P = paths.shape
    close_h = paths[:, :, HORIZON_BARS - 1]
    pred_rets = (close_h - cur_close[:, None]) / cur_close[:, None]
    periods_per_year = 24 * 365 / HORIZON_BARS

    print(f"\n========== MODEL: {model_key} ==========")
    print(f"N={N}, samples={S}, horizon={HORIZON_BARS}h")

    print("\n=== 採樣數比較 (cost=0%) ===")
    rows = []
    for S_use in [5, 10, 20]:
        pr = pred_rets[:, :S_use]
        prob = (pr > 0).mean(axis=1)
        sig = np.where(prob > 0.5, 1, -1)
        rets = sig * actual
        acc = ((prob > 0.5) == (actual > 0)).mean()
        m = metrics(rets, periods_per_year)
        m.update(samples=S_use, acc=acc,
                 ic=float(pd.Series(pr.mean(axis=1)).corr(pd.Series(actual), method="spearman")))
        rows.append(m)
    print(pd.DataFrame(rows)[["samples","n","acc","ic","win_rate","sharpe","cum_cmp","maxdd"]].to_string(
        index=False, formatters={"acc":"{:.3f}".format,"ic":"{:+.3f}".format,
            "win_rate":"{:.3f}".format,"sharpe":"{:+.2f}".format,
            "cum_cmp":"{:+.2%}".format,"maxdd":"{:.2%}".format}))

    pr = pred_rets
    prob = (pr > 0).mean(axis=1)

    print("\n=== 交易成本敏感度 ===")
    rows = []
    for cost_bp in [0, 4, 8, 16]:
        c = cost_bp / 10000
        sig = np.where(prob > 0.5, 1, -1)
        rets = sig * actual - c
        m = metrics(rets, periods_per_year); m["cost_bp"] = cost_bp
        rows.append(m)
    print(pd.DataFrame(rows)[["cost_bp","n","win_rate","sharpe","cum_cmp","maxdd"]].to_string(
        index=False, formatters={"win_rate":"{:.3f}".format,"sharpe":"{:+.2f}".format,
            "cum_cmp":"{:+.2%}".format,"maxdd":"{:.2%}".format}))

    print("\n=== Walk-forward (cost=0.08%) ===")
    cut = int(N * 0.6); cost = 8 / 10000
    sig = np.where(prob > 0.5, 1, -1); rets = sig * actual - cost
    rows = []
    for name, sl in [("前 60%", slice(0, cut)), ("後 40%", slice(cut, N)), ("全部", slice(0, N))]:
        m = metrics(rets[sl], periods_per_year); m["seg"] = name
        m["acc"] = ((prob[sl] > 0.5) == (actual[sl] > 0)).mean()
        rows.append(m)
    print(pd.DataFrame(rows)[["seg","n","acc","win_rate","sharpe","cum_cmp","maxdd"]].to_string(
        index=False, formatters={"acc":"{:.3f}".format,"win_rate":"{:.3f}".format,
            "sharpe":"{:+.2f}".format,"cum_cmp":"{:+.2%}".format,"maxdd":"{:.2%}".format}))

    print("\n=== Regime 拆解 (200h slope) ===")
    regime = np.where(slope > 0.2, "UP", np.where(slope < -0.2, "DOWN", "SIDE"))
    rows = []
    for r in ["UP", "SIDE", "DOWN", "ALL"]:
        mask = np.ones(N, bool) if r == "ALL" else (regime == r)
        if mask.sum() == 0: continue
        m = metrics(rets[mask], periods_per_year); m["regime"] = r
        m["acc"] = ((prob[mask] > 0.5) == (actual[mask] > 0)).mean()
        m["long_share"] = float((prob[mask] > 0.5).mean())
        rows.append(m)
    print(pd.DataFrame(rows)[["regime","n","acc","long_share","win_rate","sharpe","cum_cmp","maxdd"]].to_string(
        index=False, formatters={"acc":"{:.3f}".format,"long_share":"{:.2f}".format,
            "win_rate":"{:.3f}".format,"sharpe":"{:+.2f}".format,
            "cum_cmp":"{:+.2%}".format,"maxdd":"{:.2%}".format}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS.keys()), required=True)
    ap.add_argument("--phase", choices=["collect", "analyze", "both"], default="both")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--samples", type=int, default=20)
    args = ap.parse_args()

    if args.phase in ("collect", "both"):
        collect(args.model, n_points=args.n, samples=args.samples)
    if args.phase in ("analyze", "both"):
        analyze(args.model)
