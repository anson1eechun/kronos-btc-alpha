"""
離線 Alpha 驗證 v2
- 多時間框架: 15m / 30m / 1h / 4h
- 多 horizon (從同一次 24 根預測中萃取多個前瞻點)
- Long-only / Short-only 拆分
- Sharpe / Max Drawdown / Win rate
- 100 點 × 10 採樣 walk-forward
"""

import sys, time, argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append("/home/anson/Kronos/repo")
from model import Kronos, KronosTokenizer, KronosPredictor

DATA_DIR = "/home/anson/quant-btc-gold/user_data/data/binance"
TOK_PATH = "/home/anson/Kronos/models/Kronos-Tokenizer-base"
MDL_PATH = "/home/anson/Kronos/models/Kronos-base"

# 每個 TF: (檔名, 1 根對應的小時數, 要評估的 horizon 索引 -> 名稱)
TF_CFG = {
    "15m": dict(file="BTC_USDT-15m.feather", bar_h=0.25,
                horizons={1: "15m", 4: "1h", 12: "3h", 24: "6h"}),
    "30m": dict(file="BTC_USDT-30m.feather", bar_h=0.5,
                horizons={1: "30m", 4: "2h", 12: "6h", 24: "12h"}),
    "1h":  dict(file="BTC_USDT-1h.feather",  bar_h=1.0,
                horizons={1: "1h", 4: "4h", 12: "12h", 24: "24h"}),
    "4h":  dict(file="BTC_USDT-4h.feather",  bar_h=4.0,
                horizons={1: "4h", 3: "12h", 6: "24h", 12: "48h"}),
}


def load_predictor():
    tok = KronosTokenizer.from_pretrained(TOK_PATH)
    mdl = Kronos.from_pretrained(MDL_PATH)
    return KronosPredictor(mdl, tok, device="cuda:0", max_context=512)


def run_tf(tf: str, predictor, n_points=100, lookback=400, pred_len=24,
           samples=10, max_years=2):
    cfg = TF_CFG[tf]
    df = pd.read_feather(f"{DATA_DIR}/{cfg['file']}")
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)
    df = df.sort_values("date").reset_index(drop=True)

    # 限制最近 max_years
    bars_per_year = int(round(365 * 24 / cfg["bar_h"]))
    df = df.tail(bars_per_year * max_years).reset_index(drop=True)

    min_idx = lookback
    max_idx = len(df) - pred_len - 1
    test_idx = np.linspace(min_idx, max_idx, n_points, dtype=int)

    rows = []
    t0 = time.time()
    for k, i in enumerate(test_idx):
        x_df = df.iloc[i - lookback:i][["open", "high", "low", "close", "volume"]].reset_index(drop=True)
        x_ts = df.iloc[i - lookback:i]["date"].reset_index(drop=True)
        y_ts = df.iloc[i:i + pred_len]["date"].reset_index(drop=True)
        cur_close = float(df.iloc[i - 1]["close"])

        # 採樣 samples 次，每次取得整條 24 根路徑
        paths = np.zeros((samples, pred_len), dtype=np.float32)
        for s in range(samples):
            pred_df = predictor.predict(
                df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
                pred_len=pred_len, T=1.0, top_p=0.9, sample_count=1,
                verbose=False,
            )
            paths[s] = pred_df["close"].values

        # 對每個 horizon 萃取 prob_up / mean_ret / actual_ret
        row = {"tf": tf, "idx": int(i), "time": df.iloc[i - 1]["date"], "cur_close": cur_close}
        for h, name in cfg["horizons"].items():
            pred_closes = paths[:, h - 1]
            pred_rets = (pred_closes - cur_close) / cur_close
            actual_close = float(df.iloc[i - 1 + h]["close"])
            actual_ret = (actual_close - cur_close) / cur_close
            row[f"prob_up_{name}"] = float((pred_rets > 0).mean())
            row[f"pred_mean_{name}"] = float(pred_rets.mean())
            row[f"pred_std_{name}"] = float(pred_rets.std())
            row[f"actual_{name}"] = actual_ret
        rows.append(row)

        elapsed = time.time() - t0
        eta = elapsed / (k + 1) * (len(test_idx) - k - 1)
        if (k + 1) % 10 == 0 or k == 0:
            print(f"  [{tf}] {k+1}/{len(test_idx)} elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

    return pd.DataFrame(rows)


def sharpe(rets: np.ndarray, periods_per_year: float) -> float:
    if len(rets) < 2 or rets.std() == 0:
        return float("nan")
    return float(np.sqrt(periods_per_year) * rets.mean() / rets.std())


def max_drawdown(rets: np.ndarray) -> float:
    equity = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(dd.min())


def summarize_horizon(res: pd.DataFrame, tf: str, h_name: str, bar_h: float, horizon_bars: int):
    prob = res[f"prob_up_{h_name}"].values
    pmean = res[f"pred_mean_{h_name}"].values
    actual = res[f"actual_{h_name}"].values

    # 方向準確率
    pred_dir = (prob > 0.5).astype(int)
    actual_dir = (actual > 0).astype(int)
    acc_all = (pred_dir == actual_dir).mean()

    # Long / Short 分開
    long_mask = prob > 0.5
    short_mask = prob < 0.5
    acc_long = (actual[long_mask] > 0).mean() if long_mask.sum() else float("nan")
    acc_short = (actual[short_mask] < 0).mean() if short_mask.sum() else float("nan")

    # IC
    ic = pd.Series(pmean).corr(pd.Series(actual), method="spearman")

    # 簡易策略: sign(prob - 0.5) 的 horizon-period returns
    sig = np.where(prob > 0.5, 1, -1)
    strat_rets = sig * actual
    cum_pnl = float(np.sum(strat_rets))
    cum_pnl_compound = float(np.prod(1 + strat_rets) - 1)
    win_rate = float((strat_rets > 0).mean())

    # 年化 Sharpe (假設每筆獨立、無重疊；horizon 越長越粗略)
    horizon_hours = bar_h * horizon_bars
    periods_per_year = 24 * 365 / horizon_hours
    shp = sharpe(strat_rets, periods_per_year)
    mdd = max_drawdown(strat_rets)

    # 高信心子集
    hi_mask = (prob > 0.7) | (prob < 0.3)
    if hi_mask.sum():
        hi_pred = (prob[hi_mask] > 0.5).astype(int)
        hi_act = (actual[hi_mask] > 0).astype(int)
        hi_acc = float((hi_pred == hi_act).mean())
        hi_n = int(hi_mask.sum())
    else:
        hi_acc, hi_n = float("nan"), 0

    return {
        "tf": tf, "horizon": h_name, "n": len(res),
        "acc_all": acc_all, "acc_long": acc_long, "acc_short": acc_short,
        "ic": ic,
        "cum_pnl_sum": cum_pnl, "cum_pnl_cmp": cum_pnl_compound,
        "win_rate": win_rate, "sharpe": shp, "max_dd": mdd,
        "hi_n": hi_n, "hi_acc": hi_acc,
        "n_long": int(long_mask.sum()), "n_short": int(short_mask.sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--tfs", nargs="+", default=["15m", "30m", "1h", "4h"])
    ap.add_argument("--out_dir", type=str, default="/home/anson/Kronos/alpha_v2")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(exist_ok=True)
    predictor = load_predictor()

    summary_rows = []
    for tf in args.tfs:
        print(f"\n===== TF {tf} =====")
        df_res = run_tf(tf, predictor, n_points=args.n, samples=args.samples)
        df_res.to_csv(out / f"raw_{tf}.csv", index=False)

        cfg = TF_CFG[tf]
        for h_bars, h_name in cfg["horizons"].items():
            s = summarize_horizon(df_res, tf, h_name, cfg["bar_h"], h_bars)
            summary_rows.append(s)

    sdf = pd.DataFrame(summary_rows)
    sdf.to_csv(out / "summary.csv", index=False)

    # 漂亮輸出
    print("\n\n=================== ALPHA SUMMARY ALL ===================")
    cols = ["tf", "horizon", "n", "acc_all", "acc_long", "acc_short", "ic",
            "win_rate", "sharpe", "max_dd", "cum_pnl_cmp", "hi_n", "hi_acc"]
    print(sdf[cols].to_string(index=False,
        formatters={
            "acc_all": "{:.3f}".format, "acc_long": "{:.3f}".format,
            "acc_short": "{:.3f}".format, "ic": "{:+.3f}".format,
            "win_rate": "{:.3f}".format, "sharpe": "{:+.2f}".format,
            "max_dd": "{:.2%}".format, "cum_pnl_cmp": "{:+.2%}".format,
            "hi_acc": "{:.3f}".format,
        }))


if __name__ == "__main__":
    main()
