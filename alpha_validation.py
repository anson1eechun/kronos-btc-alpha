"""
離線 Alpha 驗證：Kronos 對 BTC 的方向預測準確率。

流程：
  1. 載入 BTC 1h K 線
  2. 在歷史中隨機/等距抽 N 個時間點作為「現在」
  3. 每點: lookback=400, 預測未來 pred_len=24
  4. 採樣 SAMPLES 次 → 計算「H 小時後 close 上漲的機率」
  5. 與實際比對 → directional accuracy / 簡易 PnL
"""

import sys, time, argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append("/home/anson/Kronos/repo")
from model import Kronos, KronosTokenizer, KronosPredictor

DATA = "/home/anson/quant-btc-gold/user_data/data/binance/BTC_USDT-1h.feather"
TOK_PATH = "/home/anson/Kronos/models/Kronos-Tokenizer-base"
MDL_PATH = "/home/anson/Kronos/models/Kronos-base"


def run(n_points=50, lookback=400, pred_len=24, samples=20, horizon=24,
        seed=42, freq="1h", out_csv=None):
    df = pd.read_feather(DATA)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)
    df = df.sort_values("date").reset_index(drop=True)

    # 只用最近 ~2 年作為測試集，避免遠古資料分布差異
    df = df.tail(24 * 365 * 2).reset_index(drop=True)

    # 抽 n_points 個測試點：等距取樣（更穩定可重現）
    min_idx = lookback
    max_idx = len(df) - pred_len - 1
    test_idx = np.linspace(min_idx, max_idx, n_points, dtype=int)

    tok = KronosTokenizer.from_pretrained(TOK_PATH)
    mdl = Kronos.from_pretrained(MDL_PATH)
    predictor = KronosPredictor(mdl, tok, device="cuda:0", max_context=512)

    rng = np.random.default_rng(seed)
    rows = []
    t0 = time.time()

    for k, i in enumerate(test_idx):
        x_df = df.iloc[i - lookback:i][["open", "high", "low", "close", "volume"]].reset_index(drop=True)
        x_ts = df.iloc[i - lookback:i]["date"].reset_index(drop=True)
        y_ts = df.iloc[i:i + pred_len]["date"].reset_index(drop=True)

        cur_close = float(df.iloc[i - 1]["close"])
        actual_close_h = float(df.iloc[i - 1 + horizon]["close"])
        actual_ret = (actual_close_h - cur_close) / cur_close

        # 採樣 samples 次 → 機率
        pred_closes_at_h = []
        # KronosPredictor 已內建 sample_count，但每次採樣輸出一條 path 的均值，
        # 我們要原始多條 path 統計 → 用 for-loop 控制
        for s in range(samples):
            pred_df = predictor.predict(
                df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
                pred_len=pred_len, T=1.0, top_p=0.9, sample_count=1,
                verbose=False,
            )
            pred_closes_at_h.append(float(pred_df["close"].iloc[horizon - 1]))

        pred_closes_at_h = np.array(pred_closes_at_h)
        pred_rets = (pred_closes_at_h - cur_close) / cur_close
        prob_up = float((pred_rets > 0).mean())
        pred_mean_ret = float(pred_rets.mean())
        pred_med_ret = float(np.median(pred_rets))

        rows.append({
            "idx": int(i),
            "time": df.iloc[i - 1]["date"],
            "cur_close": cur_close,
            "actual_ret": actual_ret,
            "prob_up": prob_up,
            "pred_mean_ret": pred_mean_ret,
            "pred_med_ret": pred_med_ret,
            "pred_std_ret": float(pred_rets.std()),
        })

        elapsed = time.time() - t0
        eta = elapsed / (k + 1) * (len(test_idx) - k - 1)
        print(f"[{k+1}/{len(test_idx)}] idx={i} prob_up={prob_up:.2f} "
              f"pred_mean={pred_mean_ret:+.3%} actual={actual_ret:+.3%} "
              f"elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

    res = pd.DataFrame(rows)
    if out_csv:
        res.to_csv(out_csv, index=False)
    summarize(res)
    return res


def summarize(res: pd.DataFrame):
    n = len(res)
    # 方向準確率（用機率 > 0.5 當預測方向）
    pred_dir = (res["prob_up"] > 0.5).astype(int)
    actual_dir = (res["actual_ret"] > 0).astype(int)
    acc = (pred_dir == actual_dir).mean()

    # 用平均預測收益的方向
    pred_dir2 = (res["pred_mean_ret"] > 0).astype(int)
    acc2 = (pred_dir2 == actual_dir).mean()

    # IC: 預測收益 與 實際收益 的 Spearman
    ic = res["pred_mean_ret"].corr(res["actual_ret"], method="spearman")

    # 簡易 PnL：訊號 = sign(prob_up - 0.5)，position = ±1，計算 horizon 收益總和
    sig = np.where(res["prob_up"] > 0.5, 1, -1)
    pnl = (sig * res["actual_ret"]).sum()
    win_rate = ((sig * res["actual_ret"]) > 0).mean()

    # 高信心子集（|prob - 0.5| > 0.2）
    hi = res[(res["prob_up"] > 0.7) | (res["prob_up"] < 0.3)]
    hi_acc = ((hi["prob_up"] > 0.5).astype(int) == (hi["actual_ret"] > 0).astype(int)).mean() \
        if len(hi) else float("nan")

    print("\n========= ALPHA SUMMARY =========")
    print(f"樣本數              : {n}")
    print(f"方向準確率 (prob>0.5): {acc:.3f}")
    print(f"方向準確率 (mean_ret): {acc2:.3f}")
    print(f"IC (Spearman)        : {ic:+.3f}")
    print(f"模擬 PnL 累積收益    : {pnl:+.3%}  (每筆 horizon 收益相加)")
    print(f"勝率                 : {win_rate:.3f}")
    print(f"高信心樣本 (|p-.5|>.2): {len(hi)}  準確率: {hi_acc:.3f}")
    print("================================\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="測試點數")
    ap.add_argument("--samples", type=int, default=15, help="每點採樣次數")
    ap.add_argument("--horizon", type=int, default=24, help="預測 horizon（小時）")
    ap.add_argument("--lookback", type=int, default=400)
    ap.add_argument("--pred_len", type=int, default=24)
    ap.add_argument("--out", type=str, default="/home/anson/Kronos/alpha_results.csv")
    args = ap.parse_args()

    run(n_points=args.n, lookback=args.lookback, pred_len=args.pred_len,
        samples=args.samples, horizon=args.horizon, out_csv=args.out)
