# kronos-btc-alpha

評估 [Kronos](https://github.com/shiyu-coder/Kronos)（K 線基礎模型）能不能拿來當 BTC 的交易訊號。做了 5 輪離線驗證和 3 種微調，結論是**不能**：扣掉 0.08% 來回成本後，所有設定的累積報酬都是負的，walk-forward 前後段也不穩定。

這個 repo 只放實驗程式、log 和報告；Kronos 本體與模型權重請從上游取得。

## 結果

| 輪次 | 設定 | 樣本 | 方向準確率 | 扣成本後 | 判讀 |
|---|---|---|---|---|---|
| v1 | 預訓練，1h × 24h horizon | 30 點 | 73%，IC +0.42 | — | 小樣本假象 |
| v2 | 預訓練，15m / 30m / 1h / 4h × 多 horizon | 100 點 | 只有 15m × 3h 看似可行（64%） | 其餘 Sharpe 全負 | 訊號疑似集中在短週期 |
| v3 | 預訓練，15m × 3h | 300 點 | 47.3%，IC +0.019 | 累積 −23.3% | 前述訊號消失 |
| v3 OOS | 預訓練 vs BTC 微調，2025-09-15 之後 | 300 點 | 46.3% → 51.0% | −27.7% → −17.8% | 微調修正了 bear bias，但仍虧 |
| v3 multi | 6 幣（BTC/ETH/SOL/BNB/XRP/ADA）15m 微調 | 300 點 | 49.7% | −20.0% | 多標的沒有幫助 |
| v4 | 1h × 12h，BTC + 黃金、白銀、ES、NQ 多資產微調 | 300 點 | 50.3% | −49.2% | 比預訓練（−18.5%）更差 |

**主要發現**

- **樣本越大，IC 越接近 0**：30 點 0.42 → 100 點 0.15 → 300 點 0.02。初步結果至少要 300 點 walk-forward 才能信。
- **預訓練模型對 BTC 有強烈的看空偏誤**：整體只有 26% 預測上漲，上漲盤勢中也只有 11%。推測是預訓練資料（以中國 A 股為主）和 BTC 的報酬分布差太多。
- **微調能校準偏誤**（看多比例 30% → 52%），但提升不了預測力。

完整過程、數字與踩過的坑在 [REPORT.md](REPORT.md)；Phase 2 多標的計畫在 [TODO.md](TODO.md)。

## 方法

每個驗證點取 lookback 視窗（400 根或 256 根），讓 Kronos 預測未來 12–24 根並抽樣 10–20 條路徑，以「horizon 後收盤高於現價的路徑比例」當看漲機率，再對照實際報酬。每輪都拆：

- 抽樣數敏感度（5 / 10 / 20 條）
- 交易成本（0 / 0.04% / 0.08%）
- walk-forward（前 60% vs 後 40%）
- 盤勢（依 200 根 SMA 斜率分上漲、盤整、下跌）
- 高信心門檻（|機率 − 0.5| > 0.2）

微調的驗證區間一律從 2025-09-15 開始，避開微調用過的 train / val 區間。

## 檔案

| 檔案 | 用途 |
|---|---|
| `alpha_validation.py` / `alpha_validation_v2.py` | v1、v2 驗證 |
| `alpha_v3_15m_3h.py` / `alpha_v3_baseline_oos.py` / `alpha_v3_finetuned.py` / `alpha_v3_multi.py` | v3 各組：預訓練、OOS 基準、BTC 微調、多標的微調 |
| `alpha_v4_1h.py` | v4 1h 驗證，`--model` 切換 baseline / finetuned |
| `finetune_btc/prep_data.py` | BTC 15m → Kronos 微調格式（train / val / test pickle） |
| `finetune_btc/prep_multi.py` | 6 幣 15m 微調資料；test 集只留 BTC |
| `finetune_btc/prep_data_multi_1h.py` | BTC + yfinance 期貨 1h 微調資料 |
| `kronos-finetune-config.patch` | 對上游 `finetune/config.py` 的修改 |
| `logs/` | 每輪完整輸出 |
| `results/alpha_v2_summary.csv` | v2 各週期 × horizon 摘要 |

## 重現

```bash
git clone https://github.com/shiyu-coder/Kronos repo
cd repo && git apply ../kronos-finetune-config.patch && cd ..
# 從 Hugging Face 下載 NeoQuasar/Kronos-base 與 Kronos-Tokenizer-base 到 models/
```

K 線資料由 Freqtrade 下載（Binance）。腳本裡的資料與模型路徑寫死在 `/home/anson/...`，需要依環境修改。原本在 RTX 2080 Ti 22GB 上執行；微調前要先關掉占用 VRAM 的 llama-server。

Kronos 以 MIT 授權釋出，版權屬原作者。
