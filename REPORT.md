# Kronos × BTC 交易系統可行性研究 — 階段報告

**日期**: 2026-05-15
**目標**: 評估 Kronos 預測模型用於 BTC 交易訊號的可行性
**範圍**: 離線 alpha 驗證 + 微調實驗

---

## 0. 環境設置

| 項目 | 內容 |
|------|------|
| GPU | RTX 2080 Ti (22 GB) |
| Repo | https://github.com/shiyu-coder/Kronos |
| 預訓練模型 | NeoQuasar/Kronos-base (102M params) + Kronos-Tokenizer-base |
| 主要 Python | /home/anson/quant-btc-gold/.venv (Python 3.11 + torch 2.5.1+cu121) |
| 資料來源 | Binance BTC/USDT (Freqtrade 下載) |
| 工作目錄 | /home/anson/Kronos/ |

---

## 1. 實驗時序與發現

### 1.1 第一輪：30 點驗證 → 假陽性
- 樣本: 30 點 × 15 採樣，1h K線 × 24h horizon
- 結果: **acc 73%, IC +0.42, cum +22.6%** 看似很強
- 結論: 後續發現是 small-sample fluke

### 1.2 第二輪：100 點 × 4 timeframe (15m/30m/1h/4h) × 多 horizon
- 唯一看似有 alpha: **15m × 3h horizon (acc 64%, IC +0.15, cum +35.6%)**
- 其他組合 Sharpe 全為負，長 horizon 全失敗
- 結論: 信號疑似集中在短週期短預測

### 1.3 第三輪：300 點深度驗證 15m × 3h
- acc 47.3%, IC +0.019, cum -2.5% (零成本)
- 加入 0.08% round-trip 成本後: cum -23.3%
- Walk-forward 前 60% acc 38% vs 後 40% acc 61% **嚴重不穩**
- **核心發現**: 模型整體 `long_share = 26%`，UP regime 也只 11% 預測漲
- 結論: **預訓練 Kronos 對 BTC 有強烈 bear bias**（中國 A 股訓練 vs BTC 主升市分布差異）

### 1.4 第四輪：微調後 (300 點 OOS 2025-09-15+)
- 微調設定: 5 epochs, n_iter=800×batch_size=32, lookback=256, predict=12
- Tokenizer val loss: 0.0050 → 0.0046
- Predictor val loss: 2.6110 → 2.5996 (epoch 3 最佳，4-5 反彈)

對比（同 OOS 區間）:

| 指標 | 預訓練 | 微調 | 變化 |
|------|--------|------|------|
| 方向準確率 | 46.3% | **51.0%** | +4.7% ✓ |
| Spearman IC | -0.094 | -0.052 | +0.04 ✓ |
| Long 機率 | 30% | **52%** | bear bias 大幅修正 ✓ |
| 累積 PnL (0 cost) | -8.12% | **+4.54%** | 翻正 ✓ |
| 累積 PnL (0.08% cost) | -27.74% | -17.78% | 仍負 ✗ |
| Walk-forward 穩定性 | 不穩 | 不穩 | 未改善 ✗ |

---

## 2. 結論

1. **預訓練 Kronos-base 不能直接用於 BTC**：中國股票分布與加密貨幣分布差太多
2. **短時微調有「校準」效果**：bear bias 大幅修正，accuracy 從 46% 升至 51%
3. **單獨作為交易主訊號仍不足**：IC 仍負、成本後仍虧損、walk-forward 不穩
4. **訓練未飽和**：Val loss epoch 3 觸底但 train loss 仍可下降，表示需要更多資料/輪次/正則化

---

## 3. 已產出的程式碼與資料

```
/home/anson/Kronos/
├── repo/                              # GitHub clone
│   └── finetune/config.py             # 已客製化為 BTC 設定
├── models/
│   ├── Kronos-base/                   # 預訓練 predictor (391 MB)
│   └── Kronos-Tokenizer-base/         # 預訓練 tokenizer (16 MB)
├── finetune_btc/
│   ├── prep_data.py                   # BTC 15m → pickle 格式
│   ├── data/{train,val,test}_data.pkl # 訓練資料 (175k + 38k + 23k bars)
│   └── outputs/models/btc_{tokenizer,predictor}/checkpoints/best_model/
├── alpha_validation.py                # v1: 30 點測試
├── alpha_validation_v2.py             # v2: 4 TF × 100 pts × 多 horizon
├── alpha_v3_15m_3h.py                 # v3: 預訓練 15m×3h 300 pts (近 2 年)
├── alpha_v3_baseline_oos.py           # v3: 預訓練 15m×3h OOS 區間
├── alpha_v3_finetuned.py              # v3: 微調後 15m×3h OOS 區間
├── alpha_v3/, alpha_v3_ft/, alpha_v3_base_oos/   # 原始預測路徑 pickle + summary CSV
├── alpha_v*.log                       # 各次跑出的完整 log
└── REPORT.md                          # 本檔
```

---

## 4. 已知坑（給未來自己）

1. **`MEMORY: small samples lie hard`**: 30 點 IC 0.42、100 點 IC 0.15、300 點 IC 0.02 — 樣本越多越接近 0。任何「初步看起來不錯」都要至少 300 點 walk-forward 才能信
2. **預測點不能踩到訓練/驗證集**: 微調用了 2024-09~2025-09 的 val 區間，alpha 驗證必須切到 2025-09-15+
3. **VRAM 衝突**: llama-server 默認吃 19.8 GB，微調前要關掉或切換模型 (kill PID)
4. **Comet ML 必裝**: `train_predictor.py` 第 12 行 `import comet_ml` 無條件，即使 use_comet=False
5. **Bear bias 機制**: 預訓練資料 (中國 A 股) 平均上漲幅度低，模型 learn 到「均值回歸」反應，套到主升的 BTC 上就全方向反了
