# Phase 2 — 多標的微調 (Option B)

**假設**: 加入更多 crypto 標的 (ETH/SOL/BNB ...) 進訓練集，模型能學到更普適的 crypto 動態，BTC 預測泛化更好。

**成功標準** (在 BTC OOS 2025-09-15+ 上):
- ✅ acc ≥ 53%
- ✅ IC ≥ +0.05（正且顯著）
- ✅ 加入 0.08% 交易成本後 cum PnL > 0
- ✅ Walk-forward 前/後段同號

---

## To-Do List

### Step 1: 資料下載 (~10 分)
- [ ] 用 freqtrade 下載 15m K 線:
      `ETH/USDT, SOL/USDT, BNB/USDT, XRP/USDT, ADA/USDT`
      timerange `20200101-`，存到 `quant-btc-gold/user_data/data/binance/`
- [ ] 不同幣種上幣時間不同 (SOL/BNB 可能較晚) — 各自能取多少就多少
- [ ] 印出各標的時間範圍 & 筆數

### Step 2: 多標的資料準備 (~5 分)
- [ ] 改寫 `finetune_btc/prep_data.py` → `prep_multi.py`
  - 為每個 symbol 各自跑同樣的 vol/amt 計算
  - 全部塞進 `{symbol: DataFrame}` 字典
  - 同樣切 train/val/test 時段，每個 symbol 在每段都有切片
- [ ] 訓練集應有 BTC + 4-5 個其他幣，最近 6 個月留作 test (純 BTC 評估)
- [ ] 存 `finetune_btc/data_multi/{train,val,test}_data.pkl`

### Step 3: 修改 config 並訓練 (~5h GPU)
- [ ] 更新 `repo/finetune/config.py`:
  - `dataset_path = "/home/anson/Kronos/finetune_btc/data_multi"`
  - `tokenizer_save_folder_name = 'btc_multi_tokenizer'`
  - `predictor_save_folder_name = 'btc_multi_predictor'`
  - `epochs = 15` (增加)
  - `n_train_iter = 1500 * batch_size` (增加)
  - `predictor_learning_rate = 2e-5` (略降，避免 overfit)
- [ ] 關掉 llama-server (`kill PID`)
- [ ] `torchrun --standalone --nproc_per_node=1 train_tokenizer.py` (~15 分)
- [ ] `torchrun --standalone --nproc_per_node=1 train_predictor.py` (~5 小時)
- [ ] 監看 train/val loss 曲線；觀察是否 epoch 8-10 後仍持續下降
- [ ] 如果 val loss 提早停滯 → 加 dropout 或減少 LR

### Step 4: 再次驗證 alpha (~17 分)
- [ ] 複製 `alpha_v3_finetuned.py` → `alpha_v3_multi.py`
- [ ] 指向新的 multi-symbol checkpoint
- [ ] 用同一個 BTC OOS 區間 (2025-09-15+)，300 點 × 20 採樣
- [ ] 比對指標 vs Phase 1 微調版本

### Step 5: 結論
- [ ] 若達到所有成功標準 → 進入 Phase 3 (Freqtrade 整合)
- [ ] 若部分達標 (例如 acc 提升但成本後仍負) → 評估是否再延長訓練 / 加更多標的
- [ ] 若完全沒改善 → 接受 Kronos 主訊號不行的結論，轉 Phase 3-alt: Kronos 作為 ensemble feature

---

## Phase 3 — 整合（取決於 Phase 2 結果）

**3a. 主訊號整合（若 Phase 2 達標）**
- [ ] 在 `quant-btc-gold/user_data/strategies/` 寫新 strategy
- [ ] 在 `populate_indicators` 中以 Kronos 預測為 indicator (用快取/cron 避開逐 bar 推論)
- [ ] 訊號: `prob_up_3h > 0.6` → enter long; `< 0.4` → enter short
- [ ] Backtest 2024-2025

**3b. Ensemble 整合（若 Phase 2 只能勉強過 50%）**
- [ ] 把 Kronos 預測收益 (mean, std, prob_up) 當特徵
- [ ] 餵入既有 `freqaimodels/` 的 LightGBM/XGBoost
- [ ] 對比加 Kronos feature 前後的 oos accuracy/Sharpe

---

## 可能失敗模式 & 應對

| 風險 | 跡象 | 應對 |
|------|------|------|
| 多標的反而互相干擾 | 訓練 loss 下降但 BTC val loss 上升 | 降權其他標的 / 改成 BTC fine-tune over multi-pretrained |
| 訓練時間爆掉 | epoch 1 超過 30 分鐘 | 減小 lookback (256→128) 或 batch 提高吞吐 |
| 持續 overfit | val loss 在 epoch 3-5 反彈 | 加 dropout、weight decay 提高、減少 epoch |
| 不同幣種上幣時間差異引入 leakage | val/test 時段中某些 symbol 沒有資料 | 嚴格用「上幣前」的時間做訓練 |
