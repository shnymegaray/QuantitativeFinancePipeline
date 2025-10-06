Pipeline

Goal: Estimate next‑day direction from simple return‑based features using:
1) Similarity kernel in feature space to get an up‑probability `q`  
2) Exponential relaxation (ODE‑like) to blend prior stock probability `p_current` toward `q` → `p_next`  
3) Thresholding: `p_next` → {INVEST, UNSURE, PULL OUT}

- Model achieved ~54.3% accuracy over 100 runs after default parameters were set

> Academic framing: This is educational research code (non‑investment advice until further improved on). Used in Wharton Investment Competition.  
> Data via `yfinance` (Yahoo Finance).



- Quickstart (Colab or local)

- Google Colab (RECOMMENDED): run `pipeline.py` as-is. It will save to `/content/...` and prompt a download.  
- Local: change the `output_path` string near the bottom to your preferred folder.

```bash
pip install -r requirements.txt
python pipeline.py
```



- Files

- `pipeline.py` — main script with clear comments, light input validation, and optional progress logging.   
- `requirements.txt` — tested versions.
- `Examples` — pre-made files for Apple, Tesla, and GameStop stocks from 01/01/2022 - 10/05/25.



- Method sketch

- Features per day from `Average = (Open+High+Low+Close)/4`:
  - `r_t` = daily percent change;  
  - `m_t` = |`r_t`|;  
  - `v_t` = RMS of `r` over window `W`;  
  - `a_t` = Δ`r_t`;  
  - `c_t` = `r_t - 2r_{t-1} + r_{t-2}`.  
- Kernel: For current `x_t`, compute Mahalanobis‑like distances to up to `N_MAX` past points using covariance `Σ`, weight with Gaussian `exp(-d^2/(2H^2))`, and estimate `q = Σ w_i  y_i / Σ w_i` where `y_i = 1` if next day was up.  
- Relaxation (ODE‑like): `p_next = q + (p_current - q)  exp(-k_t)`, where `k_t = K0 + K1m_t + K2v_t`.  
- Decision:  
  - `p_next ≥ TAU_HIGH` → INVEST  
  - `p_next ≤ TAU_LOW` → PULL OUT  
  - otherwise UNSURE.



- Evaluation convention

When you open the results sheet:
- The decision recorded at date t+1 is made using information up to t.  
- Next‑day correctness is checked against return at t+1.
- INVEST if p_t+1 ≥ TAU_HIGH  
- PULL OUT if p_t+1 ≤ TAU_LOW   
- UNSURE if TAU_LOW < p_t+1 < TAU_HIGH


- Parameters (defaults; calculated from 80 runs, where the amount of incorrect recommendations was minimized)

- `W=20` (rolling RMS window), `N_MAX=250`, `H=1.0`  
- `K0=0.5, K1=10.0, K2=5.0`  
- `TAU_LOW=0.47, TAU_HIGH=0.52`  
- `RIDGE_EPS=1e-8`, `MIN_HISTORY=30`, `TINY=1e-12`

- Across 100 independent runs (AAPL, GME, TSLA, etc.), the model achieved ~54.3% next-day directional accuracy using these parameters (this was done after calculating defaults from the 80 run trial).

These reflect the model-level thresholds that maximized micro‑averaged accuracy on a multi‑run trial.



- Limitations & next steps

- Horizon is strictly next day. Extending to multi‑day horizons or intraday bars would require re‑labeling and feature changes.  
- Stationarity is assumed implicitly; market regime shifts can break the similarity kernel.  
- UNSURE correctness depends on an arbitrarily chosen flat band; be explicit in reports.  
- Walk‑forward robustness could be improved with rolling re‑tuning of `TAU_` and/or `H` using only past data to reduce look‑ahead bias.



- Reproducibility

- See `requirements.txt` for versions.