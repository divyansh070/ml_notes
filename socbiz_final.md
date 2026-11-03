# MAPC: EV Charging Dynamic Tariff — Revision Notes

**Project:** Multi-Agent Predictive Control (MAPC) for EV Charging Networks
**Author:** Divyansh Verma (23112035)
**Purpose:** Interview revision. Each agent: what it does, input, output, maths, trade-offs, likely questions. Written to match what the notebook *actually* implements.

---

## 0. One-Minute Mental Model

```
PREDICT → MODEL BEHAVIOUR → OPTIMISE → CONSTRAIN → MEASURE
```

Think of one charging station run by a small team. Every 5-minute step:

| # | Agent | Question it answers |
|---|---|---|
| 1 | Forecasting | How busy will this station be in ~1 hour? |
| 2 | Elasticity | How price-sensitive are drivers right now? |
| 3 | Economist | Which price maximises profit? |
| 4 | Congestion | Is the grid in danger? If yes, what is the minimum price? |
| 5 | Mediator | Combine both, enforce safety, limit price change to ±20% |
| 6 | Queueing | At the resulting utilisation, how long do drivers wait? |

**Core design idea:** separate *learned prediction* (ML) from *explicit decision rules* (economics + safety). Prediction is data-driven; decisions are transparent and auditable.

---

## 1. Complete Flow

```
         UrbanEV (Shenzhen)                    ACN (Caltech/JPL)
   zone × 5-min occupancy, kWh, chargers     session-level behaviour
                 │                                    │
                 ▼                                    ▼
        ┌─────────────────┐                ┌────────────────────┐
        │ Forecasting     │                │ ACN calibration    │
        │ (3 × LightGBM)  │                │ μ = 1/402.2 min    │
        └────────┬────────┘                │ hour → elasticity  │
                 │                         └─────────┬──────────┘
   pred_load, pred_util, prob_cong                   │
        ┌────────┼──────────────────┐                │
        │        │                  │                ▼
        │        │          ┌───────┴──────┐  ┌──────────────┐
        │        │          │  Economist   │◄─│  Elasticity  │ (hour, CBD)
        │        │          │ 50-price grid│  └──────────────┘
        │        │          └───────┬──────┘
        │        ▼                  │ econ_price
        │  ┌────────────┐           │
        │  │ Congestion │           │
        │  │ 85% rule   │           │
        │  └─────┬──────┘           │
        │        │ risk, floor      │
        │        ▼                  ▼
        │      ┌──────────────────────┐
        │      │      Mediator        │◄── previous price (memory)
        │      │ max → ±20% → [8,30]  │
        │      └──────────┬───────────┘
        │                 │ final tariff
        │                 ▼
        │   demand response: load × (p/15)^e
        ▼                 ▼
   ┌────────────────────────────┐
   │ Queueing (Erlang-C)        │  → wait time (evaluation only)
   └────────────────────────────┘
```

**Simulation loop (per test row, 2,000 rows):**

1. Look up TOU wholesale cost for the hour.
2. Forecasting Agent → `pred_load`, `pred_util`, `prob_cong`.
3. Economist (calls Elasticity internally) → `econ_price`.
4. Congestion Agent → `(risk, floor)`.
5. Mediator → `final_price`; stored as the station's new current price.
6. Demand response: `mult = (final_price/15)^e`; `agent_load = pred_load × mult`; `agent_util = pred_util × mult`.
7. Baseline: flat ₹15, `pred_load`, `pred_util`.
8. Queueing Agent computes wait for both.
9. Aggregate → profit gain, off-peak uplift, congestion drop, wait reduction.

---

## 2. Datasets

### 2.1 UrbanEV (Shenzhen) — "How busy is the network?"

- Public charging data, Shenzhen, **5-minute intervals**, ~24,798 piles (dataset-level figure).
- **Aggregated by zone (grid)**, not per car. In the code, `station_id` = **zone ID**. "Station 102" = zone 102.
- Files are **wide** (rows = timestamps, columns = zones) and are melted to long format.

| File | Contains | Used for |
|---|---|---|
| `occupancy.csv` | chargers in use per zone | `active_chargers` → utilisation |
| `volume.csv` | kWh delivered per zone | load target |
| `stations.csv` | `count`, `fast`, `slow` per zone | capacity features, `c` in queueing |
| `information.csv` | zone attributes incl. `CBD` | CBD feature |
| `duration.csv` | avg charging duration | EDA only |
| `price.csv` | price per zone over time | EDA only |
| `time.csv` | timestamp → date/time | EDA |

```
utilization_rate = active_chargers / count        (clipped ≤ 1)
hour_of_day      = (timestamp // 12) % 24         (12 steps = 1 hour)
```

**Gives:** when/where demand happens, energy volume, capacity.
**Lacks:** per-driver sessions, dwell time per car, energy requested, usable price variation.

### 2.2 ACN (Caltech/JPL, California) — "How do individual drivers behave?"

- One row = one charging session (plug-in → unplug). App users also enter needs.
- Key fields: `connectionTime`, `disconnectTime`, `kWhDelivered`, `kWhRequested`, `minutesAvailable`.

| Derived | Formula | Used by |
|---|---|---|
| `occupancy_mins` | disconnect − connect (kept 5 min – 24 h) | Queueing: mean 402.2 min → μ |
| `urgency_kw` | kWhRequested ÷ (minutesAvailable/60) | Elasticity hourly map |

**Datasets are never joined row-by-row.** ACN only produces *parameters* (one μ, a 24-value elasticity map) used inside the Shenzhen pipeline.

### 2.3 Defending ACN for Shenzhen

> "UrbanEV tells me how busy stations are but not how individual drivers behave — no session-level energy requests or dwell times, and no usable price variation. ACN is one of the few public datasets with that. I use it as a **behavioural prior**, not Shenzhen ground truth, isolated in one agent so it can be replaced with local data. Economic results are conditional on that assumption."

- **Why not learn elasticity from UrbanEV prices?** Prices follow fixed TOU schedules → price is perfectly confounded with hour of day. You need price variation independent of time (experiments, unexpected tariff changes).
- **What is transferred:** the *relative pattern* of urgency across hours (rescaled to a chosen range) + one service-time number. Not absolute behaviour.
- **Known differences:** Caltech = workplace, slow L2 chargers, all-day parking; Shenzhen = public, many fast chargers, taxi/ride-hailing fleets.
- **Do NOT say** "physical laws of elasticity are global" or "deployable in any city on Day 1".

---

## 3. Forecasting Agent

**Role:** the "look-ahead". Predicts the station state ~1 hour ahead so pricing can act *before* congestion.

### Input
Features per row: `fast`, `slow` (charger counts), `util_lag_1`, `util_lag_2`, `load_lag_1`, `hist_station_time_util`, `CBD`, `hour_of_day`, `day_of_week`.

### Output
| Output | Model | Meaning |
|---|---|---|
| `pred_load` | LightGBM regressor | kWh demand in ~1 h (interpreted as demand at ₹15) |
| `pred_util` | LightGBM regressor | utilisation in ~1 h (clipped to 0.99) |
| `prob_cong` | LightGBM classifier | P(utilisation > 0.85) in ~1 h |

Targets are shifted by **12 steps** (12 × 5 min = 1 hour). Models: 300 trees, max depth 6.

### Maths — Gradient boosting (LightGBM)
```
F₀(x) = constant
Fₘ(x) = Fₘ₋₁(x) + η · hₘ(x)        hₘ = tree fitted to residuals / negative gradients
Final prediction = sum of all trees
```
LightGBM specifics: histogram-based splits, leaf-wise growth (`num_leaves`), fast on large tabular data.

### Metrics (load model)
```
R²   = 0.7167      → 71.67% of variance explained (NOT "71.67% accurate")
RMSE = 1.3691 kWh  → penalises large errors more (squares them)
MAE  = 0.9806 kWh  → average absolute error
RMSE ≥ MAE always; gap grows with large/outlier errors
```

### Why LightGBM
Tabular mixed features, captures non-linear interactions (hour × CBD), fast inference, strong without big neural nets.
**Why not LSTM/Transformer?** Lags + calendar features already encode the temporal structure; trees are faster, need less tuning, and are more interpretable for this tabular setup.

### Trade-offs
| Benefit | Cost |
|---|---|
| Fast, strong on tabular data | Point forecast — no uncertainty |
| Interpretable importances | Can't anticipate unseen events |
| Lags give short-term momentum | Hard 0.85 cutoff for the classifier label |

### ⚠️ Code reality
- **Split is by station, not by time.** Data sorted by `station_id` then time, first 80% rows = train → test = later stations over their full history. Say: *"evaluated on held-out stations."* It is NOT strict time-series CV.
- **`hist_station_time_util` leaks** — computed on the full dataset (incl. future). Fix: compute from training period only.
- **`spatial_neighbor_lag_1 = util_lag_1`** — a copy, not a real spatial feature.

---

## 4. Elasticity Agent

**Role:** tells the system how much demand reacts to price, right now, at this station.

### Input
- **Setup (once):** ACN sessions → `connectionTime`, `kWhRequested`, `minutesAvailable`.
- **Each step:** `hour_of_day`, `CBD`. (No price, load, or forecast.)

### Output
One number **e ∈ [−3.0, −0.1]**.

### What e means
```
1% price increase → ≈ e% change in demand
```
| e | Meaning |
|---|---|
| −0.2 | Inelastic (\|e\|<1) — desperate driver, barely reacts |
| −1.0 | Unit elastic — revenue unchanged |
| −2.0 | Elastic (\|e\|>1) — flexible driver, reacts strongly |

Example: price +20% (₹15 → ₹18). Multiplier = 1.2^e.
- e = −0.2 → 0.964 → demand −3.6%
- e = −2.0 → 0.694 → demand −30.6%

### Maths — building the map (Stage A, once)
```
urgency_kw      = kWhRequested / (minutesAvailable / 60)
u_h             = median urgency for hour h
normalised_h    = (u_h − min u) / (max u − min u)
e_h             = −2.5 + 2.3 × normalised_h        → range [−2.5, −0.2]
```
Logic: urgent drivers can't wait → less price-sensitive.

### Runtime rules (Stage B, each step)
```python
e = map.get(hour, -0.6)
if hour >= 22 or hour <= 6: e = min(e, -1.8)   # night forced elastic
if CBD == 1:               e += 0.5           # city centre less elastic
e = clip(e, -3.0, -0.1)
```
Example: hour 2, CBD → map −0.4 → night −1.8 → CBD **−1.3**.

Fallback map if ACN fails: −0.2 at hours 8, 9, 17, 18; −0.6 daytime; −1.8 otherwise.

### Trade-offs
| Benefit | Cost |
|---|---|
| Data-driven instead of an arbitrary constant | Proxy, not causal — no price ever changed in ACN |
| Isolated → swappable with real data | Range −2.5 to −0.2 is a design choice |
| Simple, explainable | Uses only hour + CBD; different city |

### ⚠️ Code reality
- **ACN timestamps are likely GMT.** Notebook notes sessions peak at 14:00–18:00 — that's ~7–9 AM Pacific. The "8–9 AM urgency" is likely ~1–2 AM local. Fix: `tz_convert('America/Los_Angeles')` before `.dt.hour`.
- **Night rule overrides the data** (ACN shows high urgency near hours 0/23, yet forced to −1.8).
- Verify session count (slide says 30,000+; check notebook output).

### Proper estimation (if asked)
Randomised pricing / A/B tests, difference-in-differences around tariff changes, instrumental variables, then:
```
log(Q) = α + e · log(p) + controls(hour, station) + ε      → e is the coefficient
```

---

## 5. Economist Agent

**Role:** profit-seeker. Finds the profit-maximising price, ignoring safety.

### Input
| Input | Source |
|---|---|
| e | Elasticity Agent |
| `pred_load` (Q₀) | Forecasting Agent — demand at ₹15 |
| `cost` | Hard-coded TOU schedule |
| `current_price` | Only an initial value — no real effect |

### Output
One price from **50 evenly spaced values in [₹8, ₹30]** (step ≈ ₹0.45). Discrete grid, not continuous.

### Wholesale cost (TOU, hard-coded, not from data)
```python
cost = 3.0 if (hour >= 22 or hour <= 6) else (9.0 if hour in [8,9,17,18] else 6.0)
```
| Period | Hours | ₹/kWh |
|---|---|---|
| Night (valley) | 22:00–06:59 | 3 |
| Peak | 8, 9, 17, 18 | 9 |
| Flat | rest | 6 |

### Maths
```
Demand (constant elasticity):   Q(p) = Q₀ × (p / 15)^e
Profit:                         π(p) = Q(p) × (p − c)
```
Set dπ/dp = 0:
```
p* = c × e / (1 + e)          (valid only for e < −1)
Equivalent Lerner rule:  (p − c) / p = −1 / e
```

**Three cases:**
| Elasticity | Result |
|---|---|
| −1 ≤ e < 0 | profit increases with p → **₹30 (cap)** |
| e < −1, p* < 8 | **₹8 (floor)** |
| e < −1, 8 ≤ p* ≤ 30 | **p*** |

Examples:
- Night non-CBD: c=3, e=−1.8 → p* = 6.75 → **₹8**
- Night CBD: e=−1.3 → p* = 3×1.3/0.3 = **₹13**
- Midday: c=6, e=−2.1 → p* = 6×2.1/1.1 ≈ **₹11.5**
- Any |e|<1 → **₹30**

→ Explains why the Economist line in the 24-h trace jumps between ₹8 and ₹30.

### Worked table (c=6, e=−2.1, Q₀=10)
| p | (p/15)^−2.1 | Load | Margin | Profit |
|---|---|---|---|---|
| 8 | 3.74 | 37.4 | 2 | 74.8 |
| 11.5 | 1.75 | 17.5 | 5.5 | **96.4** |
| 15 | 1.00 | 10.0 | 9 | 90.0 |
| 30 | 0.23 | 2.3 | 24 | 56.0 |

### 🔑 Key insight: forecast does not affect the Economist's price
```
π(p) = Q₀ × [(p/15)^e × (p − c)]
```
Q₀ is a positive constant multiplier → argmax independent of Q₀. Price depends only on **e and c** → only on **hour and CBD**. The forecast matters through the **Congestion Agent** and the **evaluation**, not the economic optimum.

> "Under constant elasticity the optimum is independent of demand level. To make the forecast matter economically, I'd add a capacity constraint or convex cost."

### Trade-offs
| Choice | Benefit | Cost |
|---|---|---|
| Separate profit agent | Clear, testable | Conflicts with safety → needs Mediator |
| Constant elasticity | One parameter, standard | Corner solutions when \|e\|<1 |
| 50-point grid | Simple, bounded, ~0.24 ms | ₹0.45 resolution; closed form exists |
| Per-step optimisation | Fast | Myopic, no load shifting across hours |

### ⚠️ Weaknesses
- No capacity constraint (at ₹8 demand can be 3.6× forecast).
- No demand shifting between hours — priced-out demand disappears.
- "Profit" = energy margin only (no fixed costs/demand charges/competition).
- Evaluation is partly circular (same demand model scores the result).
- Cost values and peak hours are assumptions; EDA's busiest time (midnight) is cheapest in this schedule. Currency: ₹ used for a Shenzhen model (illustrative).

---

## 6. Congestion Agent

**Role:** safety guard. Pushes price **up** when overload is predicted. Never lowers it for safety.

### Input
| Input | Source |
|---|---|
| `prob_cong` | Forecasting classifier — P(util > 0.85) in ~1 h |
| `pred_util` | Forecasting regressor |
| `hour` | current row |

(No price, elasticity, load, or Economist output.)

### Output
`(risk_label, min_price)`
| Label | Price | Used by Mediator? |
|---|---|---|
| CRITICAL | ₹22–₹30 | **Yes** |
| UNDERUTILIZED | ₹8 | No (ignored) |
| STABLE | ₹15 | No (ignored) |

### Logic (checked in order)
```python
if hour >= 23 or hour <= 5:                  return "UNDERUTILIZED", 8.0
if prob_cong > 0.85 or pred_util > 0.85:     return "CRITICAL", min(22 + 8*prob_cong, 30)
elif pred_util < 0.30:                       return "UNDERUTILIZED", 8.0
else:                                        return "STABLE", 15.0
```

### Maths
```
floor = min(22 + 8 × prob_cong, 30)
prob = 0.30 → ₹24.40   |   prob = 0.90 → ₹29.20   |   prob = 1.0 → ₹30
```
- Two 0.85 checks mean different things: *confidence* (prob) vs *capacity* (util).
- **OR** = conservative: either signal triggers.

### Why 85%
1. Erlang-C wait blows up near 100% → act early.
2. Forecast error buffer (MAE ≈ 1 kWh).
3. Engineering safety margin — design choice, not learned.

### Worked examples
| Hour | prob | util | Result |
|---|---|---|---|
| 18 | 0.90 | 0.80 | CRITICAL, ₹29.20 |
| 14 | 0.20 | 0.60 | STABLE (ignored) |
| 1 | 0.95 | 0.97 | **UNDERUTILIZED ₹8 — danger missed** |

### Does a high price actually reduce congestion? Depends on e
Price ₹15 → ₹30 at util 0.95:
- e = −0.2 → 0.95 × 2^−0.2 = **0.83** (barely moves)
- e = −1.8 → 0.95 × 2^−1.8 = **0.27** (most drivers priced away)

### Trade-offs
| Choice | Benefit | Cost |
|---|---|---|
| Separate safety agent | Safety can't be traded for profit | Needs Mediator |
| Hard threshold | Auditable | Abrupt switch, flip-flop risk |
| Acts 1 h ahead | Pre-emptive | Only as good as forecast |
| OR of two signals | Rarely misses danger | More false alarms |
| Confidence-scaled floor | Stronger when surer | Ignores elasticity |

### ⚠️ Code reality
- **Night rule runs first → CRITICAL never fires 23:00–05:00**, exactly during the EDA's midnight surge. Fix: check CRITICAL before the night rule.
- **Not transformer data** — utilisation = chargers occupied / total. Say "charger utilisation as a proxy for grid stress."
- Night windows inconsistent: Congestion 23–5 vs cost 22–6.

### Improvement: elasticity-aware floor
```
pred_util × (p/15)^e ≤ 0.85   ⟹   p ≥ 15 × (0.85 / pred_util)^(1/e)
util 0.95: e=−1.8 → ≈ ₹16 ;  e=−0.2 → ≈ ₹26
```
Also: hysteresis (on > 0.85, off < 0.80), quantile forecasts, neighbour-zone spillover.

---

## 7. Mediator Agent

**Role:** referee and final decision-maker. Its output is the **real tariff**.

### Input
| Input | Source |
|---|---|
| `econ_price` | Economist |
| `cong_risk`, `cong_min_price` | Congestion Agent |
| `current_price` | Station's last tariff (memory; starts ₹15) |

### Output
`final_price` (₹/kWh), stored as next step's `current_price`. **Only stateful agent.**

### Logic
```python
agreed = max(econ_price, cong_min_price) if cong_risk == "CRITICAL" else econ_price
final  = clip(agreed, 0.8 × current_price, 1.2 × current_price)   # ±20%
final  = clip(final, 8, 30)                                         # hard bounds
```

### Worked examples
| Situation | Current | Econ | Risk/floor | Agreed | ±20% range | Final |
|---|---|---|---|---|---|---|
| Evening surge | 15 | 12 | CRITICAL/29.2 | 29.2 | 12–18 | **18** |
| Inelastic hour | 15 | 30 | STABLE | 30 | 12–18 | **18** |
| Night | 15 | 8 | UNDERUTIL. | 8 | 12–18 | **12** |
| Already high | 25.92 | 30 | CRITICAL/29.2 | 30 | 20.7–31.1 | **30** |

### Ramp times (5 min per step)
| Change | Steps | Time |
|---|---|---|
| ₹15 → ₹30 | 15→18→21.6→25.9→30 | ~20 min |
| ₹8 → ₹30 | 8 steps | ~40 min |
| ₹30 → ₹8 | 6 steps | ~30 min |

**Link:** forecast horizon (~60 min) > slowest ramp (~40 min) → price can reach the safety level before congestion arrives.

### 🔑 Key insight: safety is not instant
Stability clip is applied **after** safety, so the effective priority is:
```
Hard bounds  >  ±20% stability  >  safety floor  >  profit
```
> "Safety decides *where* the price should go; stability decides *how fast*. The floor is reached in ~20 min, which is fine because the forecast gives ~1 h warning. For a true emergency override, I'd let CRITICAL bypass the step limit on increases."

### Trade-offs
| Choice | Benefit | Cost |
|---|---|---|
| Explicit referee | Auditable conflict rules | Fixed priority, not optimised |
| `max(econ, floor)` | Safety never undercuts profit's price | Price is the only lever |
| ±20% per step | Smooth, removes ₹8↔₹30 flipping | Slower emergency response, lost profit |
| Multiplicative limit | Proportional fairness | Slow climb from low prices |
| Price memory | Continuity | History-dependent behaviour |

### ⚠️ Notes
- It's **rule-based arbitration**, not negotiation/consensus.
- Price can still double in 20 min and changes every 5 min; mid-session pricing not modelled.
- 20% is a design choice → test 10/20/50%/none.

### Improvement: one constrained optimisation
```
maximise    π(p)
subject to  pred_util × (p/15)^e ≤ 0.85
            |p − p_prev| ≤ 0.2 × p_prev
            8 ≤ p ≤ 30
```
Also: asymmetric limits, hourly price changes, price lock at plug-in.

---

## 8. Queueing Agent

**Role:** scorekeeper for drivers. Converts utilisation → expected wait. **Does not affect price**; evaluation only (source of −61.88%).

### Input
| Input | Source |
|---|---|
| utilisation ρ | baseline: `pred_util`; MAPC: `pred_util × (p/15)^e` |
| charger count c | `stations.csv` (`count`, zone level) |
| μ | 1 / 402.2 per min (ACN) |

### Output
Expected wait in queue **Wq (minutes)**, capped at **120**.
```
wait reduction = (Σ baseline − Σ MAPC) / Σ baseline
```

### M/M/c model
| Symbol | Meaning |
|---|---|
| M | Poisson arrivals |
| M | Exponential service times |
| c | Parallel chargers |
| λ | Arrival rate |
| μ | Service rate per charger |
| A = λ/μ | Offered load (avg busy chargers) |
| ρ = A/c | Utilisation |

Code derives load from utilisation: **A = ρ × c**.

### Maths
**Erlang-B (stable recursion, avoids factorials):**
```
B₀ = 1
Bᵢ = A·Bᵢ₋₁ / (i + A·Bᵢ₋₁),   i = 1..c
```
**Erlang-C (probability of waiting):**
```
C = B / (1 − ρ(1 − B))
```
**Expected wait:**
```
Wq = C / (cμ − λ) = C / (μ(c − A))
```
If A ≥ c or Wq > 120 → return 120.

Closed form for reference:
```
C(c, A) = [Aᶜ/c! · c/(c−A)] / [Σₖ₌₀^{c−1} Aᵏ/k! + Aᶜ/c! · c/(c−A)]
```

### Non-linearity (c = 10, service 402.2 min)
| ρ | P(wait) | Wq |
|---|---|---|
| 0.50 | 0.04 | 3 min |
| 0.70 | 0.22 | 30 min |
| 0.80 | 0.41 | 82 min |
| 0.85 | 0.53 | 142 → **capped 120** |
| 0.90 | 0.67 | 269 → **capped 120** |
| 0.95 | 0.83 | 664 → **capped 120** |

Grows like **1/(1−ρ)** near capacity — "non-linear, blows up", not literally exponential.

### Effect of pooling (ρ = 0.8, service 402.2 min)
| c | Wq |
|---|---|
| 4 | ~300 min |
| 10 | ~82 min |
| 40 | ~6 min |

### μ invariance (with correction)
```
Wq = (1/μ) × f(c, ρ)
```
1/μ cancels in the % reduction **only if the 120-min cap doesn't bind**. With 402 min, cap binds at ρ ≈ 0.70 (c=4), 0.84 (c=10), 0.95 (c=40) → the reported % partly compares capped values.

### Trade-offs
| Choice | Benefit | Cost |
|---|---|---|
| Erlang-C | Standard, fast, shows non-linearity | Strong assumptions |
| A from utilisation | Fits forecast output | Hides drivers who left |
| ACN μ | Real data | Workplace dwell ≫ Shenzhen fast charging |
| 120-min cap | No infinities | Hides overload severity, distorts % |
| Evaluation only | Simple pricing | Wait doesn't influence price |
| Zone-level c | Matches data | Pooling understates waits |

### ⚠️ Assumptions violated
- Service time bimodal (ACN histogram), not exponential.
- Fast and slow chargers are not identical.
- Drivers abandon/balk.
- Arrival rate time-varying; **steady-state formula applied every 5 min** while service is ~6.7 h → each step is a "what if this lasted forever" estimate.
- Part of the reduction = priced-away drivers (zero wait, but didn't charge).

### Improvements
Realistic service time, remove/raise cap, M/G/c (Allen–Cunneen), Erlang-A (abandonment), SimPy discrete-event simulation, per-station modelling, add wait penalty to objective:
```
maximise  π(p) − α · Wq(p)
```
Report lost demand separately.

---

## 9. Results (Simulation, Not Deployment)

| Metric | Value |
|---|---|
| Profit gain vs flat ₹15 | +10.12% |
| Off-peak uplift | +48.20% |
| Congestion events (>85%) | −66.58% |
| Wait-time reduction | −61.88% |
| Average price | ₹11.69/kWh |
| Pipeline latency | 16.96 ms (Oracle 16.49, Economist 0.24, Mediator+Queue <0.1) |

**How computed:** baseline (₹15, `pred_load`, `pred_util`) vs MAPC (final price, demand × (p/15)^e) on the same 2,000 test rows.

**Honest caveats:**
1. Results partly follow from assumptions — same elasticity sets price and scores response.
2. No real demand shifting — each step independent; priced-out demand vanishes.
3. Both scenarios use predicted load; forecast error not penalised.
4. "Off-peak" = all hours except 8, 9, 17, 18 (not just deep night).
5. Latency = one row predicted repeatedly in Colab — "lightweight", not "production-ready".

> **"Is the 10% profit real?"** — "No. It's a counterfactual simulation result vs a flat-rate baseline under the modelled elasticity and cost assumptions. Validating it would need real price variation or a controlled pilot."

---

## 10. Code vs Slides — Fix or Reframe

| Slide claim | Code reality | Say instead |
|---|---|---|
| Strict time-series CV | Station-based 80/20 split | "Evaluated on held-out stations" |
| — | `hist_station_time_util` leaks future | Compute on train period only |
| Spatial neighbour matrices | Copy of `util_lag_1` | Don't claim spatial modelling |
| 7-day MA fallback, 50 ms timeout | Not found in notebook | Implement or remove |
| Decentralised, 24,000 nodes, zero latency | Sequential functions in one loop | "Modular, lightweight per-station pipeline" |
| Proxy proves Day-1 deployment | Notebook itself says priors | "Informed prior, recalibrate locally" |
| Transformer risk | Charger occupancy | "Utilisation as proxy for grid stress" |
| Congestion handles midnight surge | Night rule skips CRITICAL | Admit + fix ordering |
| Consensus / negotiation | max + clip rules | "Rule-based arbitration" |
| Exponential wait | ~1/(1−ρ) | "Non-linear blow-up near capacity" |
| Off-peak +40% vs +48.2% | Inconsistent | Use +48.2% |
| 30,000+ sessions | Verify count | Quote notebook output |

---

## 11. Formula Cheat Sheet

```
Utilisation         ρ = active_chargers / count
Elasticity map      e_h = −2.5 + 2.3 × normalised_urgency_h
Urgency             kWhRequested / (minutesAvailable/60)
Demand              Q(p) = Q₀ (p/15)^e
Profit              π(p) = Q(p)(p − c)
Optimal price       p* = c·e/(1+e)        (e < −1)
Lerner              (p − c)/p = −1/e
Congestion floor    min(22 + 8·prob_cong, 30)
Elasticity floor    p ≥ 15 (0.85/util)^(1/e)
Mediator            clip(clip(agreed, 0.8p₀, 1.2p₀), 8, 30)
Erlang-B            Bᵢ = A·Bᵢ₋₁/(i + A·Bᵢ₋₁)
Erlang-C            C = B / (1 − ρ(1 − B))
Wait                Wq = C / (μ(c − A))
Offered load        A = λ/μ = ρ·c
R²                  1 − SS_res/SS_tot
MAE                 mean|y − ŷ|
RMSE                √mean(y − ŷ)²
```

---

## 12. Ablations & Sensitivity (if asked "how do you know each part matters?")

**Ablations:** no lags · fixed elasticity · no Congestion Agent · no Mediator · flat ₹15 · simple TOU tariff · elasticity ∈ {−0.5, −1, −2, −3}.
**Sensitivity:** elasticity, congestion threshold, price cap, step limit, service time, cost schedule, forecast error.
**Metrics:** profit, congestion events, wait, off-peak load, price volatility.

Quick cost/elasticity check:
```python
import numpy as np
for e in [-0.5, -1.3, -1.8, -2.5]:
    for c in [3, 6, 9]:
        ps = np.linspace(8, 30, 50); prof = (ps/15)**e * (ps - c)
        print(f"e={e:5} cost={c} -> best price ₹{ps[prof.argmax()]:.2f}")
```

---

## 13. Most Dangerous Questions

1. **"ACN is California — why Shenzhen?"** → Behavioural prior, not ground truth; UrbanEV lacks session data; isolated & replaceable; results conditional.
2. **"Did you learn elasticity?"** → No; proxy from urgency; no price variation available.
3. **"Is 10% profit real?"** → Simulation counterfactual; needs experiments to validate.
4. **"Why multi-agent?"** → Specialised modules with conflicting objectives + explicit rule-based arbitration; modular and auditable.
5. **"Weakest part?"** → Elasticity model. Next: estimate from real price variation, sensitivity analysis.
6. **"What role does the forecast play in pricing?"** → Drives safety + evaluation; economic optimum independent of Q₀ under constant elasticity.
7. **"Is safety instant?"** → No; ±20% ramp, ~20 min; forecast gives 1 h lead.

---

## 14. 30-Second Answers

**Project:** "I built a multi-agent dynamic tariff system for EV charging. LightGBM forecasts load, utilisation and congestion probability an hour ahead from UrbanEV Shenzhen data. An elasticity agent uses an urgency proxy from Caltech ACN sessions. An economist agent picks the profit-maximising price with a constant-elasticity model. A congestion agent sets a safety floor when overload is predicted, and a mediator resolves conflicts and limits price changes to ±20%. An Erlang-C model measures wait times. The key idea is separating learned prediction from explicit economic and safety rules. Results are simulation-based."

**Elasticity:** "Hour + CBD in, one elasticity out. Built from ACN urgency (kWh needed ÷ hours parked), median per hour, normalised, mapped to −2.5…−0.2. Night forced elastic, CBD less elastic. Behavioural proxy, not causal."

**Economist:** "Tries 50 prices ₹8–₹30, demand Q₀(p/15)^e, profit volume × margin. Gives p* = c·e/(1+e) — discounts when elastic, caps at ₹30 when inelastic. Optimum independent of Q₀."

**Congestion:** "If predicted utilisation or congestion probability > 0.85, CRITICAL with floor 22 + 8·prob up to ₹30. 85% = buffer before queue blow-up. Limits: hand-set, occupancy proxy, elasticity-blind, night ordering bug."

**Mediator:** "max(econ, floor) when CRITICAL, else econ; then ±20% of last price; then ₹8–₹30. Stability applies after safety, so the floor is reached over ~20 min — fine given the 1-h forecast."

**Queueing:** "M/M/c with μ = 1/402 min. A = ρc → Erlang-B → Erlang-C → Wq = C/(μ(c − A)), capped 120. Evaluation only. Shows non-linear blow-up near capacity. Limits: assumptions, steady state, long ACN dwell, cap."
