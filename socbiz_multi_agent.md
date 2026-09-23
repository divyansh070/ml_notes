# Multi-Agent Predictive Control (MAPC) — EV Charging Dynamic Tariff Optimization

### Interview Prep Dossier

**Project**: Multi-Agent Predictive Control (MAPC) for EV Charging Networks  
**Core idea**: Forecast EV charging demand → estimate behavioral response → optimize tariff → enforce grid-safety constraints → estimate queueing impact.

---

## 1. What I Built

Problem

EV charging stations face two competing problems:

1. Peak congestion — too many vehicles charging simultaneously can push transformer/station utilization toward capacity.
2. Under-utilization — some periods, especially overnight, have significant unused charging capacity.

A flat electricity tariff does not account for either condition.

I built a multi-agent dynamic tariff system that uses demand forecasting, behavioral proxies, economic optimization, grid constraints, and queueing theory to determine a station-level charging price.

The important architectural idea is:

```text
Historical / Real-Time Data
            │
            ▼
     Forecasting Agent
            │
            ▼
    Predicted Demand
            │
      ┌─────┴─────┐
      ▼           ▼
 Elasticity    Congestion
   Agent          Agent
      │           │
      ▼           ▼
 Demand Response  Grid Risk
      │           │
      └─────┬─────┘
            ▼
      Economist Agent
            │
       Optimal Price
            │
            ▼
       Mediator Agent
            │
    Safety + Stability
            │
            ▼
       Final Tariff
            │
            ▼
       Queueing Agent
            │
            ▼
      Expected Wait Time
```

---

## 2. Data

Macro-grid data — UrbanEV / Shenzhen

Used to model the physical charging-network side.

Important information:

* ~24,798 charging piles
* 5-minute intervals
* station utilization
* charging/load information
* temporal information
* station capacity
* spatial/location information

The project uses this dataset as the primary grid-side dataset.

Micro-behavior data — ACN / Caltech-JPL

Used to construct behavioral proxies.

Important information:

* 30,000+ charging sessions
* parking/occupancy duration
* charging-session behavior
* charging urgency / willingness-to-pay proxies

The project uses ACN because the Shenzhen dataset does not contain equivalent granular behavioral information.

Important limitation

The ACN data is from a different geographic environment.

Therefore:

I should describe it as a behavioral calibration/proxy dataset, not as ground-truth Shenzhen consumer behavior.

The project’s assumption is that basic charging/parking and price-response patterns can provide a transferable behavioral prior.

---

## 3. Preprocessing & Feature Engineering

The project creates features from several dimensions.

Temporal features

hour_of_day
day_of_week

These capture recurring demand cycles.

Example:

00:00–02:00 → large overnight charging surge
08:00–09:00 → high urgency

Lag features

load_lag_1
util_lag_1

These provide short-term temporal context.

Conceptually:

Current demand ≈ f(previous demand, time, station, capacity, ...)

Station-level historical features

Examples include historical station/hour utilization.

These attempt to capture persistent differences between stations.

Spatial/context features

The project also uses CBD/location information to capture differences in demand behavior across station locations.

---

## 4. Exploratory Data Analysis

Finding 1 — Midnight charging surge

UrbanEV shows substantial utilization between approximately midnight and 02:00.

Interpretation:

* commercial/fleet charging contributes to overnight demand
* daytime capacity is comparatively underutilized

This motivates the Economist Agent’s attempt to shift flexible demand into cheaper/off-peak periods.

Finding 2 — Long parking duration

The ACN behavioral data shows a mean charger occupancy duration of approximately:

402.2 minutes

This suggests that congestion is not purely determined by energy delivery; vehicles can occupy physical charging infrastructure for long periods.

This becomes an input to the queueing model.

Finding 3 — Morning urgency

Behavioral proxies indicate stronger charging urgency around approximately 08:00–09:00.

The system therefore models morning consumers as less price-sensitive and avoids relying on aggressive surge pricing during that period.

---

## 5. Forecasting Agent

The Forecasting Agent is the ML component that predicts future charging demand.

Model

LightGBM

It is used for:

* short-term kWh load prediction
* station utilization prediction
* downstream congestion prediction

Why LightGBM?

Things I should be able to explain:

* gradient-boosted decision trees
* why trees work well on heterogeneous tabular data
* nonlinear feature interactions
* relatively low inference cost
* good performance without requiring huge neural networks
* compatibility with engineered temporal/station features

Important features

The strongest features include:

hour_of_day
load_lag_1
util_lag_1
station capacity
historical utilization
day_of_week
location/CBD information

Reported forecasting performance

R²   = 0.7167
RMSE = 1.3691 kWh
MAE  = 0.9806 kWh

These are the reported test metrics for the forecasting model.

---

## 6. Important ML Evaluation Issue

The project uses a chronological train/test approach to reduce future-information leakage.

The correct interview explanation should be:

“Because this is a forecasting problem, randomly shuffling observations would allow future patterns to enter training. I therefore used chronological evaluation.”

I should NOT casually claim:

“I used strict time-series cross-validation.”

Unless I actually implement rolling/blocked cross-validation.

---

## 7. Elasticity Agent

The Elasticity Agent estimates how strongly charging demand should respond to price.

The project does not directly observe a clean:

price → demand

relationship in the available data.

Instead, it creates behavioral proxies using:

* charging-session characteristics
* parking duration
* urgency
* hour of day
* location/CBD

The resulting elasticity is then bounded and adjusted based on contextual factors.

Core idea

Morning:

High urgency
→ relatively inelastic
→ less aggressive price response

Deep night:

Lower urgency / flexible demand
→ more elastic
→ stronger discounting can shift demand

Important terminology

I should call this:

behaviorally calibrated elasticity proxy

rather than claiming that the ML model directly learned consumer price elasticity.

---

## 8. Economist Agent

The Economist Agent converts predicted demand and estimated elasticity into a pricing decision.

It evaluates 50 candidate prices.

The project uses a constant-elasticity demand model:

Q(p) = Q₀ × (p / p₀)^e

where:

Q₀ = baseline predicted demand
p₀ = reference price
p  = candidate price
e  = price elasticity

The agent evaluates the candidate tariffs and chooses the price producing the highest simulated economic objective.

Core question

The Forecasting Agent answers:

“How much demand do I expect?”

The Economist answers:

“Given that demand and my behavioral assumptions, what price should I charge?”

This separation is one of the most important ideas in the project.

---

## 9. Congestion Agent

The Congestion Agent represents the grid-safety objective.

It checks predicted station/grid utilization.

The project uses an 85% utilization threshold.

Conceptually:

if predicted congestion > threshold:
       CRITICAL
else:
       NORMAL

Under critical conditions, it imposes a safety price floor/constraint.

The emergency ceiling/floor logic can reach:

₹30/kWh

The purpose is not profit maximization.

It is:

prevent the economic optimizer from choosing a tariff that would worsen an already dangerous congestion condition.

---

## 10. Mediator Agent

This is the system’s conflict-resolution layer.

There are competing objectives:

```text
Economist
    ↓
maximize economic return
Congestion Agent
    ↓
protect grid capacity
```

The Mediator determines which constraint takes precedence.

Critical grid condition

```text
Congestion Agent
       ↓
CRITICAL
       ↓
Override economic recommendation
       ↓
Apply safety constraint
```

Price stability

The final tariff is also constrained by a:

±20% maximum step change

This prevents sudden tariff shocks.

So the Mediator provides:

Safety constraint
+
Price stability constraint
+
Agent coordination

---

## 11. Queueing Agent

The Queueing Agent translates utilization into expected waiting-time effects.

The project uses:

Erlang-C queueing theory

The relevant conceptual quantities are:

λ = arrival rate
μ = service rate per charger
c = number of chargers
ρ = utilization

The model estimates the probability of waiting and expected queueing delay.

Why Erlang-C?

Because charging stations resemble a multi-server queue:

```text
EVs arriving
      ↓
Waiting line
      ↓
Multiple chargers
      ↓
Service completion
```

Unlike a single-server queue, multiple charging points operate simultaneously.

---

## 12. Why Queueing Theory Matters

The project assumes that as utilization approaches capacity, waiting time can increase nonlinearly.

Therefore:

```text
Small reduction in peak demand
              ↓
Lower utilization
              ↓
Much lower probability of queueing
              ↓
Potentially large waiting-time reduction
```

This gives the tariff optimizer a second objective beyond revenue.

---

## 13. Autonomous Negotiation

The system’s central conflict is demonstrated during a high-demand period.

Example:

```text
Demand increases
      ↓
Forecasting Agent predicts high utilization
      ↓
Congestion Agent detects critical risk
      ↓
Economist wants a profitable price
      ↓
Congestion Agent imposes safety constraint
      ↓
Mediator overrides the economic recommendation
      ↓
Final tariff increases within ±20% step limit
```

This is the core multi-agent control loop.

---

## 14. Simulation Results

Relative to the project’s flat-rate baseline, the submission reports:

Profit gain             +10.12%
Off-peak uplift         +48.20%
Congestion reduction    -66.58%
Wait-time reduction     -61.88%
Average price           ₹11.69/kWh

The project also reports approximately:

16.96 ms

for the local computational pipeline benchmark.

VERY IMPORTANT

These are simulation/counterfactual results.

They are not measurements from a deployed EV charging network.

The economic and behavioral results depend on:

* elasticity assumptions
* demand-response model
* queueing assumptions
* simulated pricing environment

So if asked:

“Did you actually increase profit by 10.12%?”

The correct answer is:

“No real-world deployment was performed. The 10.12% is the improvement produced by the simulation relative to the project’s flat-rate baseline under the modeled elasticity assumptions.”

---

## 15. Failure Handling

One particularly useful engineering component is the forecasting fallback.

If the forecasting model fails or exceeds the latency threshold:

```text
Oracle timeout / failure
          ↓
7-day historical moving average
          ↓
Continue tariff generation
```

The submission uses a 50 ms timeout for the Oracle fallback.

This prevents a temporary ML failure from stopping the entire pricing system.

---

## 16. What I Should Be Able to Explain From First Principles

ML

* LightGBM
* decision trees
* gradient boosting
* regression
* R²
* MAE
* RMSE
* feature importance
* temporal forecasting
* data leakage
* chronological train/test splitting
* time-series cross-validation

Economics

* price elasticity
* elastic vs inelastic demand
* constant-elasticity demand
* revenue vs profit
* demand curves
* price optimization
* constrained optimization

Queueing Theory

* M/M/c queues
* Erlang-C
* arrival rate λ
* service rate μ
* number of servers c
* utilization ρ
* waiting probability
* expected waiting time
* why queueing becomes unstable near capacity

EV/Grid

* EV charging demand
* charging station utilization
* transformer capacity
* peak demand
* load shifting
* time-of-use pricing
* demand response

Multi-Agent Systems

* what an agent is
* agent specialization
* coordination
* conflict resolution
* centralized vs decentralized control
* constraints
* fallback mechanisms
* local vs global optimization

---

## 17. Interview Questions — Basic

Project Understanding

Q1. Explain your project in 30 seconds.

Q2. What problem were you trying to solve?

Q3. Why dynamic pricing?

Q4. Why did you need multiple agents?

Q5. What exactly is the role of each agent?

Q6. What was your input and what was your final output?

Q7. What datasets did you use?

Q8. Why did you need two datasets?

---

## 18. Interview Questions — Data

Q9. Why use UrbanEV?

Q10. Why use ACN?

Q11. Why can ACN behavioral information be transferred to Shenzhen?

Q12. What assumptions are you making when transferring behavioral patterns across locations?

Q13. What preprocessing did you perform?

Q14. How did you deal with missing values?

Q15. How did you handle timestamps?

Q16. Why did you create lag features?

Q17. What does load_lag_1 mean?

Q18. What does util_lag_1 mean?

Q19. Why is hour-of-day useful?

Q20. Why is day-of-week useful?

Q21. What is the difference between utilization and load?

---

## 19. Interview Questions — LightGBM

Q22. Why LightGBM?

Q23. How does LightGBM work?

Q24. How is boosting different from bagging?

Q25. What is a weak learner?

Q26. What is gradient boosting?

Q27. Why might LightGBM outperform linear regression here?

Q28. Why not use an LSTM?

Q29. Why not use a Transformer?

Q30. What hyperparameters matter in LightGBM?

Study:

n_estimators
learning_rate
max_depth
num_leaves
min_child_samples
subsample
colsample_bytree
regularization

Q31. What is overfitting in gradient boosting?

Q32. How would you reduce it?

---

## 20. Interview Questions — Forecasting

Q33. Why can’t you randomly shuffle a forecasting dataset?

Q34. What is lookahead bias?

Q35. What is temporal leakage?

Q36. How did you split train/test?

Q37. Why is chronological evaluation important?

Q38. What would proper rolling-origin validation look like?

Q39. Why is R² = 0.7167 not necessarily “71.67% accurate”?

Q40. Explain MAE.

Q41. Explain RMSE.

Q42. Why can RMSE be larger than MAE?

Q43. Which metric would you choose for this problem and why?

Q44. What would you do if demand suddenly changed because of an event not present in training?

---

## 21. Interview Questions — Feature Engineering

Q45. Why are lag features useful?

Q46. What happens if you create a lag incorrectly?

Q47. Can historical averages cause leakage?

Q48. How would you construct a leakage-free historical station feature?

Q49. How would you add actual spatial information?

Q50. How could nearby stations influence each other?

Possible approach:

```text
station coordinates
       ↓
nearest neighbors
       ↓
neighbor utilization
       ↓
spatial lag features
```

---

## 22. Interview Questions — Elasticity

Q51. What is price elasticity of demand?

Q52. What does elasticity = -2 mean?

Q53. What’s the difference between elastic and inelastic demand?

Q54. How did you estimate elasticity?

Q55. Did you actually learn elasticity from observed prices?

Q56. What is the biggest limitation of your elasticity model?

Q57. Why is morning demand modeled as less elastic?

Q58. Why might nighttime demand be more elastic?

Q59. What data would you need to estimate elasticity properly?

Q60. How would you estimate elasticity using causal inference?

Study:

price variation
↓
demand response
↓
causal estimation

Potential methods:

* randomized pricing experiments
* A/B tests
* instrumental variables
* difference-in-differences
* panel regression

---

## 23. Interview Questions — Economic Optimization

Q61. Explain your demand equation.

Q62. Why use constant elasticity?

Q63. What happens when elasticity becomes more negative?

Q64. Why search over 50 candidate prices?

Q65. Why not use gradient-based optimization?

Q66. What is the objective function?

Q67. What happens if maximizing profit conflicts with grid safety?

Q68. What constraints does your optimizer have?

Q69. How would you formulate this as constrained optimization?

Conceptually:

maximize       Profit(p)
subject to:
               utilization(p) <= capacity
               price_change <= 20%
               price >= safety floor
               price <= safety ceiling

---

## 24. Interview Questions — Multi-Agent Architecture

Q70. What makes this a multi-agent system?

Q71. Why not put everything into one ML model?

Q72. What is the benefit of separating agents?

Q73. What happens if two agents disagree?

Q74. Why is the Mediator necessary?

Q75. Is this truly decentralized?

Q76. What information does each agent need?

Q77. Could the system scale to thousands of stations?

Q78. What would centralized control look like?

Q79. What would decentralized control look like?

Q80. What happens if communication between agents fails?

---

## 25. Interview Questions — Queueing Theory

Q81. What is an M/M/c queue?

Q82. What does each M/M/c assumption mean?

Q83. What is Erlang-C?

Q84. What is λ?

Q85. What is μ?

Q86. What is c?

Q87. What is utilization ρ?

Q88. Why does waiting time rise sharply near full utilization?

Q89. Why is Erlang-C appropriate for charging stations?

Q90. What assumptions of Erlang-C may not hold for EV charging?

This last question is particularly important.

EV charging has:

* variable service times
* different charging rates
* different vehicle battery states
* parking behavior
* drivers who may leave before charging completes

Therefore M/M/c is an approximation.

---

## 26. Interview Questions — Results

Q91. How did you calculate the 10.12% profit gain?

Q92. How did you calculate the 48.2% off-peak uplift?

Q93. How did you calculate the 61.88% wait-time reduction?

Q94. What was your baseline?

Q95. What happens if the elasticity assumption changes?

Q96. Did you compare against other pricing policies?

Q97. What would happen if consumers did not respond to price?

Q98. Could your system actually increase congestion?

Q99. How would you perform a sensitivity analysis?

Q100. What experiment would convince you that the result is real?

---

## 27. The Most Dangerous Questions

These are the questions I should prepare especially well.

1. “Your ACN data is from California. Why should it represent Shenzhen?”

Answer structure:

It doesn't represent Shenzhen ground truth.
I used it as a behavioral proxy because UrbanEV lacks
the equivalent session-level behavioral variables.
Therefore the elasticity component is a modeling assumption,
and the resulting economic gains should be interpreted as
simulation results rather than real-world estimates.
A real deployment would recalibrate elasticity using
local price-demand observations.

---

2. “Did you actually learn price elasticity?”

Answer:

Not directly. The available data did not provide a clean randomized or observational price-demand relationship. I therefore constructed a behavioral elasticity proxy from charging-session characteristics and temporal context, then used it inside a constant-elasticity demand model.

---

3. “Is the 10.12% profit increase real?”

Answer:

No. It is a counterfactual simulation result relative to the flat-rate baseline. It depends on the assumed elasticity and demand-response model. I would need real pricing interventions or historical price variation to validate the economic improvement.

---

4. “Why call this multi-agent?”

Answer:

I separated the system into specialized decision modules with different objectives: forecasting, behavioral modeling, economic optimization, grid protection, conflict mediation and queueing analysis. The important part is that these modules can produce conflicting decisions, and the Mediator imposes explicit priority and safety constraints.

---

5. “What’s the weakest part of your project?”

Best answer:

The behavioral elasticity model. Because I don’t have local causal price-response data, the elasticity is proxy-based. My next step would be to estimate elasticity from actual price variation or a controlled pricing experiment, then evaluate the policy with out-of-sample counterfactual testing.

That answer shows research maturity.

---

## 28. Topics I Should Study

Priority 1 — MUST KNOW

Machine Learning

* LightGBM
* Gradient Boosting
* Decision Trees
* Regression
* Feature importance
* Overfitting
* Regularization
* MAE
* RMSE
* R²
* Train/test leakage

Time-Series

* chronological splitting
* lag features
* rolling features
* temporal leakage
* lookahead bias
* walk-forward validation
* TimeSeriesSplit
* seasonality
* trend
* forecasting horizon

Economics

* price elasticity
* demand curves
* revenue
* profit
* constant elasticity
* demand response
* constrained optimization

Queueing

* M/M/1
* M/M/c
* Erlang-C
* λ
* μ
* utilization
* waiting time
* queue stability

---

## 29. Priority 2 — VERY USEFUL

Optimization

Study:

* constrained optimization
* grid search
* objective functions
* constraints
* feasible regions
* Lagrangian intuition
* sensitivity analysis
* Pareto trade-offs

Multi-Agent Systems

Study:

* agent architecture
* coordination
* decentralized optimization
* consensus
* conflict resolution
* centralized vs decentralized systems
* fault tolerance

Energy Systems

Study:

* EV charging
* transformer capacity
* peak shaving
* load shifting
* time-of-use pricing
* demand response
* smart charging
* vehicle-to-grid

---

## 30. Priority 3 — If I Have Extra Time

Causal Inference

Especially useful for defending elasticity.

Study:

* randomized experiments
* A/B testing
* causal vs correlational relationships
* confounding
* instrumental variables
* difference-in-differences
* treatment effects

Advanced Forecasting

* ARIMA
* SARIMA
* Prophet
* XGBoost/LightGBM forecasting
* LSTM
* Temporal Fusion Transformer
* probabilistic forecasting

You don’t need to implement these.

You should know when they would and wouldn’t be appropriate.

---

## 31. Ablations I Should Know How to Design

An interviewer may ask:

“How do you know each component actually matters?”

I should be able to propose:

Ablation 1 — No lag features

Compare:

Full model
vs
No load/utilization lags

Ablation 2 — No behavioral elasticity

Use fixed elasticity.

Ablation 3 — No congestion agent

Let the Economist operate without grid constraints.

Ablation 4 — No Mediator

Allow the Economist’s price to pass directly.

Ablation 5 — Flat pricing

Use fixed ₹15/kWh.

Ablation 6 — Time-of-use pricing

Compare against a simple predefined peak/off-peak tariff.

Ablation 7 — Different elasticity assumptions

For example:

e = -0.5
e = -1
e = -2
e = -3

Then measure:

profit
congestion
wait time
off-peak demand

This would make the simulation much more convincing.

---

## 32. Sensitivity Analysis

This is another topic I should be prepared to discuss.

Ask:

What happens if your assumptions are wrong?

Vary:

elasticity
congestion threshold
price cap
price step limit
service time
station capacity
forecast error

Then examine how the final policy changes.

This is particularly important because the project contains several assumptions.

---

## 33. What I Would Improve Next

If I had another iteration, my roadmap would be:

Step 1

Fix/strengthen temporal evaluation.

Train → Validation → Test

with genuinely time-based splits.

Step 2

Remove any historical feature leakage.

Step 3

Build actual spatial neighbor features.

Step 4

Estimate elasticity from real price-demand observations.

Step 5

Perform elasticity sensitivity analysis.

Step 6

Compare against stronger baselines:

Flat pricing
TOU pricing
Simple congestion pricing
MAPC

Step 7

Perform agent ablations.

Step 8

Use rolling/walk-forward forecasting evaluation.

Step 9

Validate queueing assumptions.

Step 10

Eventually test the controller in a realistic simulator or controlled pilot.

---

34. 30-Second Interview Explanation

“I built a multi-agent dynamic tariff system for EV charging networks. I used UrbanEV charging data to forecast short-term station load and utilization with LightGBM, and ACN session data to construct behavioral proxies for charging urgency and price sensitivity. A forecasting agent supplies demand estimates, an elasticity agent models behavioral response, and an economist agent searches candidate tariffs using a constant-elasticity demand model. A congestion agent imposes grid-safety constraints, while a mediator resolves conflicts and limits tariff changes. Finally, an Erlang-C queueing model estimates how the resulting demand shift affects waiting time. The key idea was separating learned prediction from explicit economic and safety constraints.”

---

35. 2-Minute Explanation

“The problem I was trying to solve was that EV charging networks can simultaneously have peak congestion and under-utilized capacity. A flat tariff doesn’t respond to either condition.

I therefore designed a multi-agent system. The first component is a LightGBM forecasting model that predicts short-term load and utilization using temporal, lagged, station-capacity and location features. Because it’s a forecasting problem, I use chronological evaluation rather than random shuffling.

The next challenge is that the Shenzhen dataset doesn’t contain the behavioral information needed to directly estimate price elasticity. I therefore use ACN session data as a behavioral proxy and construct an elasticity model based on charging urgency, occupancy and temporal behavior. This is a modeling assumption rather than a causal estimate of Shenzhen elasticity.

The Economist Agent takes the forecast and elasticity estimate and evaluates 50 candidate prices using a constant-elasticity demand model. The Congestion Agent independently checks whether predicted utilization approaches the grid-safety threshold. If the grid is critical, its safety constraint overrides the economic recommendation. The Mediator also limits price changes to 20% per step to avoid volatility.

Finally, the Queueing Agent uses an Erlang-C approximation to translate utilization changes into expected waiting-time changes.

The reported improvements—such as roughly 10% simulated profit improvement and 62% simulated wait-time reduction—are counterfactual simulation results under the project’s behavioral and queueing assumptions, not real-world deployment measurements.”

---

## 36. One-Line Mental Model

When revising before an interview, remember:

PREDICT → MODEL BEHAVIOR → OPTIMIZE → CONSTRAIN → MEASURE

Specifically:

```text
LightGBM
   ↓
Demand forecast
   ↓
Elasticity proxy
   ↓
Economic optimization
   ↓
Grid constraints + mediator
   ↓
Erlang-C
   ↓
Profit / congestion / waiting-time trade-off
```

---

## 37. The Five Things I Must Know Cold

If I only have limited preparation time, I should be able to explain these without looking at the notebook:

1. LightGBM

How it works, why I chose it, and how I evaluated it.

2. Time-series leakage

Why random splitting is dangerous and exactly how my chronological evaluation works.

3. Price elasticity

What elasticity means and exactly how my project approximates it.

4. Erlang-C

What M/M/c means, what λ/μ/c represent, and why waiting time explodes near capacity.

5. Simulation validity

Why the reported economic improvements are model-based counterfactual results, and what evidence would be needed to validate them in the real world.

Those five areas cover most of the technical depth of this project.
