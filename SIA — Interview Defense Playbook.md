# SIA — Interview Defense Playbook

Nov 26, 2026 · @Divyansh verma

## 0. Hurry mode (last 10 minutes before the interview)

SIA flags support tickets whose assigned priority looks wrong. There were no labels, so it builds weak labels from keyword rules plus a resolution-time signal, distils them into DistilBERT, tunes the threshold for macro F1, and explains each flag with deterministic keyword evidence in a Streamlit app.

**Core sentence:** "There were no mismatch labels, so I generated weak labels from two signals, distilled them into a DistilBERT classifier, tuned the threshold for macro F1, and explained each flag with deterministic keyword evidence. The numbers measure agreement with those weak labels, and I know exactly where that stops being evidence."

| Metric (20% validation split, vs pseudo-labels) | Value |
| --- | --- |
| Accuracy | 0.8708 |
| Macro F1 | 0.8639 |
| Recall, Consistent | 0.8408 |
| Recall, Mismatch | 0.8895 |
| Adversarial set | 10 / 10 (all 10 are mismatch cases) |
| Tuned threshold | about 0.42 (searched 0.30 to 0.69) |
| Label fusion | 0.80 keywords + 0.20 resolution time, rounded |
| Training | DistilBERT uncased, 4 epochs, batch 16, lr 3e-5, max\_length 256 |

**Five facts you must say before the interviewer does**

1. The labels equal the keyword tier in 14 of 16 signal combinations. The model mostly learns the keyword rule.
2. The 20% split chose the epoch, chose the threshold and reported the metrics. It is a validation split, not a test set.
3. All 10 adversarial cases expect Mismatch, so "always Mismatch" also scores 10/10. The keyword rule alone scores 9/10.
4. The dataset is very likely the synthetic Kaggle set (suraj520), where priorities look randomly assigned. That makes mismatch the majority class (about 75% if priorities are uniform), not the minority.
5. Three dossier lines are fixed templates (SLA baselines, routing tier, semantic cluster), and the 0.55 weight is hard-coded.

**The weakest point, said first:** no ground truth, so there is no measurement of real mismatch detection. The fix is a 300-ticket gold set with two annotators (section 9).

**The best single example:** "No rush, but all customer records disappeared" filed as Low. No keyword matches, so the rule says Consistent. If the model flags it, it learned something beyond the list.

## 1. The problem and the framing

SIA answers one question per ticket: given what the ticket says, is its assigned priority plausible? It was built for MARS Open Projects 2026 (AI/ML problem statement 1). [Repo](https://github.com/divyansh070/mars_project_ml_sia) · [Live demo on Hugging Face Spaces](https://huggingface.co/spaces/divyansh89/SIA-Ticket-Auditor).

| Failure | Example | Cost |
| --- | --- | --- |
| Hidden Crisis (priority too low) | Production outage filed as Low | SLA breach, churn, the expensive error |
| False Alarm (priority too high) | Password reset filed as Critical | Wasted senior attention, alert fatigue |

**Why it is hard:** there is no "priority was wrong" column; "correct priority" is partly subjective; and the two errors cost different amounts.

**How to frame it in an interview:** this is a *label-scarce auditing* problem, not a classification problem with a given dataset. The real engineering is in manufacturing a training signal, keeping it leak-free, and being honest about what that signal can prove.

**What SIA is, precisely:** a learned, contextual approximation of a heuristic auditor, packaged with deterministic evidence. Calling it that (instead of "an AI that detects mispriorities") is the single framing that makes the rest of the interview easy.

## 2. End-to-end workflow

SIA has two paths: an offline pipeline that manufactures labels and a model, and an online path that scores one ticket and builds a dossier.

&#91;embedded content: SIA workflow · offline training and online scoring\]

Resolution time (highlighted) only shapes labels; at inference the model sees priority, channel, subject and description as one string. The dashboard adds a keyword override before the threshold, which the CLI does not.

**Training, step by step (`train_pipeline.py`)**

1. Load the CSV, normalise headers and priorities, drop rows missing key fields.
2. Split 80 / 20 stratified by priority, before any labelling.
3. Compute category medians and `sla_reversed` on train only; run both signals on both splits.
4. Fuse to a pseudo-severity; label = assigned priority ≠ pseudo-severity.
5. Build the text input and fine-tune DistilBERT for 4 epochs, keeping the best epoch by macro F1.
6. Grid-search the threshold 0.30 to 0.69 for macro F1 on validation; save to `threshold.json`.
7. Save `model/` and the tokenizer; run the 10-case adversarial check.

**Scoring, step by step (`predict.py`)**

1. Load the model and threshold (clamped to 0.10 to 0.95; 0.50 if the file is missing).
2. Build the same text, softmax, take P(mismatch).
3. Below threshold: return Consistent. Above: run the keyword engine for tier, delta, type and matched words, and return the JSON dossier.
4. Batch mode loops over a CSV row by row and writes all dossiers to JSON.

## 3. Data

The training script reads `customer_support_tickets.csv` (4.34 MB, committed to the repo). Its column names match the public Kaggle "Customer Support Ticket Dataset" by suraj520, which has 8,469 rows and is synthetic. [Kaggle source](https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset) · [a mirror showing the rows](https://huggingface.co/datasets/PIISA/dataset).

| Step | What the code does |
| --- | --- |
| Rename | `ticket_priority` → `priority_level`, `time_to_resolution` → `resolution_time_hours`, `ticket_type` → `issue_category` (after lower-casing and snake-casing headers) |
| Priority | Lower-cased, mapped to Low / Medium / High / Critical; anything else dropped |
| Resolution time | `pd.to_numeric(..., errors='coerce')`, then rows with NaN dropped |
| Missing values | `dropna` on description, subject, priority, channel, resolution time, category |
| Split | 80 / 20, stratified by `priority_level`, `random_state=42`, done **before** labelling |

**Gap to close before the interview: what the resolution column really is.** In the original Kaggle file, "Time to Resolution" is a timestamp and is empty for Open and Pending tickets. Coercing a timestamp to numeric gives NaN for every row. So either your CSV is a modified version with hours, or the pipeline is silently keeping far fewer rows than you think. Run this once and write the answers into section 14:

```python
import pandas as pd
df = pd.read_csv("customer_support_tickets.csv")
print(len(df), df.columns.tolist())
print(df["Time to Resolution"].head())
print(df["Ticket Priority"].value_counts(normalize=True))
print(pd.to_numeric(df["Time to Resolution"], errors="coerce").notna().sum())
```

If it is a timestamp, the correct fix is `hours = (to_datetime(resolution) - to_datetime(first_response)).dt.total_seconds() / 3600`, kept only for Closed tickets. Say it as: "I'd derive resolution hours from the two timestamps and restrict the label signal to closed tickets."

**What "synthetic" changes, and how to say it.** In this dataset, descriptions are template sentences (some still contain `{product_purchased}`), and priorities appear to be assigned independently of the text. If priority is independent of content, the chance that it equals any pseudo-severity is about 25%, so **about 75% of tickets get labelled Mismatch**. That is why Consistent recall (0.84) is the weaker class. Confirm with the printed "Fused System Mismatch Rate".

> "The dataset is synthetic and its priorities look random relative to the text, so most tickets are genuine mismatches by any reasonable rule. That made it a good sandbox for the pipeline, and a bad place to claim real-world accuracy."

## 4. Weak supervision: how the labels are made

The label is `1` when the assigned priority differs from a pseudo-severity built from two labelling functions. One reads what the ticket *says*, the other what *happened* to it.

**Signal 1: keyword tiers on the description** (lower-cased, regex with word boundaries, first match from the top wins)

| Tier | Trigger words |
| --- | --- |
| Critical | outage, breach, lawsuit, security, fraud, wiped, gone, down, fatal |
| High | urgent, asap, escalate, manager, unacceptable, broken, crash, immediately |
| Medium | delay, waiting, cancel, refund, error, issue, problem, help, stuck |
| Low | nothing matched |

**Signal 2: resolution time against the category median** (medians from the training split only; `ratio = hours / median[issue_category]`)

| Mode | Critical | High | Low | else |
| --- | --- | --- | --- | --- |
| `sla_reversed` (Critical median < Low median, so fast = urgent) | ratio < 0.35 | < 0.75 | > 1.50 | Medium |
| normal (slow = severe) | ratio > 2.0 | > 1.5 | < 0.5 | Medium |

`sla_reversed` is the detail to be proud of: the code reads from the data whether the SLA is working (urgent tickets closed faster) instead of assuming a direction.

**Fusion:** `pseudo = round(0.8 × sev1 + 0.2 × sev2)` with Low = 0 … Critical = 3. Rebuild this table from memory:

| Keyword ↓ / Resolution → | Low | Medium | High | Critical |
| --- | --- | --- | --- | --- |
| Low | Low | Low | Low | **Medium** (0.6) |
| Medium | Medium | Medium | Medium | Medium |
| High | High | High | High | High |
| Critical | **High** (2.4) | Critical | Critical | Critical |

Two things an interviewer may probe. First, the keyword tier wins in 14 of 16 cells; resolution time only acts when the signals sit at opposite extremes. Second, `np.round` uses banker's rounding, but no cell can land on x.5 here (8a + 2b would have to be odd), so rounding is unambiguous.

**How to defend the design (and own its flaw)**

- *Why two signals:* they rest on different evidence, so their errors should be roughly independent. Agreement and Cohen's kappa between them are printed as a sanity check on that.
- *Why resolution time only as a label:* it is known only after a ticket closes. As a model input it would be target leakage; as a label signal offline it is legitimate.
- *The flaw:* 0.8 / 0.2 plus rounding makes the second signal nearly decorative. Say: "I weighted keywords high because I trusted them more, but the rounding meant resolution time almost never mattered. In v2 I'd use a label model that learns each signal's accuracy (section 9)."
- *Keyword false positives:* `help`, `issue`, `problem` and `error` appear in most template descriptions, so most tickets land at Medium; `down` and `gone` fire on harmless phrasing ("slowed down", "gone through the manual"). Naming these yourself shows you looked at the data.

## 5. Model and training

DistilBERT reads one string that contains the assigned priority and the ticket text, and outputs P(mismatch). The same format is used in training, the CLI and the app:

```text
Assigned Priority: Low | Channel: Email | Subject: Database Outage | Description: URGENT: Entire production database wiped.
```

**Why the priority is in the input:** a mismatch is a relation between priority and content. Without the priority the model could only predict severity. The cost: the model must learn a 4 × 4 comparison (assigned vs inferred tier) from text, which is harder than predicting severity alone. That is one honest reason it tops out near 87%.

**Why resolution time is not in the input:** it does not exist when a ticket arrives.

| Setting | Value | One-line reason |
| --- | --- | --- |
| Model | `distilbert-base-uncased`, 2 labels | 66M params, 6 layers, runs on a free CPU Space |
| max\_length | 256 | Tickets are short; caps compute |
| Epochs | 4, best kept by `eval_f1` | Small binary task; best-epoch selection guards over-fitting |
| Batch / lr | 16 / 3e-5 | Standard BERT fine-tuning range (2e-5 to 5e-5) |
| Weight decay | 0.01 | Mild L2-style regularisation |
| Warmup | 10% of steps | Avoids large early updates to pretrained weights |
| Loss | Cross-entropy, no class weights | Macro F1 selection + threshold tuning handle the imbalance |

**Design alternatives you should be able to argue**

| Option | Why not (or when yes) |
| --- | --- |
| Keyword rule alone | Transparent, but blind to anything off the list and to negation |
| TF-IDF + logistic regression | Cheap and strong on keyword-made labels; *should* be your baseline (section 9) |
| BERT-base | About 2× slower for a small gain; not worth it on CPU hosting |
| Cased model | Would keep "URGENT" in capitals as a cue; a fair v2 experiment |
| Severity regressor + compare | Predict severity from text only, then compare to the priority in code. Simpler task for the model, and the comparison becomes exact. A strong v2 idea |

The last row is a good "what would you do differently" answer: it separates *understanding the text* (learned) from *comparing to the priority* (deterministic), which also makes the explanation faithful.

## 6. Evaluation: what the numbers prove

The model agrees with the weak labels on 87% of validation tickets. That measures how well it learned the labelling rules, not how well it finds real mispriorities.

| Metric | Achieved | Brief's target | Plain meaning |
| --- | --- | --- | --- |
| Accuracy | 0.8708 | ≥ 0.83 | 87% of validation tickets match their pseudo-label |
| Macro F1 | 0.8639 | ≥ 0.82 | Mean of per-class F1; neither class can be ignored |
| Recall, Consistent | 0.8408 | ≥ 0.78 | 84% of pseudo-consistent tickets left alone |
| Recall, Mismatch | 0.8895 | ≥ 0.78 | 89% of pseudo-mismatches caught |
| Adversarial | 10/10 | ≥ 7/10 | All 10 hand-written mismatch cases flagged |

**Put the numbers next to baselines (you should compute these; expected values shown)**

| Predictor | Expected on pseudo-labels | What it tells the interviewer |
| --- | --- | --- |
| Always "Mismatch" | accuracy ≈ mismatch rate (≈ 0.75 if priorities are random), macro F1 ≈ 0.43 | 0.87 accuracy is about +12 points over trivial; macro F1 is where the real gap shows |
| Keyword rule vs priority | Very high (it differs from the label only in the 2 extreme cells) | The label is nearly the rule, so the rule is close to a ceiling here |
| TF-IDF + LR | Likely close to DistilBERT | If equal, the labels are keyword-driven; DistilBERT's value must be shown elsewhere |
| DistilBERT | 0.87 / 0.86 | Learned the rule well but not perfectly |

The second row is the trap question: "So your model is worse than the rule that made its labels?" Answer: "On pseudo-labels, yes, by construction; the rule is close to an upper bound there. The model's job is to generalise beyond the exact words, like 'records disappeared'. That can only be measured on human labels, which is my first next step."

**Three caveats, stated plainly**

1. **Labels are mostly the keyword rule** (14 of 16 fusion cells). The 13% error is part imperfect rule-learning and part resolution-time cases the model cannot see.
2. **No untouched test set.** The 20% split chose the best epoch, chose the threshold and reported the metrics. Each choice makes the score slightly optimistic. The README's "held-out test split" should read "validation split".
3. **The adversarial set is one-sided.** All 10 expect Mismatch, so it cannot detect over-flagging. The keyword rule alone gets 9/10; only case 4 ("No rush, but all customer records disappeared", Low) needs more than the list.

**Adversarial cases checked against the rule** (so you can answer "which ones does the model actually earn?")

| # | Description (assigned) | Rule tier | Rule alone flags? |
| --- | --- | --- | --- |
| 1 | "This is NOT urgent, take your time." (Critical) | High | Yes, but for the wrong reason (matches "urgent") |
| 2 | Sarcastic "love when your servers crash" (Low) | High | Yes |
| 3 | "Database wiped." (Low) | Critical | Yes |
| 4 | "No rush, but all customer records disappeared." (Low) | Low | **No: the one real test** |
| 5 | "Refund pending for 2 weeks." (Critical) | Medium | Yes |
| 6 | "Forgot my password, please help me reset." (Critical) | Medium | Yes |
| 7 | "Minor typo on the homepage footer." (High) | Low | Yes |
| 8 | "URGENT: … system is down, lawsuit pending." (Low) | Critical | Yes |
| 9 | "Need a copy of last month's invoice." (High) | Low | Yes |
| 10 | "Your UI is very confusing." (Critical) | Low | Yes |

**The sentence to say:** "The 87% accuracy and 89% mismatch recall are against pseudo-labels on a validation split, so they show the model learned the weak-supervision signal. The real value of the neural model is generalising past exact keywords, which I can only prove with human-labelled tickets."

## 7. The evidence dossier: computed vs templated

A flag above the threshold returns `binary_judgment: "Mismatch Detected"`, a confidence string and an `evidence_dossier`. The keyword and delta parts are real computations; three evidence lines are fixed text.

| Dossier field | Status | What to say |
| --- | --- | --- |
| `inferred_severity`, `severity_delta` | Computed (keyword tier − assigned) | Deterministic and reproducible |
| `mismatch_type` | Computed: Hidden Crisis (+), False Alarm (−), Consistent (0) | Can read "Consistent" inside a "Mismatch Detected" dossier: model and rule disagree |
| `keyword_analysis.value` | Computed (matched words) | The part a support lead can verify |
| `keyword_analysis.weight` = "0.55" | Hard-coded | Does not match the 0.80 used in training; remove it |
| `resolution_time` "Evaluated against category SLA baselines" | Template | No comparison is done at inference |
| `intake_channel` "Modifies routing urgency tier" | Template | Channel only enters as text to the model |
| `semantic_cluster` "Neural Match" | Template | No clustering exists |
| `constraint_analysis` | Templated sentence from computed values | Fine, as long as it only restates computed facts |

**How to answer "is it zero-hallucination?"** "No generative model writes the explanation, so nothing is invented by an LLM. But three evidence lines were placeholder text describing checks I hadn't built. I've marked them for removal or real computation. The honest claim is deterministic keyword evidence."

**How to answer "does it explain the model?"** "No, it explains the rule. For a faithful explanation of DistilBERT I'd add token attributions with integrated gradients (Captum), and show the top tokens next to the keyword evidence. Where they disagree is exactly where a lead should look."

Small code smells worth knowing: the Consistent response uses key `confidence_score` while the mismatch response uses `confidence`; the threshold loader's comment says the default is 0.10 but the code uses 0.50.

## 8. Deployment: the CLI and the dashboard

The CLI (`predict.py`) uses the raw model score; the Streamlit dashboard (`app.py`) adds rule overrides, so the two can disagree on the same ticket.

**Dashboard features:** model cached with `@st.cache_resource`; sidebar threshold slider (0.10 to 0.95, starts at the saved value); analytics page (mismatches by channel pie, by priority bar, channel × priority heatmap); single-ticket form; batch CSV upload with column validation, progress bar, de-duplication by ticket ID; JSON export. State lives in `st.session_state`, so it is per browser and lost on restart.

**The override the dashboard adds**

| Keyword delta | Model P | Score shown |
| --- | --- | --- |
| \|delta\| ≥ 2 | any | max(P, 0.88 + 0.03 × \|delta\|), so at least 0.94 |
| 0 | P > 0.50 | min(P, 0.12) |
| otherwise | any | P |

**How to defend it without getting caught:** "The dashboard is a demo, so I added guardrails: a two-tier keyword gap always surfaces, and a ticket the rule sees as consistent is suppressed. That makes it a hybrid system, which is a legitimate pattern, but it means the 'confidence' shown is no longer a probability and the reported metrics are for the raw model only. In v2 there is one `score()` function shared by CLI and app, and I evaluate whatever runs in production."

Note what the override does to the one ticket that matters: "records disappeared" has keyword delta 0, so if the model flags it with P > 0.5, **the dashboard suppresses it to 0.12**. The override removes exactly the case where the neural model adds value. Say this yourself; it is the strongest argument for unifying the scoring.

**Hosting trade-off:** Hugging Face Spaces gives free CPU with enough memory for DistilBERT and a shareable link. Costs: row-by-row inference, no API for other systems, no persistence.

## 9. Filling the gaps: SIA v2

Five fixes take about a day and turn every weak spot into a story of "I found it and fixed it". Do tier A before interviews; talk through tier B.

| Gap | Fix | Tier | Effort |
| --- | --- | --- | --- |
| Resolution column may be a timestamp | Derive hours from timestamps; Closed tickets only | A | 30 min |
| No test set | 70 / 15 / 15 split; touch test once | A | 15 min |
| No baselines | Always-mismatch, rule, TF-IDF + LR, DistilBERT on the same test | A | 1 h |
| One-sided adversarial set | Add 10 consistent cases; report both sides | A | 30 min |
| Templated dossier lines, two scoring paths | One `score()`, delete the 3 template lines and the 0.55 | A | 1 h |
| Resolution signal nearly decorative | Label model over more labelling functions | B | half day |
| No ground truth | 300-ticket gold set, 2 annotators | B | 1 to 2 days |
| Uncalibrated scores, F1-tuned threshold | Temperature scaling + cost-based threshold | B | 1 h |
| No uncertainty on metrics | Bootstrap 95% CI | A | 15 min |

**A1. Proper split**

```python
train, rest = train_test_split(df, test_size=0.30, stratify=df.priority_level, random_state=42)
val, test   = train_test_split(rest, test_size=0.50, stratify=rest.priority_level, random_state=42)
# medians + sla_reversed from train only; epoch + threshold from val; report on test ONCE
```

**A2. Baselines on the same test split**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

def report(name, y, p):
    print(f"{name:22s} acc={accuracy_score(y,p):.3f} macroF1={f1_score(y,p,average='macro'):.3f}")

y = test.is_mismatch.values
report("always mismatch", y, [1]*len(y))
rule = (test.priority_level != test.description.map(signal_1_nlp_rules)).astype(int)
report("keyword rule", y, rule)

vec = TfidfVectorizer(ngram_range=(1,2), min_df=2)
lr  = LogisticRegression(max_iter=2000, class_weight="balanced")
lr.fit(vec.fit_transform(train.text), train.is_mismatch)
report("tfidf + LR", y, lr.predict(vec.transform(test.text)))
```

**A3. Bootstrap confidence interval** (lets you say "0.86 ± 0.02" instead of a bare number)

```python
rng = np.random.default_rng(0)
scores = [f1_score(y[i], pred[i], average="macro")
          for i in (rng.integers(0, len(y), len(y)) for _ in range(1000))]
print(np.percentile(scores, [2.5, 97.5]))
```

**A4. Balanced adversarial set.** Keep the 10 mismatch cases and add 10 consistent ones, including hard negatives: "Production database down for all users" (Critical), "Not urgent: please update my billing address" (Low), "Is the security settings page down for maintenance?" (Low, a keyword false positive). Report flags on positives and false flags on negatives separately.

**A5. One scoring function.** Move scoring into `sia/score.py`, imported by both `predict.py` and `app.py`. Drop the override, or keep it as an explicit, evaluated `hybrid` mode. Delete the SLA, routing and cluster lines and the 0.55 weight; add the model's top tokens when attributions exist.

**B1. Better labels.** Add labelling functions that can abstain: a negation-aware keyword rule ("not urgent" → Low), subject-line keywords, issue category priors, and the resolution signal. Combine them with a label model that learns each function's accuracy from their agreements, such as Snorkel's `LabelModel`, and train on its probabilistic labels. Now resolution time matters in proportion to how reliable it proves to be.

**B2. Gold set study** (the answer to "how would you know it works?")

1. Sample 300 tickets stratified by assigned priority and by model score (oversample 0.3 to 0.7, where the model is unsure).
2. Write a one-page guideline: severity definitions with 2 examples each, and a rule for ambiguous tickets.
3. Two annotators label severity independently; compute Cohen's kappa; a third resolves disagreements.
4. Hold out 200 as gold test. Report rule, TF-IDF, DistilBERT on it, plus the precision of each labelling function.
5. Use the other 100 for threshold and calibration.

**B3. Calibration and cost-based threshold.** Fit a single temperature T on validation logits (minimise NLL), check a reliability diagram and Brier score, then pick the threshold minimising `C_FN × FN + C_FP × FP`. With a missed Hidden Crisis costed at, say, 5× a False Alarm, the threshold drops and recall rises for a stated business reason, not just for F1.

**B4. Production path.** FastAPI endpoint with batched inference, ONNX or int8-quantised DistilBERT for CPU, dossiers stored in Postgres, a confirm / reject button for leads that becomes gold labels, and monitoring of daily flag rate, score drift, and the share of flags where model and rule disagree.

## 10. Defense playbook: attacks and responses

Every hard question here has the same shape of answer: **concede the fact, say why the choice was still reasonable, name the fix.** Never argue a fact the code contradicts.

| Attack | Concede | Why it was still reasonable | Fix |
| --- | --- | --- | --- |
| "Your model just learned your keyword list." | Mostly, yes: 14 of 16 fusion cells are the keyword tier. | Distilling rules into a contextual model is a known weak-supervision pattern; the model can generalise beyond exact triggers (case 4). | Label model with more signals; gold-set test on keyword-free tickets. |
| "87% of what?" | Agreement with pseudo-labels. | It verifies the pipeline learns the signal end to end, which must hold before anything else. | Gold test set. |
| "The keyword rule beats your model on your own labels." | By construction, yes. | The rule is near a ceiling on its own labels; that comparison can't show neural value. | Compare both on human labels. |
| "That's not a test set." | Correct: validation, used for 3 decisions. | Common in a time-boxed project; the bias is small but real. | 70/15/15, report test once, bootstrap CI. |
| "10/10 adversarial is meaningless." | One-sided; always-Mismatch scores 10/10, the rule 9/10. | It was a targeted recall check for known failure types. | Balanced 20+ case set. |
| "The dataset is synthetic with random priorities." | Yes; so \~75% are mismatches by any rule. | It's a sandbox to build and test the pipeline, not to claim accuracy. | Real tickets or a public real dataset + gold labels. |
| "Your explanations claim things you don't compute." | Three lines were templates. | The core evidence (keywords, delta) is deterministic and checkable. | Delete or compute them; add integrated gradients. |
| "The demo overrides the model with rules." | Yes, and it hides case-4-type flags. | Guardrails for a live demo. | One `score()`; evaluate the hybrid if kept. |
| "Why DistilBERT and not logistic regression?" | LR may match it on these labels. | DistilBERT handles order and negation and fits a free CPU host. | Report TF-IDF + LR as a baseline. |
| "Resolution time is leakage." | It would be, as a feature. | Used only offline to make labels; never in the model input. | Already handled; restrict to Closed tickets. |

**Three phrases that buy credibility**

- "That number measures agreement with my weak labels, not real-world accuracy."
- "I'd rather show you where it breaks than oversell it."
- "Here's the experiment that would settle it."

**Do not say:** "held-out test set", "calibrated threshold", "zero-hallucination", "metadata embeddings", "100% adversarial robustness", or "the model detects mispriorities with 87% accuracy".

## 11. Q&A bank

Short answers you can say in 20 to 40 seconds. Each matches what the code does.

**Problem and data**

- **Why is this hard?** No ground truth, subjective "correct priority", and asymmetric error costs.
- **Why split before labelling?** Category medians and `sla_reversed` must come from training data only, or validation statistics leak into training labels.
- **Why stratify by priority, not by label?** The label doesn't exist at split time. In v2 I'd label first using train-only statistics, then stratify on both.
- **Is the data real?** Synthetic Kaggle data with templated text and priorities that look random. Fine for building the pipeline; not for claiming accuracy.

**Weak supervision**

- **Walk me through the labels.** Keyword tier on the description, resolution ratio vs category median with direction set by `sla_reversed`, fused 0.8 / 0.2, rounded, compared with the assigned priority.
- **How much does resolution time matter?** Only in 2 of 16 cells: Low + Critical → Medium, Critical + Low → High.
- **How do you know labels are good?** I don't yet. Agreement and kappa show consistency between signals, not correctness. A gold set measures correctness.
- **Risks of weak labels?** The model inherits rule errors; correlated rules double-count; systematic bias (short tickets with no keywords) becomes "truth".
- **What is Snorkel's label model doing?** It treats each labelling function as a noisy voter, estimates each one's accuracy from how they agree and disagree, without gold labels, and outputs probabilistic labels.

**Model**

- **Why put the priority in the text?** Mismatch is a relation; without the priority the model can only predict severity.
- **How does it handle "NOT urgent"?** Attention lets "not" modify "urgent", but my adversarial case can't prove it: the rule also flags it, via the word "urgent".
- **Downside of uncased?** Capitalised "URGENT" is lost as a cue.
- **Why 4 epochs?** Small binary task; best epoch chosen by macro F1, so extra epochs are not kept if they over-fit.

**Evaluation**

- **Why macro F1?** It weights both classes equally. Here the minority class is actually *Consistent* (\~25%), so it stops the model winning by flagging everything.
- **Why 0.42 and not 0.5?** Grid search for macro F1 on validation. Lower threshold favours mismatch recall. Better: cost-based selection.
- **Is 0.42 calibrated?** No. Threshold tuning is not calibration. I'd use temperature scaling and a reliability diagram.
- **Which adversarial case proves the model adds value?** Case 4, "records disappeared": no keyword, so the rule calls it consistent.

**Explainability and serving**

- **Does the dossier explain the model?** It explains the rule. Integrated gradients would explain the model.
- **Why not SHAP or LIME?** Slower per ticket, and LIME is unstable; the rule gives an instant readable reason. The trade-off is faithfulness.
- **Why do CLI and dashboard differ?** The dashboard has keyword overrides. I'd unify into one scoring function.
- **Scale to 1M tickets?** Offline batch job, batched tokenisation, quantised or GPU model, results to a table; roughly minutes on a GPU instead of hours row by row.
- **How would you monitor it?** Flag rate per day and channel, score distribution drift, the share of flags with keyword delta 0, and lead confirm / reject rate as live precision.
- **Feedback-loop risk?** If agents re-prioritise because of SIA, future labels reflect SIA itself. Keep a random audit sample untouched by SIA.

**Reflection**

- **Weakest part?** The labels. Everything downstream inherits their quality.
- **With 5,000 human labels?** 1,000 as a true test; train on weak then fine-tune on gold; active learning on uncertain tickets; measure each labelling function against gold.
- **What did you learn?** That a pipeline can hit every target metric and still not prove its main claim; evaluation design matters as much as the model.

## 12. Concepts to know from first principles

**Weak supervision.** Labelling functions (rules, heuristics, other models) each give a noisy label or abstain; a combiner (majority vote, fixed weights as here, or a learned label model) produces training labels. Benefit: a discriminative model trained on them can generalise beyond the rules. Risk: it also learns their blind spots.

**Cohen's kappa.** Agreement corrected for chance. p\_o is observed agreement, p\_e is the agreement expected if both raters labelled at random with their own label frequencies.

```latex
\kappa = \frac{p_o - p_e}{1 - p_e}
```

Practice: two raters on 100 tickets, both say Mismatch on 40 and Consistent on 30. Rater A says Mismatch 60 times, rater B 50 times. p\_o = 0.70, p\_e = 0.6 × 0.5 + 0.4 × 0.5 = 0.50, so κ = 0.20 / 0.50 = 0.40 (moderate).

**Self-attention.** Each token's new representation is a weighted mix of all tokens' values; weights come from query-key similarity.

```latex
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
```

The square root of d\_k keeps dot products from growing with dimension and saturating the softmax.

**BERT and DistilBERT.** BERT: 12 encoder layers pretrained by masked-language modelling (predict hidden words from both sides). DistilBERT: 6 layers, trained by knowledge distillation from BERT (student matches the teacher's soft output distribution, plus MLM and a hidden-state cosine loss); about 40% smaller, 60% faster, keeping about 97% of BERT's GLUE score. WordPiece splits rare words into sub-words, so there are no unknown tokens.

**Fine-tuning for classification.** Add \[CLS\], take its final vector through a pre-classifier layer, a linear layer maps to 2 logits, softmax gives probabilities, all weights trained with cross-entropy at a small learning rate. Warmup ramps the learning rate up from 0 over the first 10% of steps, then it decays linearly.

**Two meanings of distillation.** Model distillation: DistilBERT learned from BERT. Rule distillation: your DistilBERT learns to imitate your labelling rules. SIA does the second.

**Metrics.** Precision = TP / (TP + FP); recall = TP / (TP + FN); F1 = 2PR / (P + R). Macro F1 averages per-class F1 equally; micro F1 equals accuracy in single-label classification; weighted F1 weights by class size.

**Thresholds vs calibration.** Lowering the threshold raises recall and lowers precision; a PR curve shows the trade-off. Calibration asks whether a predicted 0.8 is right 80% of the time (reliability diagram, Brier score; fixed by temperature or Platt scaling). Tuning one does not fix the other.

**Splits and leakage.** Train fits weights; validation chooses epochs, hyperparameters and thresholds; test gives one untouched estimate. Leakage types: target leakage (a feature known only after the outcome, like resolution time), contamination (statistics computed on all data before the split), temporal (training on the future).

**Explainability methods.** Rule evidence explains the rule; attention weights are not reliable explanations; LIME fits a local surrogate by perturbation (unstable); SHAP uses Shapley values (principled, slow for transformers); integrated gradients sums gradients along a path from a baseline input (faithful, needs a baseline choice).

## 13. Pitches and resume bullets

**30-second pitch**

> "SIA audits support tickets for wrong priorities: outages filed as Low, password resets filed as Critical. No labels existed, so I built them with weak supervision: keyword severity tiers plus resolution time against each category's median, with the direction learned from the data. I distilled those labels into DistilBERT, kept resolution time out of the inputs to avoid leakage, tuned the threshold toward catching hidden crises, and every flag ships with deterministic keyword evidence. It's live as a Streamlit app on Hugging Face Spaces."

**2-minute pitch** (add these beats to the 30-second version)

1. *Why it matters:* priority decides queue order; wrong priorities cause SLA breaches and wasted attention.
2. *The signals:* one from what the ticket says, one from what happened. The resolution signal checks whether fast resolution means urgent (`sla_reversed`).
3. *The result:* 0.87 accuracy, 0.86 macro F1, 0.89 mismatch recall on a validation split, against pseudo-labels.
4. *The honesty beat:* "Those labels are mostly keyword-driven, so the numbers show the model learned the signal. Its real value is generalising beyond the keyword list, which needs human labels to prove."
5. *What's next:* gold set, proper test split with baselines, a label model so resolution time actually matters.

**Resume bullets (safe versions)**

- Built a weak-supervision pipeline to detect support-ticket priority mismatches without ground-truth labels, fusing keyword severity rules with a category-normalised resolution-time signal; kept resolution time out of model inputs to prevent target leakage.
- Fine-tuned DistilBERT on the pseudo-labels and tuned the decision threshold for macro F1 (0.87 accuracy, 0.89 mismatch recall on validation); deployed a Streamlit auditing dashboard on Hugging Face Spaces with per-flag JSON evidence.

If you finish tier A of section 9, upgrade the second bullet to: "…benchmarked against rule and TF-IDF baselines on a held-out test split (macro F1 X ± Y)". Only write that once the numbers exist.

## 14. Pre-interview checklist

**Fill in from one training run** (the script already prints these)

| Value | Yours |
| --- | --- |
| Rows after cleaning (train / validation) |  |
| Format of "Time to Resolution" in your CSV |  |
| Fused mismatch rate |  |
| Signal agreement and Cohen's kappa |  |
| Exact tuned threshold |  |
| Training time and hardware |  |
| Did case 4 ("records disappeared") get flagged, and at what P? |  |

**Fix in the repo** (an interviewer may open it)

- [ ] README: "held-out test split" → "validation split"; "text and metadata" → "priority, channel, subject and description as one text"
- [ ] README: add a short Limitations section (labels keyword-dominant, no gold set, one-sided adversarial set)
- [ ] Remove the three template evidence lines and the hard-coded 0.55
- [ ] Make `predict.py` and `app.py` share one scoring function
- [ ] Add `.DS_Store` to `.gitignore`; move CSVs to Git LFS or a download link
- [ ] Run tier A from section 9 and add a baselines table to the README

**Practise out loud**

- [ ] 30-second and 2-minute pitches, timed
- [ ] Rebuild the 4 × 4 fusion table without looking
- [ ] Draw the training and inference flows (section 2)
- [ ] Compute kappa for a 2 × 2 table
- [ ] Answer every row of section 10 with "concede, justify, fix"
- [ ] Explain why tuning the threshold on the reporting split is optimistic

**Sources:** [Repository](https://github.com/divyansh070/mars_project_ml_sia) · [train\_pipeline.py](https://github.com/divyansh070/mars_project_ml_sia/blob/main/train_pipeline.py) · [predict.py](https://github.com/divyansh070/mars_project_ml_sia/blob/main/predict.py) · [Kaggle dataset (suraj520)](https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset) · [Hugging Face mirror of the dataset](https://huggingface.co/datasets/PIISA/dataset)
