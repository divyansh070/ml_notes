# HiLabs AI Engineer Interview — Internship Prep (SBM Traders)

Nov 1, 2026 · @Divyansh verma

The interviewer only sees your three resume bullets, so every section here is a conversation you should be able to hold without notes: the question, your answer, and where they push next.

## 0. Hurry mode

Your internship in one breath: SBM Traders sold goods on credit to 100+ dealers and decided who gets credit from memory; you built a Django + PostgreSQL platform that turned their messy Zoho/Tally accounting exports into one clean payments database, scored every dealer with two credit scores, and showed the owners who owes what and how old the debt is — and with that, default-related losses fell by 50%.

### The 60-second story (say this when they say "tell me about your internship")

> "SBM Traders is a B2B trading business that sells to over 100 dealers on credit — dealers take goods now and pay later. Credit decisions were made from memory and scattered Excel reports, so the same dealers kept getting credit even when they paid late.
>
> I built an internal web platform for them. Staff upload their accounting exports from Zoho or Tally, and the system cleans them, fixes reporting quirks, and stores every invoice and payment in one database. From that history I designed a CIBIL-style credit score for every dealer — actually two: one for recent behaviour and one for long-term reliability — plus a simulator so the owners could test policy changes before applying them. On top sits a dashboard with outstanding balances and debt-ageing buckets.
>
> The owners started using the scores to cut or tighten credit for risky dealers, and default-related losses dropped by about half. The hardest part wasn't the scoring — it was making messy accounting data trustworthy, which is very close to what HiLabs does."

### Each bullet decoded — what it really means

| Resume words | What you actually built | One-line defence |
| --- | --- | --- |
| "real-time CIBIL-style credit scoring simulator" | Two scores (V1 short-term, V2 lifetime) that recompute on every data upload, plus a what-if tool where changing a policy setting instantly shows the new score | "Real-time means the scores are always current — every upload recomputes them; nobody waits for a monthly report." |
| "historical repayment records and outstanding balances" | Zoho "Payments Received" / invoice exports (history) + AR-aging reports (what's still owed) | "Paid history tells me how a dealer behaves; outstanding balances tell me current exposure." |
| "100+ dealer accounts" | Every dealer in SBM's books | Know the exact number. |
| "internal web platform replacing manual spreadsheet tracking" | Upload → clean → database → scores → dashboard; Zoho/Tally stay the accounting system | "Accounting stayed in Zoho; I replaced the Excel reconciliation and analysis around it." |
| "adopted by dealers and staff" | Owners, accountants and sales staff used it; dealers felt it through credit terms and statements | Be clear on how dealers touched it (section 4). |
| "cut default-related losses by 50%" | Business outcome after owners used the scores for credit decisions | Have definition, period and mechanism ready (section 4). |
| "operational dashboard… debt-ageing KPIs" | Outstanding per dealer, delays, pending vs paid, ageing buckets | Know every KPI's definition (section 5). |

### Three stories that carry the whole interview

1. **The ₹3 lakh that didn't exist** — Zoho repeated one payment's full amount on every invoice it covered; totals were tripled until you fixed it.
2. **Two scores, not one** — a dealer can be big and loyal but slipping right now; one number hides that.
3. **Breaking your own model** — you invented 10 extreme dealer profiles, found two cases where the score was unfair, and fixed the logic.

### Where they'll grill hardest — read these three twice

| Topic | Section | The one idea to carry in |
| --- | --- | --- |
| The 50% loss cut | 4A | Definition → rate, not rupees → mechanism → honest limits |
| 100 concurrent users | 8A | Users → requests per second; slow requests and the database are the limits; heavy work goes to a queue |
| Models at scale | 8B | Precompute in batch, recompute only what changed, load once, vectorise, monitor drift |

For anything else, use the six-step pattern in section 10A.

## 1. What you actually did — the internship rebuilt

This is your memory refresh, taken from your own repo and notes: seven phases, in the order the work naturally happened. Tell it in this order and every follow-up has a home.

&#91;embedded content: the SBM system end to end · 7 steps\]

Everything up to the dashboard is what you built; the last two steps are how the owners used it. Phases 2–4 below are the top row's first three boxes.

### Phase 1 — Understanding the problem

- SBM sells goods to dealers on credit (e.g. "pay within 30 days"). Money owed but not yet paid = **accounts receivable**.
- Some dealers paid on time, some weeks late, some forgot small invoices, some stopped ordering.
- Questions like "should we give this dealer ₹5 lakh more credit?" were answered from the owner's memory.
- Data lived in Zoho Books / Tally. Staff exported reports to Excel and reconciled by hand.

**What you did:** sat with owners/accountants, learned which reports they used (Invoices, Payments Received, AR Aging), and what decision they wanted to make.

### Phase 2 — Getting the data in

- Built an upload page: an Excel file or a Google Sheets link.
- Zoho exports put a title row above the real headers — the system detects that and fixes it.
- Different reports call the same thing different names ("Payment Date", "Receipt Date", "Date"). You wrote a mapping layer that normalises names and maps them to one standard schema.

### Phase 3 — Making the data correct (your hardest work)

- **Split payments:** one ₹1 lakh payment covering three invoices showed ₹1 lakh on each of three rows → totals tripled. Fixed by using the "amount applied to this invoice" per row.
- **Duplicate IDs:** after the fix, one payment ID appeared on several rows. Fixed with a combined ID: payment ID + invoice ID.
- **Files with no ID:** built a fallback key from dealer + date + amount.
- **AR-aging meaning:** every row in an ageing report is money *still owed*, even if it shows a "last payment date". Marked them unpaid so their delay keeps growing each day.
- **Re-uploads:** accountants re-export constantly. You made imports **idempotent** — uploading the same file twice updates rows instead of duplicating them.

### Phase 4 — Turning payments into signals

For every invoice: days taken to pay; delay beyond the credit terms; delay counted only when late (early payment isn't "negative lateness"); status — Paid, Pending (unpaid), or Advance (paid before an invoice existed); and each dealer's last order date.

### Phase 5 — Designing the credit scores

- **V1 "Dynamic" (short-term pulse):** starts at 1000, loses points for how late the dealer's last few late payments were (bigger invoices count more) and for days without ordering. Range 100–1000.
- **V2 "Standard" (lifetime):** starts at 300, adds points by delay tier (Gold if typically ≤ 4 days late, Average ≤ 15, Poor beyond), adds a boost for total sales on a log scale, subtracts for going quiet over 90 days. Range 300–1000.
- **Simulator:** change a policy knob (what counts as Gold, how harsh the delay penalty is, how much volume matters) and see the score move immediately.

### Phase 6 — Testing it hard

- Created 10 extreme dealer profiles: Perfect Whale, Micro Newbie, Reformed Defaulter, Fallen Angel, "Oops" Payer, Late Whale, One-Hit Wonder, Installment Grinder, Forgotten Invoice, Advance Payer.
- Checked each score against common sense. Two failed:
  - **Reformed Defaulter** couldn't recover — V1 only looked at late payments, ignoring recent on-time ones.
  - **Forgotten Invoice** — one ₹10 bill left unpaid 500 days wrecked a great dealer's scores.
- Proposed fixes: look at the last few payments overall; ignore tiny invoices below a threshold.
- Also generated synthetic edge-case files to test imports, and kept a regression test for the split-payment bug.

### Phase 7 — Dashboard, deployment, rollout

- Dashboard: dealers, both scores, delays, outstanding amounts, ageing.
- Deployed it (Django app with PostgreSQL); staff and partner businesses used it.
- Owners used the scores to reduce credit or demand faster payment from risky dealers → default-related losses fell \~50%.

**Fill in from memory before the interview (they WILL ask):** internship dates and duration · who supervised you · whether you worked alone · the exact dealer count · how often staff uploaded data · one specific dealer decision the owner made using the score.

## 2. Business and domain — dealers, credit, receivables, defaults

Interviewers start here to check you understood *why* the system exists. If you can explain the business in plain words, everything technical afterwards sounds credible.

### Terms you must use comfortably

| Term | Plain meaning | Example in your project |
| --- | --- | --- |
| Trade credit | Selling now, getting paid later | Dealer buys ₹2 lakh of goods, pays in 30 days |
| Credit terms | Agreed days to pay ("Net 30") | Delay is counted only after these days |
| Accounts receivable (AR) | Total money customers still owe you | Sum of all Pending invoices |
| Days past due (DPD) | Days beyond the due date | Invoice due 1 Oct, unpaid on 31 Oct → 30 DPD |
| Ageing buckets | Outstanding money grouped by DPD: current, 1–30, 31–60, 61–90, 90+ | The debt-ageing view on the dashboard |
| Default | Money that realistically won't be collected | Usually 90+ DPD or written off |
| Bad debt write-off | Accepting a receivable is lost and removing it from the books | The "loss" in default-related losses |
| DSO | Average days it takes to collect | (AR ÷ credit sales) × days in period |
| Advance payment | Money received before an invoice exists | Status "Advance" in your system |
| Credit limit | Max a dealer can owe at once | What the owners adjusted using scores |

### Back-and-forth

**Q: Why does a trading business care so much about late payments?** A: Their cash is tied up in dealers' hands. If dealers pay late, SBM still has to pay its own suppliers, so it borrows or stops buying stock. And the longer an invoice stays unpaid, the less likely it's ever paid — that's how late payment turns into default.

→ *They push:* **Why not just stop giving credit?** A: Credit is how B2B trade works — dealers expect it, and competitors offer it. Refusing credit loses good dealers. The goal was to give credit *selectively*: more to reliable dealers, less to risky ones.

→ *They push:* **What's different about B2B credit vs a bank loan?** A: No interest, no collateral, no bureau check — it's based on relationship and payment history. That's exactly why SBM's own history was the only data available, and why the score is "CIBIL-style" rather than an actual bureau score.

**Q: What is CIBIL, and why call your score that?** A: TransUnion CIBIL is India's main credit bureau; its scores run 300–900 and lenders use them to judge borrowers. Mine is an internal trade-credit score built only from SBM's own invoices and payments. I called it CIBIL-style because the owners instantly understood "higher number = more trustworthy".

→ *They push:* **So it has nothing to do with real CIBIL?** A: Correct — no bureau data, no regulatory standard. Same idea, internal use only.

**Q: Who are the dealers, and how big is the range?** A: Over 100 businesses that buy from SBM. Sizes vary hugely — from ₹1,000 total purchases to tens of lakhs — which is why the score had to reward volume without letting size override payment behaviour.

**Q: What were the owners actually deciding?** A: Three things: whether to give a dealer more credit, how much (the limit), and whom to chase for payment first. The scores answer the first two; the ageing dashboard answers the third.

**Q: What causes defaults in this business?** A: Usually gradual slipping — a dealer's delays creep from 5 days to 20 to 60, then they go silent. That's why V1 watches *recent* delays and inactivity: it catches the slide before the debt turns into a write-off.

## 3. Bullet 1 — "real-time CIBIL-style credit scoring simulator"

This bullet gets the deepest grilling in an AI-engineer interview: they'll push on how the score works, why it's designed that way, and how you know it's right. Lead with the question the score answers, never with the formula.

### The scores in plain words (memorise this)

- **V1 "Dynamic" — is this dealer paying well *right now*?** Start at 1000. Take the dealer's last 5 late payments, average how late they were (bigger invoices count more), and subtract 5 points per day. Subtract 2 points per day since their last order. Never below 100.
- **V2 "Standard" — is this dealer valuable and reliable *overall*?** Start at 300. Find the "typical" delay — the delay at which half of the dealer's late money had been paid (an amount-weighted median). Gold (≤ 4 days) adds 300, Average (≤ 15) adds 200, Poor adds 100 minus 10 per extra day. Add 25 × log₁₀(total sales). After 90 days of no orders, subtract 0.5 per day. Range 300–1000.
- **Simulator:** pick a dealer, change a policy knob (Gold limit 4 → 10 days, delay penalty, volume weight, median → 80th percentile) and watch the score change instantly.

### Opening chain

**Q: Explain your credit score.** A: Two scores answer two questions. V1 is a short-term pulse — it drops fast when recent payments slip or the dealer stops ordering. V2 is lifetime value — typical lateness, how much they buy, and whether they've gone quiet. Together they separate "always risky" from "good dealer recently slipping".

→ **Why two? Wouldn't one number be simpler?** One number hides things. A dealer who's bought ₹35 lakh over two years but paid the last three invoices 60 days late would still look fine on a blended score. With two scores, V2 says "valuable", V1 says "slipping — watch closely".

→ **How does the owner read two numbers?** A simple 2×2: both high → VIP, raise limits · V1 high, V2 medium → good new or small dealer · V1 low, V2 high → recently slipping, watch closely · both low → don't extend credit.

### Design-choice chain

**Q: Why weight by invoice amount?** A: A ₹5 lakh invoice paid 30 days late hurts cash flow far more than a ₹500 one. Weighting measures how late the *money* is, not how many invoices.

→ **Why an average in V1 but a median in V2?** V1 should react quickly to recent trouble, so an average over a short window is right. V2 represents typical long-term behaviour, so one extreme invoice shouldn't drag it — the median resists outliers.

→ **Why log of sales?** Sales ranged from ₹1,000 to ₹35 lakh. On a log scale, 100× more sales adds only 50 points, so big dealers get credit for value but can't buy a good score while paying late.

→ **Why count only delay beyond credit terms?** Paying on day 25 of a 30-day term is perfectly fine. Lateness only starts after the agreed date. And paying early isn't rewarded as negative lateness — otherwise one early payment could cancel a late one.

→ **What does an unpaid invoice do to the score?** Its delay is counted up to today, so it keeps getting worse every day until it's paid. That's intentional — an ignored invoice is the strongest risk signal.

### The "where did the numbers come from" chain (they will push here)

**Q: Why 1000, 5, 2, 25, 90 days? Aren't these arbitrary?** A: They're policy parameters, not learned weights. I started from what the owners considered good behaviour — "within 4 days is excellent, beyond 15 is a problem" — then tuned with the simulator until my 10 test profiles landed where the owners agreed they should.

→ **So it's just your opinion encoded?** It's business judgement made explicit and adjustable. That was the right first version because there was no record of who defaulted to learn from. Once outcomes are logged, I'd fit the weights with logistic regression and check if the hand-set ones were close.

→ **Why didn't you use machine learning?** Three reasons: no labelled defaults, only \~100 dealers, and the owner had to explain a decision to a dealer face-to-face. A transparent score you can explain line by line beats a black box there. (Section 7 covers how you'd build the ML version.)

### "Real-time" chain

**Q: What's real-time about it?** A: Every upload recomputes every affected dealer's scores immediately, and the simulator responds instantly to a parameter change. The owners always saw current scores, not last month's report.

→ **So it's not streaming?** No — it's event-driven: triggered by each data upload, not a continuous stream. For this business, data changed when accountants exported, so that was the right design. One gap: V1's inactivity part changes daily even without uploads, so a nightly recompute would keep it exact.

### Validation chain (most important for AI roles)

**Q: How do you know the scores are right?** A: With no default labels, I validated behaviour instead: (1) 10 extreme dealer profiles where the right answer is obvious, (2) sensitivity — how much each knob moves the score, (3) sanity rules — more delay must never raise a score, more sales must never lower V2, and (4) owners' review of real dealers they knew well.

→ **Give an example of a profile.** "Perfect Whale" — ₹35 lakh in sales, always on time → V1 990, V2 763, top tier. "Late Whale" — ₹15 lakh but always 20 days late → V2 about 504, medium; volume kept them from bottom but couldn't buy a top score.

→ **Did anything fail?** Yes, two. **Reformed Defaulter**: V1 looked only at the last 5 *late* payments, so a dealer who'd paid on time for a year still carried old defaults — they could never recover. **Forgotten Invoice**: one ₹10 bill unpaid for 500 days dragged a perfect dealer to the floor. Fixes: use the last 5 payments overall, and ignore late invoices below a small threshold like ₹1,000.

→ **What would you do now to validate properly?** Back-test: freeze scores as of a past date, then check what dealers actually did in the next 3 months. Low scores should mean more late payments, band by band.

## 4. Bullet 2 — "internal web platform… cut default-related losses by 50%"

This bullet is tested in two halves: the engineering (how data gets in correctly) and the impact (how 50% was measured and whether your tool caused it). Prepare the 50% answer word for word — it's the question most likely to be asked twice.

### Engineering chain

**Q: What did the platform replace?** A: Staff exported Zoho or Tally reports into Excel, reconciled them by hand, and the owner decided credit from memory. Now they upload the export, and the system cleans it, stores it, scores every dealer and shows it on one dashboard. Zoho and Tally stayed the accounting system — I replaced the manual spreadsheet layer around them.

→ **Why not just build reports inside Zoho?** Zoho shows invoices and payments, but it doesn't score dealer behaviour, combine reports, or let owners simulate credit policies. Tally users and Zoho users also needed one common view.

**Q: Walk me through what happens when someone uploads a file.** A: The system reads the Excel or Google Sheet, finds the real header row (Zoho adds a title row above it), drops empty columns, maps column names to one standard set, fixes split payments, builds a unique ID for each row, works out each invoice's status and delay, saves dealers and payments, and recomputes scores.

→ **What was the hardest bug?** The ₹3 lakh that didn't exist. In Zoho's "Payments Received" export, one ₹1 lakh payment covering three invoices appears on three rows, each showing ₹1 lakh. Summed, it looked like ₹3 lakh. I switched to the per-invoice "amount applied" column, made each row's ID payment + invoice, and kept a test so the bug can't come back.

→ **How did you even notice it?** Totals didn't match what the accountant expected for a dealer. Tracing one dealer's rows showed the same payment repeated. *(Confirm your own version of this moment — interviewers love "how did you find it".)*

→ **What if someone uploads the same file twice?** Nothing duplicates. Every row has a stable unique ID and the database enforces uniqueness; a row that already exists is updated instead of inserted. Uploading the same file twice gives the same result as once — that's idempotency.

→ **What if the upload fails halfway?** Ideally the import runs as one database transaction — all or nothing — and because it's idempotent, re-uploading is safe. *(Check whether yours is wrapped in a transaction; if not, say it's the improvement you'd make.)*

→ **How fast is it?** Instead of saving row by row, rows are written in batches of 2,000; existing dealers are loaded in one query; all payments are loaded once and grouped per dealer in memory, so scoring 100+ dealers takes a handful of queries, not hundreds.

### Adoption chain

**Q: Who used it?** A: Owners for credit decisions, accountants for uploads and reconciliation, sales staff to check a dealer before taking an order, and partner businesses for their own dealers.

→ **How did dealers use it?** *Pick the true one before the interview:* dealers saw their statements/outstanding balance, or dealers were affected through credit limits and payment reminders driven by the scores. Don't claim dealers logged in unless they did.

→ **How did you get people to adopt it?** It was easier than what they did before — upload one file instead of hours in Excel — and it answered the exact question the owner asked every week: "who should I worry about?"

→ **What did users complain about?** *Recall one real piece of feedback* — e.g. a score that looked unfair for a dealer the owner trusted, which led to the Reformed Defaulter / Forgotten Invoice fixes. Real feedback stories are gold.

### The 50% chain — prepare these answers word for word

**Q: Default-related losses fell 50% — what does "default-related loss" mean?** A: Money owed by dealers that was written off or stuck long overdue — in practice, invoices over 90 days past due plus bad debt written off. *(Confirm the exact definition the owners used.)*

→ **50% compared to what? Over what period?** A: Compared to the same length of time before the platform. *(Know: e.g. "six months before vs six months after rollout".)* Ideally as a share of credit sales, so a change in sales volume doesn't distort it.

→ **Who measured it — you or the business?** Answer truthfully: the owners from their books, or you from AR-aging exports before and after. Either is fine; saying "the owners reported it from their accounts" is honest and normal for an internship.

→ **How did your tool cause that?** A: Through decisions. The scores made it obvious which dealers were slipping, so the owners reduced credit limits, asked for advance or faster payment from low scorers, and chased the oldest debts first using the ageing view. Fewer new invoices went to dealers who were about to default.

→ **Couldn't it be something else — seasonality, fewer sales?** A: Yes, and I wouldn't claim all of it. Sales seasonality and a general tightening of credit could contribute. But the tightening itself happened because the tool showed whom to tighten on — before, the owners couldn't see it.

→ **You said you never back-tested the scores. How do you claim an outcome?** A: Those are two different things. The 50% is a business result the owners measured on their books. Back-testing would prove the *scores themselves* predict late payment band by band — that's still open and the first thing I'd add.

→ **With 100 dealers, isn't 50% just one or two big defaulters?** A: Possibly — with a small portfolio a single large default moves the number a lot. That's why I'd present it alongside a steadier measure like the share of receivables over 90 days.

**Write your own answers here before the interview:** definition used · before period and after period · before value and after value (₹ or %) · who measured · one concrete dealer whose credit was cut thanks to the score.

## 4A. The 50% loss reduction — full deep dive

If they spend ten minutes on one thing, it will be this. You survive it by having four things fixed in your head: a **definition**, a **measurement**, a **mechanism** (how the tool led to it), and **honest limits**. Never change any of them mid-conversation.

### The four fixed pieces

| Piece | What you say | Lock it before the interview |
| --- | --- | --- |
| Definition | Default-related loss = money from dealer invoices that was written off as bad debt, or stuck more than 90 days past due | Confirm with the owners which one they counted |
| Measurement | Same-length period before vs after rollout, as a share of credit sales | Know the two periods and the two values |
| Mechanism | Scores flagged slipping dealers → owners cut limits, asked for advance or faster payment, chased old debt first → less money went to dealers about to default | Have one real dealer example |
| Limits | Small portfolio, seasonality, owners also tightened policy; scores themselves not yet back-tested | Say these yourself before they do |

### How the measurement works (understand this, don't memorise it)

Raw rupees can mislead: if sales fell by half, losses would fall by half without any improvement. So you compare a **rate**:

**Loss rate = default-related loss ÷ credit sales in the same period**

*Illustration only — use your real numbers:* before, ₹2 crore credit sales and ₹12 lakh lost → 6%. After, ₹2.2 crore and ₹6.6 lakh lost → 3%. Reduction = (6 − 3) ÷ 6 = 50%. Notice sales went *up* while losses halved — that's the strongest version of the story, because it shows you didn't just stop selling.

If the owners measured raw rupees rather than a rate, say so honestly and add: "sales were roughly flat / grew, so the rate fell too" — only if true.

### Why losses take time to show (they may test this)

A default shows up 90+ days after the credit is given. So a credit decision made in March only proves itself in June or July. That means:

- The "after" window must start a few months after rollout, or be long enough to include that lag.
- Early on, you watch **leading indicators** — share of debt over 30 or 60 days, average delay of new invoices — because they move first.
- Losses are a **lagging indicator** — the final proof, but slow.

Saying this unprompted shows you understand the business, not just the number.

### The mechanism, step by step

1. Upload → scores recomputed → a dealer's V1 drops because recent payments slipped.
2. Owner sees it on the dashboard before the debt is badly overdue.
3. Owner acts: lowers the credit limit, asks for part-payment upfront, pauses new supply until old invoices clear, or calls them first in the collection list.
4. Less new money goes out to that dealer while they're struggling; old money is chased earlier while it's still collectible.
5. Fewer invoices reach 90+ days or get written off.

The tool didn't collect money by itself — it **moved decisions earlier and targeted them**. Saying it this way is both honest and impressive.

### The long back-and-forth (practise all of it)

**Q: How did you measure 50%?** A: Loss rate before vs after: write-offs plus 90+ day overdue, divided by credit sales, over the same length of period. It fell from about X% to Y%.

→ **Who calculated it?** *(Truth: the owners from their books, or you from AR-aging exports.)* "The owners tracked it from their accounts; I helped compare the ageing reports."

→ **Over what period?** *(Real periods.)* Add: "defaults lag credit decisions by about three months, so we looked at a window long enough to include that."

→ **How do you know it was your tool and not luck?** A: I can't prove full causation — there was no controlled experiment. But the drop came from exactly the dealers the scores flagged: their credit was tightened and their overdue amounts fell, while reliable dealers were treated as before. *(Only if true — otherwise: "the timing matched the rollout, and the owners attributed it to acting on the scores.")*

→ **How would you prove it properly?** A: Three ways, from simple to rigorous. (1) Year-on-year: compare the same months last year to remove seasonality. (2) Compare groups: dealers whose credit was tightened vs similar dealers who weren't — if losses dropped mainly in the tightened group, that points to the tool. (3) For future changes, a small holdout: apply a new policy to some flagged dealers first and compare. This "compare the change in two groups" idea is called difference-in-differences.

→ **Couldn't the owners have just become stricter on their own?** A: They did become stricter — but the scores told them *whom* to be strict with. Before, being stricter meant tightening everyone and losing good dealers. That's the difference.

→ **Did cutting credit hurt sales?** A: That's the real trade-off — the easiest way to cut losses to zero is to stop giving credit. The goal was losses down without losing good dealers. *(If you know: "sales stayed flat / grew".)* Tightening only applied to low scorers, and high scorers kept or got higher limits.

→ **What about good dealers wrongly flagged?** A: That's a false positive. The Reformed Defaulter case was exactly that, which is why I fixed V1 to look at recent payments overall. The owner also had the final say — the score advised, it didn't auto-reject.

→ **With \~100 dealers, isn't 50% fragile?** A: Yes. One large default could swing it either way. That's why I'd always report it next to steadier numbers — share of receivables over 90 days, number of dealers who defaulted, average delay of new invoices.

→ **What's the money impact in rupees?** *(Know it if you can: "roughly ₹\_\_ lakh less lost over \_\_ months".)* A rupee number makes the 50% concrete.

→ **What would make the 50% not last?** A: The scores going stale (no nightly refresh), the economy turning so everyone pays late, or owners ignoring the score over time. Monitoring the late-payment rate per score band would warn early.

→ **So is your score a good predictor?** A: The business result is encouraging, but it isn't the same as proving the score predicts defaults. To prove that I'd back-test: take scores from three months ago and check whether low bands really paid later. That's my top next step.

### One sentence to fall back on under pressure

> "Default-related losses — write-offs plus 90-day-overdue debt — fell about half as a share of credit sales, comparing the period before and after rollout; the tool's contribution was showing the owners which dealers to tighten on early, and I'd confirm it rigorously with a back-test and a group comparison."

## 5. Bullet 3 — "operational dashboard… receivables and debt-ageing KPIs"

Here they test whether you know what each KPI means and why a business watches it. Explain every metric as "what it is → why the owner cares → what action it triggers".

### KPIs you should be able to explain

| KPI | What it is | Why the owner cares | Action it triggers |
| --- | --- | --- | --- |
| Total outstanding | All money dealers currently owe | Cash locked outside the business | Overall credit tightening or loosening |
| Ageing buckets | Outstanding split into current, 1–30, 31–60, 61–90, 90+ days past due | Older debt is much less likely to be collected | Chase the oldest buckets first |
| % over 90 days | 90+ bucket ÷ total outstanding | Headline "how much might we lose" number | Escalate collection, stop new credit |
| DSO | Average days to collect: (AR ÷ credit sales) × days | Rising DSO = cash coming in slower | Review credit terms |
| Per-dealer exposure | Outstanding per dealer, largest first | Too much owed by one dealer is concentration risk | Cap that dealer's limit |
| Payment status mix | Paid / Pending / Advance | Shows prepaying vs owing dealers | Reward advance payers |
| Score distribution | How many dealers in each score band | Health of the whole dealer base | Policy changes |

Days past due = days since the due date, where due date = invoice date + credit terms.

### Back-and-forth

**Q: What's on the dashboard?** A: Every dealer with their two scores, how late they typically pay, what they owe right now, and how old that debt is. At the portfolio level: total outstanding, the ageing buckets, and which dealers carry the most risk.

→ **Why ageing buckets specifically?** Because collection chances drop with age. ₹1 lakh that's 10 days late is usually collected; ₹1 lakh that's 120 days late often isn't. Buckets tell the team where to spend collection effort.

→ **Why 30/60/90?** It's the standard accounting convention, and the owners already thought in months. Matching their mental model made adoption easy.

→ **Walk me through DSO with numbers.** If dealers owe ₹20 lakh and credit sales over the last 90 days were ₹60 lakh, DSO = 20 ÷ 60 × 90 = 30 days — on average it takes a month to collect.

→ **What's the difference between the ageing view and the score?** Ageing is about *money* — how old today's debt is. The score is about the *dealer* — how they've behaved over time. A great dealer can have one old invoice; a risky dealer can owe nothing today but still deserve caution.

**Q: How did it "streamline reporting workflows"?** A: Before, the weekly overdue list meant exporting three reports and reconciling them by hand. After, the accountant uploads once and the view is ready. *(Have the real before/after time — e.g. "a few hours a week down to minutes".)*

→ **And "cash-flow visibility"?** The owners could see at a glance how much money was out, how old it was, and who was likely to pay soon versus late, so they planned purchases with a realistic idea of incoming cash.

**Q: How does the dashboard load quickly?** A: Scores are computed at upload time and stored with each dealer, so the page just reads them instead of recalculating on every visit. Trade-off: they're only as fresh as the last upload, which is why a nightly refresh is on my improvement list.

**Q: Who decided which KPIs to show?** A: I asked the owners what they looked at before giving credit — total due, how overdue, and the dealer's history — and built those first. Standard AR metrics like ageing and DSO filled in the rest.

**Q: What would you add next to the dashboard?** A: A short explanation for each score ("+300 Gold tier, +137 volume, −40 inactivity"), a weekly expected-collections forecast, and alerts when invoices cross 30/60/90 days.

## 6. Data engineering topics they branch into

HiLabs is a data-quality company, so expect them to leave your bullets and ask general questions on cleaning, schemas and deduplication — answer each by tying it back to something you actually hit at SBM.

### Data cleaning

**Q: What kinds of dirty data did you deal with?** A: Four kinds: format problems (title rows above headers, empty columns), naming problems (same field, different column names across reports), meaning problems (an AR-aging row looks like a payment but is actually debt still owed), and duplication problems (split payments repeating amounts; files re-uploaded).

→ **Which was hardest?** Meaning problems. Format and naming break loudly. Meaning problems give you numbers that look fine but are wrong — like the tripled totals. You only catch them by understanding what each report represents.

→ **How do you catch silent errors in general?** Reconcile against something trusted: totals per dealer vs the accountant's figure, row counts in vs rows stored, sum of payments vs sum applied to invoices. Then add sanity rules: no negative amounts, payment date not before invoice date, every payment linked to a dealer.

### Schema mapping

**Q: How did you handle different report formats?** A: A mapping layer: normalise each column name (lowercase, strip spaces and underscores), then match it against a list of known synonyms for each standard field — e.g. "payment date", "receipt date" and "date" all map to Date.

→ **What happens with a new format you've never seen?** The mapping fails for that column, so the import should stop and say which column it couldn't map rather than guess. Adding a new synonym fixes it. An LLM can propose the mapping for unseen headers (section 7).

→ **Why rules instead of something smarter?** Rules are instant, free, deterministic and easy to debug — the same file always maps the same way. For a handful of report types, that's the right trade-off.

### Deduplication and identity

**Q: How did you make sure each payment is stored once?** A: A unique ID per row. Where Zoho had a payment ID, I combined it with the invoice ID (because one payment can cover several invoices). Where no ID existed, I built one from dealer + date + amount. The database enforces uniqueness, so duplicates are impossible, not just unlikely.

→ **What's the risk with a fallback key like dealer + date + amount?** Two genuinely different payments by the same dealer, same day, same amount would collide. It's rare but real — I'd add an invoice number or a sequence number when available.

→ **What about the same dealer spelled differently in two files?** That's entity resolution. I relied on Zoho's dealer IDs, which avoided most of it. Without IDs, you'd normalise names (case, spaces, "Pvt Ltd" vs "Private Limited"), use fuzzy matching, and confirm matches with extra fields like GST number or phone.

### Idempotency

**Q: What does idempotent mean, and why did it matter?** A: Doing something twice gives the same result as once. Accountants re-export and re-upload constantly; without idempotency every re-upload would double receivables and corrupt every score.

→ **How is it achieved?** Stable unique key + "insert or update" (an upsert). If the row exists, update it; if not, insert it. Newer exports simply update balances.

### Data quality in general

**Q: How would you measure data quality for a pipeline like this?** A: Completeness (missing dates, amounts), validity (dates parse, amounts numeric and positive), uniqueness (no duplicate keys), consistency (totals reconcile with the source), and freshness (when was the last upload). Each gets a check and a count in an import report: rows added, updated, skipped, and why.

## 7. ML and AI topics they branch into

For an AI-engineer role, the most likely pivot is "you built rules — how would you build the ML or LLM version?" Show you know when ML is justified, how you'd set it up without fooling yourself, and where an LLM genuinely helps.

### Rules vs ML

**Q: When is a rule-based score better than a model?** A: When you have no labels, little data, and the decision must be explained to a person. All three were true at SBM. ML wins when you have outcomes to learn from, enough examples, and patterns too subtle to write by hand.

→ **What would make you switch to ML here?** Logging outcomes. Once every invoice records whether it ended up 30/60/90 days late or written off, we have labels — after a few months, enough to train and compare against the rule-based score.

### Building the ML credit model

**Q: Design a model to predict which dealers will default.** A: Walk it in five steps.

1. **Define the label:** "this dealer will have an invoice more than 60 (or 90) days past due in the next 3 months." Defaults are rare, so a slightly earlier signal gives more positive examples.
2. **Features as of a cut-off date:** typical delay, trend of the last few delays (getting worse?), current outstanding, share of debt over 60 days, order frequency, days since last order, total purchases, how long they've been a dealer. The current V1/V2 scores themselves are useful features.
3. **Split by time, not randomly:** train on older months, test on the most recent. A random split lets the model peek at the future.
4. **Model:** logistic regression first — interpretable, works on small data. Try gradient-boosted trees only if they clearly beat it, and explain them with SHAP.
5. **Evaluate:** AUC for ranking, precision-recall because defaults are rare, calibration (does "20% risk" really mean 1 in 5?), and late-payment rate per score band.

→ **What is data leakage here?** Using information the model wouldn't have had at prediction time — e.g. including payments made after the cut-off date, or a "written off" flag that only appears after the default. Leakage gives great test scores and useless real predictions.

→ **Only \~100 dealers — isn't that too little?** Yes for dealer-level modelling. Two fixes: predict at the invoice level (thousands of invoices), and keep models simple with regularisation. Honestly, with 100 dealers the rule-based score plus back-testing may stay the better choice.

→ **Class imbalance — how do you handle it?** Use metrics that respect it (precision-recall, not accuracy), class weights in training, and pick the decision threshold from business cost — missing a defaulter costs far more than wrongly tightening a good dealer.

→ **How do you choose the threshold?** From costs: expected loss from a missed default vs lost sales from refusing credit to a good dealer. Pick the cutoff that minimises total expected cost, and let owners adjust it.

### Predicting payment timing

**Q: How would you predict when an open invoice will be paid?** A: Survival analysis. Unpaid invoices are "censored" — we know they've taken at least N days but not the final answer. Regular regression would wrongly treat them as finished or drop them. Survival models handle that properly and give a probability of payment by each future week.

→ **What would you use that for?** Cash-flow forecasting: expected collections next week = sum over open invoices of amount × probability it's paid that week.

### Explaining and trusting scores

**Q: Credit decisions affect businesses — how do you keep it fair and explainable?** A: Every score shows its breakdown, the owner makes the final call, and dealers can be told *why* their limit changed ("your last three payments averaged 40 days late"). Use only payment behaviour, never anything about who owns the business. Monitor whether the score keeps predicting well over time.

→ **What is model drift and how would you detect it?** When the world changes and the model's patterns stop holding — e.g. a market slowdown makes everyone pay later. Watch input distributions (average delay creeping up) and outcomes (late-payment rate per band changing), and re-validate regularly.

### Where LLMs fit

**Q: Where would you use an LLM in this system?** A: Three places, always behind checks:

- **Mapping unknown columns.** Rules handle known formats. For an unseen header, give the LLM the header, a few sample values and the list of standard fields, and ask it to pick one or say "unknown". Then validate: a column mapped to Date must actually parse as dates. Cache the answer so each new format costs one call.
- **Matching dealer names** across files — fuzzy matching finds candidates, the LLM judges close cases, a human confirms uncertain ones.
- **Plain-English explanations** of a score for the owner, generated from the score's breakdown — not invented from free text.

→ **How do you stop the LLM from hallucinating a wrong mapping?** Restrict its answer to the fixed list of fields, allow "unknown", check the mapped data with validation rules, and require human approval when confidence is low or checks fail.

→ **How would you evaluate the LLM mapper?** Build a test set of real headers from past exports with the correct mapping, measure accuracy per field and how often it abstains, and test on formats it hasn't seen.

→ **Would you replace the scoring with an LLM?** No. Scoring needs consistent, numeric, auditable results — the same input must give the same score. LLMs are good for messy text and unseen formats, not for the core arithmetic of credit decisions.

### Quick concepts they may test

- **Mean vs median:** the mean is pulled by outliers; the median isn't.
- **Weighted median:** sort delays, add up invoice amounts until you cross half of the total; that delay is the answer.
- **Log transform:** compresses huge ranges (₹1k to ₹35 lakh) so size helps with diminishing returns.
- **Monotonicity:** more delay should never improve a score — a simple, powerful sanity test for any scoring model.
- **Scorecards:** banks often use logistic regression on grouped (binned) features because it's explainable to regulators.

## 8. System design branches

They'll ask "what if it had to scale?" to see whether you understand the limits of what you built. Always answer in two parts: what you built and why it fit SBM's size, then what you'd change at 100× and why.

### Architecture you built

A single Django app (a monolith) with PostgreSQL. Upload → cleaning pipeline (pandas) → dealers and payments tables → scoring step → scores stored on each dealer → dashboard reads them.

**Q: Why Django and PostgreSQL?** A: The data is naturally relational — dealers have many invoices and payments. PostgreSQL enforces uniqueness (key for idempotency) and supports upserts. Django gave login, an admin panel and an ORM for free, and it's Python, so the pandas cleaning code and the web app live together.

→ **Why a monolith, not microservices?** One developer, 100+ dealers, one business. Microservices add deployment and network complexity with no benefit at that size. I'd split only if a part needed to scale independently — like imports.

### Scale chain

**Q: What breaks if data grows 100×?** A: Three things. Loading all payments into memory to score them. Long imports blocking the web request and timing out. And recomputing every dealer even when only a few changed.

→ **How would you fix each?** Recompute only the dealers touched by the upload. Push heavy aggregation into the database instead of Python. Run imports as background jobs: the upload returns immediately, a worker processes the file, the page shows progress.

→ **How do background jobs work?** The web app puts a task on a queue (e.g. Redis); a separate worker process picks it up and runs it; status is stored so the UI can show "processing / done / failed". A scheduler on the same setup can run the nightly score refresh. *(Celery and Redis were already in the project's dependencies — only claim they ran jobs if they actually did.)*

→ **Which database indexes matter?** A unique index on the payment ID (used for upserts), an index on dealer ID (every per-dealer lookup), and one on dealer + invoice date (history in order).

### Real-time chain

**Q: If the owner wanted truly live scores, what would you build?** A: Connect directly to Zoho's API instead of file uploads — pull new invoices and payments every few minutes or via webhooks when they happen — and recompute only the affected dealer. Plus a nightly job so time-based parts (inactivity, unpaid invoices ageing) stay current.

→ **Is that worth it?** Only if decisions are made within hours. SBM made credit calls daily or weekly, so upload-triggered updates were enough. Matching freshness to how fast decisions are made is the real design choice.

### Reliability and correctness chain

**Q: Two people upload at the same time — what happens?** A: The unique key plus upsert means both end in the same rows, no duplicates. Score recomputation should run one at a time — a single background job or a lock — so two updates don't interleave.

**Q: How do you store money safely?** A: As exact decimals, never floating point — floats can't represent 0.1 exactly, so totals drift by paise. *(Check what your models used; if floats, say you'd switch to decimal and round to two places.)*

### Security and multi-tenancy chain

**Q: It's "internal" — how is it protected?** A: Login required, and roles: owner, accountant, sales, partner. *(Confirm what your deployment actually had.)*

→ **Partner businesses used it — how do they not see each other's dealers?** Every dealer and payment belongs to a company, and every query is filtered by the logged-in user's company. Plus an audit log: who uploaded what, when, and what changed.

→ **What's sensitive in this data?** Dealer names, amounts owed and payment behaviour — commercially sensitive. So: no real data in public code repos, access by role, and synthetic data for testing and demos.

### Monitoring chain

**Q: How would you know the system is healthy?** A: An import report every time (rows added, updated, skipped, why), alerts when an upload fails or a column can't be mapped, a "last updated" time on the dashboard, and tracking whether score distributions suddenly shift — usually a sign of bad data rather than dealers changing overnight.

## 8A. Handling 100 concurrent users

Start every answer with the honest baseline — at SBM a handful of staff used it at once — then show you can reason about 100 using a few simple ideas. Interviewers reward the reasoning, not a memorised architecture.

### First principles — understand these five ideas and you can answer anything here

**1. "100 concurrent users" is not "100 requests at the same instant".** A user clicks, reads for \~10–30 seconds, clicks again. 100 active users clicking every \~10 seconds is about **10 requests per second**. Always convert users → requests per second first.

**2. Little's law: requests in flight = arrival rate × time per request.** If 10 requests/second each take 0.1 s, only about **1 request is being processed at any moment**. If each takes 2 s (a slow page), that becomes 20 in flight. So slow requests, not user count, are what hurt. Making requests fast is the cheapest scaling.

**3. A web server handles requests with workers.** Django runs under Gunicorn; each worker is a separate process that handles one request at a time (more with threads). 4 workers × \~10 fast requests/second each ≈ 40 requests/second — already far beyond 100 users' needs. Common starting rule: about 2 × CPU cores + 1 workers.

**4. The database is usually the real limit.** Each worker holds a database connection; PostgreSQL allows a limited number of connections (often \~100 by default), and each costs memory. Many app servers or serverless functions can exhaust them. The fix is a **connection pooler** (like PgBouncer): many app connections share a small set of real database connections.

**5. Separate reads from heavy writes.** Viewing the dashboard is a cheap read. Uploading a 20,000-row file and recomputing scores is heavy. Never let heavy work run inside a user's web request — put it on a **queue** and let a **background worker** do it.

### Your answer, in layers (give it in this order)

**Layer 1 — Make each request cheap.** Scores are precomputed at upload time and stored, so the dashboard just reads them. Add database indexes, fetch related data in one query (no N+1), and paginate the dealer list. A fast page is the biggest win.

**Layer 2 — Cache what many people read.** 100 users mostly look at the same dashboard. Cache the dashboard data (e.g. in Redis) and clear the cache whenever a new upload changes the data. Most requests then never touch the database.

**Layer 3 — Enough web workers, stateless servers.** Run several Gunicorn workers; keep no user data in server memory (sessions in DB or Redis), so you can add more servers behind a load balancer when needed. Static files (CSS, JS) served from a CDN, not Django.

**Layer 4 — Protect the database.** Connection pooling; enough indexes; if reads ever dominate, a read replica for dashboards while writes go to the main database.

**Layer 5 — Move heavy work to background jobs.** Upload returns immediately ("processing…"), the file goes on a queue, a worker imports it and recomputes scores, the page shows progress. Users never wait on a 30-second request, and web workers stay free for everyone else.

**Layer 6 — Prove it with a load test.** Simulate 100 users with a tool like Locust or k6, watch response time (especially the slowest 5% — "p95 latency"), error rate and database connections. Fix the first bottleneck, test again.

&#91;embedded content: the system scaled for 100 users · 8 parts\]

Draw this on a whiteboard if asked: page views go through the web workers and mostly hit the cache, while uploads become jobs that import workers process in the background and write to PostgreSQL, clearing the cache when done.

### The concurrency problems specific to your system

| Situation | Risk | Fix |
| --- | --- | --- |
| Two accountants upload the same file at once | Duplicate rows | Unique key + upsert — both end in the same rows (idempotency already handles it) |
| Two uploads for the same company overlap | Score recomputes interleave; one overwrites the other with stale data | Process one import per company at a time (queue per company, or a database lock) |
| An upload happens while someone views the dashboard | Viewer sees half-updated data | Import inside one transaction — viewers see old data until it fully commits |
| Owner edits a dealer's credit limit while a recompute runs | Lost update | Recompute writes only score fields; or check a version number before saving (optimistic locking) |
| 100 people upload big files at once | Web servers jammed, timeouts | Queue + limited number of workers; file-size limits; rate limits per user |
| Serverless deployment opening many DB connections | Database refuses connections | Connection pooler in front of PostgreSQL |

### Back-and-forth

**Q: Can your system handle 100 concurrent users?** A: As built for SBM, it served a few staff at a time. For 100: that's roughly 10 requests per second, which a few Gunicorn workers handle easily if pages are fast. The real risks are heavy uploads blocking workers and running out of database connections — so I'd move imports to background workers, add caching for the dashboard, and put a connection pooler in front of PostgreSQL. Then I'd load-test to confirm.

→ **Where would it break first?** A: Uploads. An import plus score recompute runs inside the web request, so a few big uploads at once would tie up workers and others would wait or time out. Second: database connections, especially on a serverless host.

→ **Why not just add more servers?** A: More web servers help only if the database can keep up — they'd each open more connections and could overload it. Fix the slow requests and protect the database first; scale out after.

→ **What's the difference between a process and a thread here?** A: A process has its own memory; Python processes run truly in parallel on multiple cores. Threads share memory and are cheaper, but Python's global interpreter lock lets only one thread run Python code at a time — threads still help when requests mostly wait on the database or network. Gunicorn typically uses several processes, optionally with a few threads each.

→ **What if one dealer's page is viewed far more than others?** A: That's a "hot key". Cache it; since scores only change on upload, the cache can last until the next upload.

→ **How do you keep cached data from being wrong?** A: Invalidate on write: every successful import clears (or rebuilds) the cached dashboard for that company. Since data only changes at upload, this is simple and exact.

→ **How do you handle a race condition in score recomputation?** A: Only one recompute per company at a time — a queue that processes that company's jobs in order, or a database lock taken at the start of the job. Because the job is idempotent, a retry after a crash is safe.

→ **What would you monitor?** A: Request rate, p95 latency, error rate, queue length and job duration, database connections and slow queries. Alerts if queue length keeps growing (workers can't keep up) or errors spike.

→ **What if the 100 users are from different partner companies?** A: Then data isolation matters as much as speed: every dealer and payment tagged with a company, every query filtered by the user's company, caches keyed by company so one company never sees another's data.

→ **What's your rough capacity estimate if asked to commit?** A: Show the method rather than a guess: users × clicks per second = requests per second; × time per request = requests in flight; compare with workers available; check database connections needed. For 100 users with \~100 ms pages, a single small server with a handful of workers is comfortably enough — the work is in uploads and the database, not the page count.

## 8B. Running models at scale, efficiently

The core idea to carry in: **decide when the model runs (ahead of time vs on request), make each run cheap, and only recompute what changed.** Your SBM scoring already follows this — scores are computed in batch at upload, stored, and read instantly — so start from what you built and scale it up.

### First principles

**Batch vs online inference.**

- *Batch:* compute predictions ahead of time for everyone (nightly or on upload), store them, read instantly. Cheap, simple, perfect when inputs change slowly. **Your system is batch.**
- *Online:* compute when a request arrives. Needed when the input only exists at request time (e.g. scoring a brand-new order as it's placed). Costs latency on every request.
- Rule of thumb: if the answer can be known before anyone asks, precompute it.

**Incremental computation.** Don't recompute 100,000 dealers because 200 had new payments. Track which dealers changed in this upload, recompute only them, plus a nightly job for time-based parts (inactivity, unpaid invoices ageing).

**Vectorise, don't loop.** Doing math on whole columns at once (pandas/NumPy, or SQL aggregates in the database) is often 10–100× faster than a Python loop per row, because the work runs in optimised C code.

**Push work to where the data is.** Aggregations like "median delay per dealer" can run inside PostgreSQL instead of pulling every row into Python — less data moved, less memory.

**Load once, use many times.** For an ML model, load it into memory once when the worker starts — never on every request. Loading a model file per request is the most common performance mistake.

**Batch requests together.** Predicting 1,000 rows in one call is far cheaper than 1,000 calls of one row — true for tabular models and especially for LLMs and GPUs.

### Scaling path for your scoring, step by step

| Data size | What to do |
| --- | --- |
| \~100 dealers, thousands of payments (SBM today) | Load in memory, group per dealer, compute, bulk-save. Seconds. |
| \~10,000 dealers, millions of payments | Recompute only changed dealers; compute aggregates in SQL; process in chunks; run in background workers |
| \~1 million dealers, billions of rows (HiLabs-scale) | Columnar files (Parquet), distributed processing (Spark/Dask), partition by dealer or date, scheduled pipelines, results written to a serving database |

### If the score becomes an ML model — how to serve it

1. **Train offline** on historical data; save the model with a version number (a model registry such as MLflow keeps track of versions, metrics and which data trained it).
2. **Same features in training and serving.** Compute features with the same code path for both — otherwise you get *training–serving skew*: the model sees slightly different inputs in production and quietly gets worse.
3. **Batch scoring job** after each upload or nightly: load model once, predict all changed dealers in one vectorised call, store scores with the model version.
4. **Online endpoint only if needed:** a small service with the model in memory, behind several workers, returning in milliseconds. Tabular models like logistic regression or gradient boosting don't need GPUs.
5. **Roll out safely:** run the new model in *shadow mode* (scores computed and logged but not shown), compare with the old one, then switch; keep the old version to roll back instantly.
6. **Monitor:** prediction distribution, input drift (are delays creeping up for everyone?), and actual outcomes per score band as they arrive. Retrain on a schedule or when performance drops.

### If LLMs are in the pipeline (e.g. column mapping) — running them efficiently

- **Use it only where rules fail.** Rules handle known formats instantly; the LLM sees only unknown headers.
- **Cache results.** The same header from the same source system gets the same mapping — one LLM call per new format, ever.
- **Batch** several unknown columns into one call instead of one call per column.
- **Smallest model that's accurate enough;** measure on a test set before choosing a bigger one.
- **Constrain output** to a fixed list of fields in a structured format, so parsing never fails and answers can't wander.
- **Timeouts, retries and a fallback** (flag for human review) so a slow or failed call never blocks the import.
- **Run asynchronously** inside the background job, never inside a web request.
- **Track cost and latency per call,** and watch rate limits.

### Back-and-forth

**Q: How do you run your scoring efficiently today?** A: In batch at upload time. One query loads the payments, they're grouped per dealer in memory, both scores are computed with no extra queries, and all results are saved in one bulk update. The dashboard just reads stored scores.

→ **Why not compute scores live when someone opens the page?** A: Scores only change when new data arrives, so recomputing on every view wastes work and slows the page. Precompute on change, read many times. The exception is the time-based part, which a nightly refresh handles.

→ **What if you had a million dealers?** A: Stop loading everything into memory. Recompute only changed dealers, push aggregations into the database or a distributed engine like Spark, process in partitions, and run it as a scheduled background pipeline that writes results to a table the app reads.

→ **How would you make the computation itself faster?** A: Vectorise it — compute delays and weighted medians on whole columns with pandas/NumPy or SQL instead of looping per payment — and parallelise across dealers, since each dealer's score is independent.

**Q: You've trained a model — how do you deploy it?** A: Save it with a version, load it once into a scoring worker, score changed dealers in batches after each upload, store scores with the model version, roll out in shadow mode first, and monitor drift and outcomes. Online serving only if a decision needs a fresh score in real time.

→ **How do you keep latency low for online predictions?** A: Model in memory, features precomputed and fetched by key (not recomputed per request), small model, several workers, and caching for repeated requests.

→ **What's training–serving skew?** A: When features at prediction time are computed differently from training — different code, different cut-off date, missing values handled differently. The model performs worse in production without any error message. Fix: one shared feature pipeline.

→ **How do you know when to retrain?** A: When input distributions drift or the late-payment rate per score band stops lining up with the scores. Also on a regular schedule, since dealer behaviour changes with the economy.

**Q: How would you handle very large files like HiLabs' price-transparency data?** A: Don't load the whole file into memory — stream it in chunks, convert to a columnar format like Parquet, validate and deduplicate per chunk with stable keys, process partitions in parallel, and make every step idempotent so a failed run can restart safely. Same principles as SBM, bigger scale.

→ **What's the most expensive mistake you see in ML systems?** A: Doing heavy work on the request path — loading a model per request, recomputing features live, calling an LLM synchronously. Move work ahead of time or into background jobs.

## 9. Behavioral and ownership questions

These check whether you really owned the work. Use the shape: situation → what *you* did → result → what you learned. Keep each under a minute, and fill the bracketed parts with your real memories.

**Q: What was your exact role? Did you work alone?** A: *(Be precise: "I was the only developer; the owners and accountant gave requirements and feedback" — or name who else contributed.)* Own what you built; credit others for what they decided.

→ **Who designed the score — you or the owner?** A: I designed the structure — two scores, what goes into each, the formulas. The owners set what "good" meant — e.g. that paying within 4 days is excellent — and approved the final tuning. *(Adjust to what actually happened.)*

**Q: Tell me about a time you found a bug in your own work.** A: The tripled totals. A dealer's balance didn't match what the accountant expected. I traced the rows, found Zoho repeats a payment across every invoice it covers, fixed it with the per-invoice amount and a combined ID, and kept a test so it can't recur. Lesson: understand what a report *means* before adding up its numbers.

**Q: Tell me about a time you were wrong.** A: The stress tests. I was happy with the scores until I built extreme dealer profiles and found two unfair results — a reformed dealer who could never recover, and a ₹10 invoice destroying a great dealer's score. I'd designed both flaws. Lesson: try to break your own model before someone else does.

**Q: Did you ever disagree with a stakeholder?** A: *(Use a real one if you have it.)* A natural example: the owner wanted one simple score; I argued for two because one number hid dealers who were valuable but slipping. I showed the 2×2 with real dealers, and once he saw a big dealer he trusted flagged as "slipping", he agreed.

**Q: How did you explain technical things to non-technical owners?** A: Through their own dealers. Instead of formulas, I showed: "this dealer is 763 because they always pay within 4 days and buy a lot; this one dropped to 300 because of 60-day delays recently." The simulator helped — they changed a setting and saw the effect.

**Q: What would you do differently?** A: Record outcomes from day one — which invoices ended up very late or written off — so the score could be tested against reality early. And build background imports and login controls before rollout instead of after.

**Q: What was the biggest challenge — technical or not?** A: Trust. Owners had decided credit from instinct for years. The scores were only used once they matched what the owners already believed about dealers they knew well — and then pointed out things they hadn't noticed. That's why validation against their judgement mattered so much.

**Q: What did you learn that you'll bring to HiLabs?** A: Real data is messier than any dataset you're given in class, the hardest bugs are the ones that produce believable numbers, and a model people don't understand won't be used.

## 10. Linking the internship to HiLabs

HiLabs cleans and makes sense of messy healthcare data — provider directories, payer files, price-transparency rate files — so your internship is the same skill in a different industry. Say the link out loud once; don't force it into every answer.

| At SBM you… | At HiLabs the same problem is… |
| --- | --- |
| Mapped differently named columns from Zoho/Tally reports to one schema | Normalising payer and provider files that each use their own format |
| Caught split payments that tripled totals | Catching duplicated or exploded rates in price-transparency files |
| Built a unique key per row to stop duplicates | Matching the same provider across sources (NPI, tax ID, name, address) |
| Learned an AR-aging row means "still owed", not "paid" | Knowing a field means different things in different file types |
| Scored dealers with no ground-truth labels | Scoring data quality or provider accuracy without perfect labels |
| Made scores explainable to owners | Making outputs explainable to health-plan clients |

**Q: Why does this internship make you a fit for HiLabs?** A: "Most of my internship wasn't the score — it was making unreliable data trustworthy enough to make money decisions on. That's HiLabs' core problem in healthcare. And in your hackathon on provider rates I worked with exactly that kind of data, so I've now done it in both finance and healthcare."

**Q: How is healthcare data harder?** A: Far bigger files, many more sources, strict privacy rules, and errors affect patients and payments. The principles carry over — validate meaning, stable keys, idempotent pipelines, reconcile against trusted totals — but scale and compliance raise the bar.

**Q: What would you want to work on here?** A: Pipelines where ML or LLMs help with messy, unseen formats and entity matching — always with validation around them, which is the lesson I took from SBM.

## 10A. Handling questions this doc doesn't cover

No document can list every follow-up, so this section gives you a way to build answers on the spot. Interviewers grade how you think more than whether you've heard the question before.

### The six-step answer pattern (works for any "what if" question)

1. **Clarify the number or goal.** "100 users doing what — viewing or uploading?" "Losses measured how?" Asking one sharp question shows control.
2. **State the baseline honestly.** "At SBM it was a few staff and \~100 dealers, so I built it this way because…"
3. **Find the bottleneck.** Ask yourself: what gets slow, wrong, or expensive first? (Usually: heavy work on the request path, the database, or bad data.)
4. **Fix in layers, cheapest first.** Make it faster → cache it → move it to the background → add machines → change the architecture.
5. **Name the trade-off.** Every fix costs something: freshness (caching), complexity (queues), money (more servers), explainability (ML).
6. **Say how you'd measure.** Load test, back-test, before/after metric, monitoring dashboard. "I'd confirm it by…" closes every answer strongly.

### Mental models that answer most questions

| If the question is about… | Think… |
| --- | --- |
| Speed / users / load | Convert to requests per second; requests in flight = rate × time per request; slow requests are the enemy |
| Big data | Don't load it all; chunk, stream, push to the database, parallelise independent pieces |
| Correctness | Stable keys, idempotency, transactions, reconcile against a trusted total |
| Models | Batch if possible; load once; same features in training and serving; monitor drift |
| Impact claims | Definition, baseline, rate not raw value, lag, confounders, compare groups |
| ML vs rules | Labels? Data size? Need to explain? |
| LLMs | Only where rules fail; constrain output; validate; cache; human fallback |
| Security | Who can see what; filter every query by company; audit log; no real data in code |

### When you genuinely don't know

Say: "I haven't worked with that directly, but here's how I'd reason about it…" and use the six steps. Never invent a tool, number or result you didn't use. A clear "I didn't do that, but here's how I would" scores far better than a bluff that collapses on the next follow-up.

### When they challenge your answer

Don't defend reflexively. Say "that's a fair point" if it is, explain what you'd change, or explain the constraint you were under ("with 100 dealers and no labels, that was the right trade-off at the time"). Calmly agreeing with a valid criticism is a strong signal.

### Worked example — an unprepared question

**Q: "What if SBM wanted scores updated within one minute of every payment?"**

1. *Clarify:* "Within a minute of the payment being entered in Zoho?"
2. *Baseline:* "Today scores update when staff upload an export — usually daily."
3. *Bottleneck:* "The data doesn't arrive until someone uploads."
4. *Fix:* "Pull from Zoho's API on a schedule or via webhooks, queue each new payment, recompute only that dealer's scores in a background worker."
5. *Trade-off:* "More moving parts and API limits — worth it only if decisions are made within the hour."
6. *Measure:* "Track time from payment to updated score and alert if it exceeds a minute."

That answer took no memorisation — only the pattern.

## 11. Traps, phrases to avoid, and final checklist

Most candidates lose these interviews not on knowledge but on one overclaim that unravels under a follow-up. Admit limits early and calmly — it makes everything else you say more believable.

### Phrases to avoid → say instead

| Avoid | Say instead |
| --- | --- |
| "Streaming / live / sub-second" | "Scores update automatically on every data upload" |
| "Real CIBIL score" / "bureau standard" | "Internal CIBIL-style trade-credit score" |
| "AI-powered scoring" / "ML model" | "Rule-based, explainable scoring — ML is the next step once outcomes are logged" |
| "Loans, interest, EMIs" | "Trade credit and receivables" |
| "The tool cut losses by 50%" (alone) | "After owners used the scores to tighten credit, default-related losses fell about 50% vs the earlier period" |
| "Dealers used the platform" (if they didn't log in) | "Staff used it; dealers felt it through credit limits and statements" |
| "It scales to millions" | "Built for SBM's size; here's what I'd change at 100×" |

### Rapid-fire traps

| Trap | Calm answer |
| --- | --- |
| "So it's just an if-else formula?" | "Yes — deliberately. No labels, 100 dealers, owners needed to explain it. I validated it and know how I'd move to ML." |
| "How many dealers exactly?" | The real number. |
| "How do you know the score works?" | Stress profiles, sanity rules, owner review; back-testing is next. |
| "Where did 50% come from?" | Definition, period, who measured, mechanism (section 4). |
| "What would you have done with more time?" | Back-testing, the two flaw fixes, background imports, alerts. |
| "What did you personally build?" | Pipeline, cleaning fixes, both scores, simulator, dashboard, deployment *(adjust)*. |
| "Can I see the code?" | Only show it if real dealer data has been removed from the repo and its history. |

### Fill these in from memory or the business — before the interview

- [ ] Internship dates and duration
- [ ] Exact number of dealer accounts
- [ ] Rough number of invoices/payments processed
- [ ] How "default-related loss" was defined, the before/after periods, and who measured the 50%
- [ ] How dealers interacted with the platform
- [ ] Before/after time for the weekly or monthly credit review
- [ ] One real dealer decision the owner made using the score
- [ ] One real piece of user feedback
- [ ] Whether the two scoring fixes were shipped
- [ ] Whether the deployment had login and roles

### Practise out loud

- [ ] The 60-second story (section 0) without notes
- [ ] Explain V1 and V2 in plain words, no formula on paper
- [ ] The tripled-totals bug as a story
- [ ] The full 50% chain, staying calm through four follow-ups
- [ ] "Design the ML version" in five steps
- [ ] "Where would you use an LLM" with guardrails
