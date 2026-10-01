# DBMS Placement Numericals & SQL Output Patterns

Oct 1, 2026 · @Divyansh verma

## How to use this doc

These \~45 patterns cover the DBMS numericals and SQL output questions that keep appearing in placement OAs and GATE-style tests. Every answer was checked by code: a brute-force key finder for keys and super keys, a closure-based checker for normal forms and decompositions, and SQLite for every SQL output.

Study order: sections 1–6 are one chain (closure → keys → super keys → FD properties → normal forms → decomposition). Learn closure first; everything after it uses closure. Sections 7–11 are independent.

Conventions: R(A, B, C…) is a relation with single-letter attributes, `AB → C` means "A and B together determine C", and X⁺ is the closure of X. For each pattern, cover the answer, solve it, then check.

## 1. Attribute closure (the master tool)

The closure X⁺ is every attribute X can determine using the FDs. Almost every question in sections 2–6 is answered by computing a closure.

**Method:** start with X⁺ = X. Scan the FDs; whenever an FD's left side is entirely inside X⁺, add its right side. Repeat until nothing new is added.

Q: R(A, B, C, D, E), F = {A → B, B → C, CD → E}. Find A⁺ and (AD)⁺.

- A⁺: start {A}. A → B adds B. B → C adds C. CD → E needs D, which is missing. Stop.
- (AD)⁺: start {A, D}. Add B, then C. Now C and D are both present, so CD → E adds E.

**Answer:** A⁺ = {A, B, C}. (AD)⁺ = {A, B, C, D, E}.

What closure tells you:

- X⁺ = all attributes of R → X is a **super key**.
- Y ⊆ X⁺ → the FD **X → Y holds** (is implied by F).
- Check this for every FD of a second set → the two FD sets are **equivalent** (section 4).

Common slip: an FD fires only when its *whole* left side is in X⁺. CD → E does not fire with C alone.

## 2. Finding candidate keys

A candidate key (CK) is a *minimal* super key: its closure is all of R, and removing any attribute breaks that. The fast method classifies attributes first.

| Where the attribute appears in F | Is it in a CK? |
| --- | --- |
| Never on any right side (only left, or nowhere) | In **every** CK |
| Only on right sides | In **no** CK |
| On both sides | Maybe; try it |

**Method:** take the "in every CK" attributes. If their closure is all of R, that set is the only CK. If not, add "both sides" attributes one at a time and check closures. Then look for **substitutions**: if X is in a key and Y → X holds, swapping X for Y often gives another key.

Prime attribute = belongs to at least one CK. Non-prime = belongs to none. Normal forms (section 5) depend on this split.

### 2.1 One obvious key

Q: R(A, B, C, D), F = {A → B, B → C, C → D}.

A is never on a right side, and A⁺ = {A, B, C, D}.

**Answer:** CK = {A}. Prime: A. Non-prime: B, C, D.

### 2.2 Two must-have attributes

Q: R(A, B, C, D, E), F = {A → B, BC → D, D → E}.

A and C never appear on a right side, so both are in every key. (AC)⁺: A gives B, then BC gives D, D gives E → all of R.

**Answer:** CK = {AC}, the only one. Prime: A, C. Non-prime: B, D, E.

### 2.3 A cycle creates several keys

Q: R(A, B, C, D), F = {A → B, B → C, C → A}.

D is never on a right side, but D⁺ = {D}. A, B and C form a cycle (each determines the next), so any one of them plus D covers everything.

**Answer:** CKs = {AD, BD, CD}. All four attributes are prime; there are **no** non-prime attributes.

### 2.4 Substitution to find the rest

Q: R(A, B, C, D, E, F), F = {AB → C, C → D, D → A, BE → F}.

B and E are never on a right side. (BE)⁺ = {B, E, F}, not enough. Add A: (ABE)⁺ → C → D → all of R, so ABE is a key. Now substitute A: D → A, so try BDE; it works. C → D, so try BCE; it works too.

**Answer:** CKs = {ABE, BDE, BCE}. Prime: A, B, C, D, E. Non-prime: F only.

### 2.5 GFG standard question

Q: R(A, B, C, D, E, F, G, H), F = {CH → G, A → BC, B → CFH, E → A, F → EG}.

D is never on a right side, so D is in every key. (AD)⁺: A gives B, C; B gives F, H; F gives E, G → all of R. Now substitute A: E → A gives ED; F → E gives FD; B → F gives BD.

**Answer:** CKs = {AD, BD, ED, FD}. This exact relation is used again in section 3 (it has 120 super keys).

### 2.6 Every single attribute is a key

Q: R(A, B, C, D, E), F = {A → B, B → C, C → D, D → E, E → A}.

The FDs form one big cycle, so each attribute alone reaches everything.

**Answer:** 5 CKs: {A}, {B}, {C}, {D}, {E}. Trap: students often stop after finding {A}.

## 3. Counting super keys

A super key is any set that contains at least one candidate key. So counting super keys means counting the subsets of R that contain some CK. Find the CKs first (section 2); the counting is then pure formula.

**One CK with k attributes** (n attributes in R): the k key attributes are fixed, and each of the other n − k attributes can be in or out.

```latex
\#SK = 2^{\,n-k}
```

**Several CKs:** inclusion–exclusion. The "overlap" term for two keys K₁ and K₂ counts sets containing *both*, which is 2 raised to (n minus the size of K₁ ∪ K₂).

```latex
\#SK = |S_1| + |S_2| - |S_1 \cap S_2| \qquad |S_1 \cap S_2| = 2^{\,n - |K_1 \cup K_2|}
```

### 3.1 Practice set (all verified by brute force)

| R | Candidate keys | Working | Super keys |
| --- | --- | --- | --- |
| ABCDE | A | 2⁴ | **16** |
| ABCDE | AB | 2³ | **8** |
| ABCD | A, B | 2³ + 2³ − 2² | **12** |
| ABCD | A, BC | 2³ + 2² − 2¹ (A∪BC = ABC) | **10** |
| ABCDE | AB, CD | 2³ + 2³ − 2¹ (AB∪CD = ABCD) | **14** |
| ABCD | A, B, C | 3·2³ − 3·2² + 2¹ | **14** |
| ABCD | AD, BD, CD (section 2.3) | 3·2² − 3·2¹ + 2⁰ | **7** |
| ABCDE | A, B, C, D, E (section 2.6) | every non-empty subset | **31** |

Trap in the A, BC row: the overlap is 2^(4 − 3), because A ∪ BC = {A, B, C} has 3 attributes. Students often subtract 2^(4 − 2) by mistake.

### 3.2 The GFG 120 question

Q: R(A … H), 8 attributes, with CKs AD, BD, ED, FD (section 2.5). How many super keys?

Every key has 2 attributes and all four share D, so any union of j keys has j + 1 attributes.

```latex
4 \cdot 2^{6} - 6 \cdot 2^{5} + 4 \cdot 2^{4} - 1 \cdot 2^{3} = 256 - 192 + 64 - 8 = 120
```

**Answer:** 120. The coefficients 4, 6, 4, 1 are just C(4,1), C(4,2), C(4,3), C(4,4).

### 3.3 Maximum counts

- Maximum super keys for n attributes: **2ⁿ − 1** (every non-empty subset; happens when every single attribute is a CK).
- Maximum candidate keys for n attributes: **C(n, ⌊n/2⌋)**, when every set of ⌊n/2⌋ attributes is a key. For n = 4, that's C(4, 2) = 6.

## 4. Functional dependency properties

OAs test FD properties in four ways: name the rule, decide if an FD is implied, compare two FD sets, and find a minimal cover. All four reduce to closure.

### 4.1 The rules

| Rule | Statement | Type |
| --- | --- | --- |
| Reflexivity | If Y ⊆ X, then X → Y (trivial FD) | Armstrong axiom |
| Augmentation | If X → Y, then XZ → YZ | Armstrong axiom |
| Transitivity | If X → Y and Y → Z, then X → Z | Armstrong axiom |
| Union | If X → Y and X → Z, then X → YZ | Derived |
| Decomposition | If X → YZ, then X → Y and X → Z | Derived |
| Pseudo-transitivity | If X → Y and WY → Z, then WX → Z | Derived |
| Composition | If X → Y and Z → W, then XZ → YW | Derived |

Armstrong's three axioms are **sound** (they never derive a false FD) and **complete** (they derive every FD that follows). The other four are proved from them.

Two traps that MCQs love:

- Decomposition works only on the **right** side. AB → C does **not** give A → C or B → C.
- FDs are one-way. A → B does **not** give B → A.

### 4.2 Is this FD implied?

Rule: X → Y is implied by F exactly when Y ⊆ X⁺ (computed using F).

Q: F = {A → B, B → C}. Which of these hold?

| FD | Closure check | Holds? |
| --- | --- | --- |
| A → C | A⁺ = {A, B, C} contains C | Yes (transitivity) |
| AC → B | (AC)⁺ = {A, B, C} contains B | Yes |
| AB → C | (AB)⁺ = {A, B, C} contains C | Yes |
| C → A | C⁺ = {C} | No |
| B → A | B⁺ = {B, C} | No |

### 4.3 Are two FD sets equivalent?

F and G are equivalent when every FD of F is implied by G **and** every FD of G is implied by F. Check both directions; one direction alone is a common mistake.

Q (GFG standard): F = {A → C, AC → D, E → AD, E → H}, G = {A → CD, E → AH}. Equivalent?

- F from G: under G, A⁺ = {A, C, D} and E⁺ = {E, A, H, C, D}. That covers A → C, AC → D, E → AD and E → H.
- G from F: under F, A⁺ = {A, C, D} and E⁺ = {E, A, D, H, C}. That covers A → CD and E → AH.

**Answer:** yes, equivalent.

### 4.4 Minimal (canonical) cover

Three steps, always in this order:

1. Split every right side into single attributes.
2. Remove extraneous attributes from left sides: in XY → Z, drop Y if X⁺ already contains Z.
3. Remove redundant FDs: drop X → Z if Z is still in X⁺ computed *without* that FD.

Q: Find the minimal cover of F = {A → BC, B → C, A → B, AB → C}.

1. Split: A → B, A → C, B → C, A → B (duplicate, drop), AB → C.
2. In AB → C, A⁺ = {A, B, C} already contains C, so B is extraneous. AB → C becomes A → C (now a duplicate).
3. A → C is redundant: without it, A → B and B → C still give C.

**Answer:** {A → B, B → C}. Verified equivalent to the original F.

## 5. Identifying the highest normal form

"What is the highest normal form of R?" is the most common DBMS numerical. Find the CKs, mark prime attributes, then test every FD against three rules.

For each non-trivial FD X → A (split right sides into single attributes):

| Normal form | The FD passes if… | Violation is called |
| --- | --- | --- |
| 2NF | NOT (X is a proper part of a CK **and** A is non-prime) | Partial dependency |
| 3NF | X is a super key, **or** A is prime | Transitive dependency |
| BCNF | X is a super key | Any non-key determinant |

The relation's highest NF is the highest one that **every** FD passes. Assume 1NF (atomic values) unless the question shows multi-valued cells.

### 5.1 Practice set (all verified by code)

| # | R and F | CKs | Failing FD | Highest NF |
| --- | --- | --- | --- | --- |
| a | R(ABCD): AB → C, B → D | AB | B → D: B is part of the key, D non-prime | **1NF** |
| b | R(ABCD): A → B, B → C, C → D | A | B → C: B not a super key, C non-prime | **2NF** |
| c | R(ABCDE): AB → C, C → D, D → E | AB | C → D: transitive, but nothing partial | **2NF** |
| d | R(ABC): AB → C, C → B | AB, AC | C → B: C not a super key, but B is prime | **3NF** |
| e | R(ABCD): A → BCD, BC → A | A, BC | none; both left sides are keys | **BCNF** |

Worked through (d), the classic "3NF but not BCNF" case: from AB → C and C → B, the keys are AB and AC, so A, B and C are all prime. C → B fails BCNF because C alone is not a super key. It passes 3NF because B is prime.

### 5.2 Shortcuts that save time

- Every CK is a single attribute → no partial dependency possible → at least **2NF**.
- Every attribute is prime → 3NF's second condition always holds → at least **3NF**.
- Any relation with exactly two attributes is always in **BCNF**.
- BCNF ⊂ 3NF ⊂ 2NF ⊂ 1NF: a relation in BCNF is automatically in all the lower forms.

## 6. Decomposition: lossless join and dependency preservation

When normalisation splits R into smaller tables, two questions follow: can the original be rebuilt exactly (lossless), and are all FDs still checkable (dependency preserving)? Lossless is mandatory; preservation is desirable.

### 6.1 Lossless join test (binary decomposition)

R split into R1 and R2 is lossless when the common attributes form a super key of at least one side:

```latex
(R_1 \cap R_2) \to R_1 \quad \text{or} \quad (R_1 \cap R_2) \to R_2
```

Also required: R1 ∪ R2 = R (no attribute dropped).

Q: R(A, B, C), F = {A → B}. Check two decompositions.

| Decomposition | Common | Closure of common | Lossless? |
| --- | --- | --- | --- |
| R1(A, B), R2(A, C) | A | A⁺ = {A, B} covers all of R1 | **Yes** |
| R1(A, B), R2(B, C) | B | B⁺ = {B} covers neither | **No (lossy)** |

Lossy means joining R1 and R2 back produces **extra, spurious tuples**, not fewer. That's a common MCQ trap.

### 6.2 Dependency preservation

A decomposition preserves dependencies when every original FD can be checked using the FDs that fall *inside* the smaller tables (their projections), without joining.

Q: R(A, B, C, D), F = {A → B, B → C, C → D}. Compare two decompositions.

| Decomposition | Lossless? | FDs inside the tables | All of F preserved? |
| --- | --- | --- | --- |
| (A, B), (B, C), (C, D) | Yes | A → B, B → C, C → D | **Yes** |
| (A, B), (A, C), (A, D) | Yes (A is the key) | A → B, A → C, A → D | **No**: B → C and C → D are lost |

In the second one, no single table contains both B and C, so B → C can't be checked without a join.

### 6.3 BCNF vs 3NF: the trade-off question

Q: R(A, B, C), F = {AB → C, C → B} (the 3NF-but-not-BCNF relation from section 5). Decompose into BCNF.

C → B violates BCNF, so split on it: R1(C, B) and R2(A, C). Common attribute C, and C → B makes C a key of R1, so the split is lossless. But AB → C now spans two tables and is lost.

**Answer:** R1(C, B), R2(A, C). Lossless, but **not** dependency preserving.

The fact to remember:

| Target | Lossless always possible? | Dependency preservation always possible? |
| --- | --- | --- |
| 3NF | Yes | Yes |
| BCNF | Yes | **No** |

## 7. Relations and joins: counting rows

Row-count questions come in two kinds: exact counts on given tables, and minimum/maximum counts from sizes alone. Sections 7 and 8 use these two tables.

**emp** (degree 4, cardinality 5)

| id | name | dept\_id | salary |
| --- | --- | --- | --- |
| 1 | Asha | 10 | 50000 |
| 2 | Ravi | 20 | 60000 |
| 3 | Neha | 10 | NULL |
| 4 | Karan | NULL | 40000 |
| 5 | Meera | 30 | 60000 |

**dept** (degree 2, cardinality 3)

| id | dname |
| --- | --- |
| 10 | HR |
| 20 | Tech |
| 40 | Sales |

Degree = number of columns. Cardinality = number of rows (your notes, Lec 7).

### 7.1 Exact counts, joining on emp.dept\_id = dept.id

| Join | Rows | Why |
| --- | --- | --- |
| INNER | **3** | Asha, Neha (HR) and Ravi (Tech) match. Karan (NULL) and Meera (30) don't. |
| LEFT (emp left) | **5** | All emp rows; Karan and Meera get NULLs for dept columns. |
| RIGHT (dept right) | **4** | HR matches 2 rows, Tech 1, Sales none but still appears once. |
| FULL OUTER | **6** | 3 matched + Karan, Meera (left only) + Sales (right only). |
| CROSS | **15** | 5 × 3, degree 4 + 2 = 6 columns. |

Trap: NULL never matches anything, not even another NULL, so Karan never joins.

### 7.2 Minimum and maximum from sizes alone

R has m rows and S has n rows (both at least 1).

| Join | Minimum rows | Maximum rows |
| --- | --- | --- |
| CROSS | m × n | m × n (always exact) |
| INNER / natural | 0 | m × n (every pair matches) |
| LEFT (R left) | m | m × n |
| RIGHT (S right) | n | m × n |
| FULL OUTER | max(m, n) | max(m × n, m + n) |

- Q: R has 10 rows, S has 5. Max rows in R ⋈ S? **Answer:** 50.
- Q: R(…, sid) has 10 rows, and sid is a NOT NULL foreign key to S's primary key. Rows in R ⋈ S on sid? **Answer:** exactly 10. Each R row matches exactly one S row.
- Q: Natural join of two tables with **no** common column? **Answer:** it becomes a cross join, m × n rows.

### 7.3 Counting from domains

Q: R(A, B), where A can take 3 values and B can take 2. (a) Max tuples in R? (b) How many different relation instances are possible?

(a) Every combination: 3 × 2 = **6** tuples. (b) Each of those 6 tuples is either present or absent: 2⁶ = **64** instances (including the empty relation).

## 8. SQL output questions

OAs show a query on a small table and ask for the output. Most traps are about NULL. All outputs below were run in SQLite on the emp and dept tables from section 7.

### 8.1 Aggregates and NULL

```sql
SELECT COUNT(*), COUNT(salary), COUNT(dept_id), SUM(salary), AVG(salary) FROM emp;
```

**Output:** 5, 4, 4, 210000, 52500.

- `COUNT(*)` counts rows (5). `COUNT(col)` skips NULLs (4).
- `AVG(salary)` = 210000 / **4**, not / 5. Aggregates ignore NULL rows entirely.
- `COUNT(DISTINCT salary)` = 3 (40000, 50000, 60000; NULL not counted).

### 8.2 Comparing with NULL

| Query (on emp) | Output | Why |
| --- | --- | --- |
| `WHERE dept_id = NULL` | no rows | `= NULL` is never true; it evaluates to UNKNOWN |
| `WHERE dept_id IS NULL` | Karan | the correct way to test for NULL |
| `WHERE dept_id NOT IN (10)` | Ravi, Meera | Karan is excluded too: `NULL NOT IN (10)` is UNKNOWN |
| `WHERE dept_id NOT IN (10, NULL)` | **no rows** | the most famous trap; see below |

Why the last one is empty: `x NOT IN (10, NULL)` means `x <> 10 AND x <> NULL`. The second part is always UNKNOWN, so the whole condition is never true. The same happens when a NOT IN **subquery** returns any NULL. Use `NOT EXISTS` instead; it isn't affected by NULLs.

### 8.3 GROUP BY and HAVING

```sql
SELECT dept_id, COUNT(*) FROM emp GROUP BY dept_id;
```

**Output:** 4 groups: (NULL, 1), (10, 2), (20, 1), (30, 1). NULLs form **one group** of their own, even though NULL ≠ NULL in comparisons.

```sql
SELECT dept_id, COUNT(*) FROM emp GROUP BY dept_id HAVING COUNT(*) > 1;
```

**Output:** (10, 2). HAVING filters groups after grouping; WHERE filters rows before grouping. An aggregate in WHERE (`WHERE COUNT(*) > 1`) is an error.

### 8.4 Second (Nth) highest salary

The most asked SQL question in interviews. Two standard answers:

```sql
-- Method 1: max below the max
SELECT MAX(salary) FROM emp
WHERE salary < (SELECT MAX(salary) FROM emp);

-- Method 2: Nth highest (here N = 2), skip N-1 distinct values
SELECT DISTINCT salary FROM emp
WHERE salary IS NOT NULL
ORDER BY salary DESC
LIMIT 1 OFFSET 1;
```

**Output:** 50000 for both. Two people earn 60000, so without `DISTINCT`, method 2 would wrongly return 60000. That duplicate is the trap. `DENSE_RANK()` is a third method interviewers accept.

### 8.5 UNION vs UNION ALL

```sql
SELECT dept_id FROM emp UNION SELECT id FROM dept;      -- 5 rows
SELECT dept_id FROM emp UNION ALL SELECT id FROM dept;  -- 8 rows
```

UNION ALL keeps all 5 + 3 = 8 values. UNION removes duplicates, leaving 10, 20, 30, 40 and NULL, so 5. UNION treats the NULL as a single distinct value.

### 8.6 Departments with no employees

```sql
SELECT d.dname FROM dept d
WHERE NOT EXISTS (SELECT 1 FROM emp e WHERE e.dept_id = d.id);
```

**Output:** Sales. This is a correlated subquery (your notes, Lec 9): it runs once per dept row. The LEFT JOIN version, `WHERE e.id IS NULL`, gives the same result.

## 9. ER diagram to minimum number of tables

"Minimum number of tables needed for this ER diagram" applies the Lec 8 conversion rules and counts. The trick is knowing which relationships need **no** table of their own.

| ER element | Tables it adds | Notes |
| --- | --- | --- |
| Strong entity | 1 | Its PK becomes the table's PK |
| M:N relationship | 1 | PK = PKs of both entities |
| 1:N relationship | 0 | Put the "1" side's PK as an FK in the "N" side |
| 1:1 relationship | 0 | FK in either side (prefer the total-participation side) |
| 1:1 with total participation on **both** sides | merges 2 entities into 1 | Saves a table |
| Multivalued attribute | 1 | PK = entity's PK + the attribute |
| Weak entity | 1 | PK = owner's PK + partial key; its identifying relationship adds 0 |
| Composite attribute | 0 | Flattened into separate columns |
| Derived attribute | 0 | Not stored (e.g. age from DOB) |

### 9.1 Practice set

| # | ER diagram | Tables | Minimum |
| --- | --- | --- | --- |
| a | E1 —M:N— E2 | E1, E2, R | **3** |
| b | E1 —1:N— E2 | E1, E2 (FK in E2) | **2** |
| c | E1 —1:1— E2, total participation on both sides | merged E1E2 | **1** |
| d | Student (multivalued phone) —M:N— Course; each Course taught by one Professor (N:1) | Student, StudentPhone, Course (FK prof\_id), Professor, Enrolls | **5** |
| e | Customer —M:N— Loan; weak entity Payment owned by Loan | Customer, Loan, Borrower, Payment | **4** |

Trap in (d): the Course–Professor relationship adds no table; the multivalued phone does. In (e): the identifying relationship between Loan and Payment adds no table.

## 10. Transactions: schedules and serializability

Your notes cover ACID and transaction states; tests add schedule counting and serializability checks. Notation: R1(A) = T1 reads A, W2(A) = T2 writes A, C1 = T1 commits.

### 10.1 Counting schedules

```latex
\text{serial schedules} = n! \qquad \text{all schedules of } T_1, T_2 = \frac{(n_1 + n_2)!}{n_1!\, n_2!}
```

n₁ and n₂ are the operation counts; each transaction's own order is fixed, so we only choose where its operations go.

- Q: 3 transactions. Serial schedules? **Answer:** 3! = 6.
- Q: T1 has 3 operations, T2 has 2. Total schedules? 5! / (3! × 2!) = **10**. Non-serial ones? 10 − 2 = **8**.
- Three transactions extend the same way: (n₁ + n₂ + n₃)! / (n₁! n₂! n₃!).

### 10.2 Conflict serializability (precedence graph)

Two operations **conflict** when they are from different transactions, on the same item, and at least one is a write (RW, WR or WW; RR never conflicts).

**Method:** for each conflicting pair, draw an edge from the transaction that acts first to the one that acts second. **No cycle** → conflict serializable, and any topological order of the graph is an equivalent serial schedule. **Cycle** → not conflict serializable.

Q: S1 = R1(A), W2(A), R3(B), W1(B), W3(A).

| Conflicting pair | Edge |
| --- | --- |
| R1(A) … W2(A) | T1 → T2 |
| R1(A) … W3(A) | T1 → T3 |
| W2(A) … W3(A) | T2 → T3 |
| R3(B) … W1(B) | T3 → T1 |

**Answer:** T1 → T3 → T1 is a cycle, so S1 is **not** conflict serializable.

Q: S2 = R1(A), R2(B), W1(A), R3(A), W2(B), W3(A). Is it serializable, and how many equivalent serial orders?

Only T1 and T3 conflict (all on A, T1 first): edge T1 → T3. T2 touches only B, which nobody else uses.

**Answer:** serializable. Equivalent serial orders = topological orders with T1 before T3: T1 T2 T3, T1 T3 T2, T2 T1 T3, so **3**.

Every conflict-serializable schedule is also view serializable. A schedule that is view serializable but *not* conflict serializable always contains a **blind write** (a write with no read before it).

### 10.3 Recoverability

Q: S = W1(A), R2(A), C2, C1. Recoverable?

T2 reads A written by T1 (a dirty read), then commits before T1 does. If T1 now aborts, T2 has committed using a value that never officially existed.

**Answer:** **irrecoverable**. It becomes recoverable if T2 commits after T1 (W1(A), R2(A), C1, C2). It's **cascadeless** if T2 reads A only after C1 (no dirty reads at all).

| Schedule type | Rule |
| --- | --- |
| Recoverable | A transaction commits only after every transaction it read from has committed |
| Cascadeless | Reads happen only from committed transactions |
| Strict | No read **or write** of an item until the last transaction that wrote it commits or aborts |

Strict ⊂ cascadeless ⊂ recoverable.

## 11. Indexing numericals

One standard problem (Elmasri/Navathe style) covers everything in Lec 14: blocking factor, sparse vs dense index size, and block accesses. Always round blocks **up** and the blocking factor **down**.

```latex
\text{bfr} = \left\lfloor \frac{\text{block size}}{\text{record size}} \right\rfloor \qquad \text{blocks} = \left\lceil \frac{\text{records}}{\text{bfr}} \right\rceil \qquad \text{binary search} = \lceil \log_2(\text{blocks}) \rceil
```

**Setup:** 30,000 records, each 100 bytes, block size 1,024 bytes. Index entry = 9-byte key + 6-byte block pointer = 15 bytes.

### 11.1 Data file

bfr = ⌊1024 / 100⌋ = 10 records per block. Blocks = 30,000 / 10 = **3,000**.

- Linear search (unsorted file): average 3,000 / 2 = **1,500** block accesses; worst case 3,000.
- Binary search (file sorted on the key): ⌈log₂ 3000⌉ = **12** accesses.

### 11.2 Primary (sparse) index on the sorted key

One entry per **data block** (your notes: entries = number of blocks). Index bfr = ⌊1024 / 15⌋ = 68.

Index blocks = ⌈3000 / 68⌉ = **45**. Binary search on the index = ⌈log₂ 45⌉ = 6, plus 1 access for the data block = **7** accesses (vs 12 without the index).

### 11.3 Secondary (dense) index on an unsorted field

One entry per **record**: 30,000 entries. Index blocks = ⌈30000 / 68⌉ = **442**. Accesses = ⌈log₂ 442⌉ + 1 = 9 + 1 = **10** (vs 1,500 average for linear search).

### 11.4 Multi-level index

Each index block holds 68 entries, so 68 is the **fan-out**. Build levels until one block remains: 45 blocks → ⌈45 / 68⌉ = 1 block. That's 2 levels.

**Answer:** 2 index accesses + 1 data access = **3** accesses. Multi-level index cost = (number of levels) + 1.

### 11.5 Order of a B+ tree node

Q: Block 1,024 bytes, key 9 bytes, block pointer 6 bytes. Find the order p (max pointers) of an internal node.

A node with p pointers holds p − 1 keys, and it must fit in one block:

```latex
6p + 9(p - 1) \le 1024 \;\Rightarrow\; 15p \le 1033 \;\Rightarrow\; p = 68
```

**Answer:** p = 68 (uses 1,011 of 1,024 bytes). Note: for leaf nodes the formula differs (each key carries a record pointer, plus one next-leaf pointer), so read which node the question asks about.

## 12. Last-minute sheet

Every formula and trap from above, in one place.

| Topic | Formula or rule | Trap |
| --- | --- | --- |
| Closure | Add an FD's right side once its whole left side is in X⁺ | CD → E needs both C and D |
| Candidate keys | Never-on-right attributes are in every CK; substitute via Y → X for more keys | Cycles create several keys; don't stop at the first |
| Prime attributes | Belongs to at least one CK | All attributes prime → automatically 3NF |
| Super keys, one CK | 2^(n − k) | n = all attributes, k = key size |
| Super keys, several CKs | Inclusion–exclusion; overlap = 2^(n − \|K₁ ∪ K₂\|) | Use the size of the **union**, not the sum |
| Max counts | Super keys 2ⁿ − 1; CKs C(n, ⌊n/2⌋) |  |
| FD implied? | X → Y holds iff Y ⊆ X⁺ | AB → C does not give A → C |
| FD sets equivalent | Check F from G **and** G from F | One direction isn't enough |
| Minimal cover | Split right sides → trim left sides → drop redundant FDs | Do the steps in that order |
| 2NF | No non-prime depends on part of a CK | Single-attribute CKs → automatically 2NF |
| 3NF | X is a super key or A is prime |  |
| BCNF | X is a super key | 2-attribute relation → always BCNF |
| Lossless | Common attributes are a key of one side | Lossy = extra spurious tuples |
| Dependency preserving | Every FD checkable inside one table | BCNF may lose FDs; 3NF never has to |
| Join rows | Cross m × n; inner 0 to m × n; left ≥ m; full ≥ max(m, n) | NULL never matches, even NULL |
| Aggregates | COUNT(\*) counts rows; others skip NULL | AVG divides by non-NULL count |
| NOT IN | Returns nothing if the list or subquery has a NULL | Use NOT EXISTS |
| GROUP BY | NULLs form one group; HAVING filters groups | Aggregate in WHERE is an error |
| Nth highest | DISTINCT + ORDER BY DESC + LIMIT 1 OFFSET N − 1 | Duplicates without DISTINCT |
| ER → tables | Entity, M:N, multivalued, weak entity each add 1; 1:1 and 1:N add 0 | 1:1 with total participation on both sides merges |
| Schedules | Serial n!; all (n₁ + n₂)! / (n₁! n₂!) | Non-serial = all − serial |
| Conflict serializable | Precedence graph has no cycle | RR pairs never conflict |
| Recoverable | Commit after the transactions you read from | Dirty read + early commit = irrecoverable |
| Indexing | bfr = ⌊B / R⌋; blocks = ⌈r / bfr⌉; sparse index = 1 entry per block, dense = 1 per record | Index access + 1 for the data block |
| B+ tree order | p·(pointer) + (p − 1)·(key) ≤ block size | Leaf nodes use a different formula |
