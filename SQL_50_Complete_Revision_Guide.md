# LeetCode SQL 50 — Complete Master Revision Guide

> **Purpose:** This document is designed as an **all-in-one SQL revision doc**. Read through this guide before any SQL interview or coding exam to refresh **core syntax**, **execution order**, **JOIN patterns**, **window functions**, **date/string manipulation**, and **all 50 LeetCode SQL 50 solutions with concept triggers**.

---

## Table of Contents
1. [SQL Quick-Reference & Fundamentals](#part-1-sql-quick-reference--fundamentals)
   - [1. Order of Execution (Crucial for Interviews)](#1-order-of-execution)
   - [2. Three-Valued Logic & NULL Handling](#2-three-valued-logic--null-handling)
   - [3. JOIN Cheat Sheet & Visual Guide](#3-join-cheat-sheet--visual-guide)
   - [4. Essential SQL Functions Toolkit](#4-essential-sql-functions-toolkit)
2. [LeetCode SQL 50 — All Modules & Problems](#part-2-leetcode-sql-50--all-modules--problems)
   - [Module 1: Basic Select (#1 – #5)](#module-1-basic-select)
   - [Module 2: Basic Joins (#6 – #14)](#module-2-basic-joins)
   - [Module 3: Basic Aggregate Functions (#15 – #22)](#module-3-basic-aggregate-functions)
   - [Module 4: Sorting and Grouping (#23 – #29)](#module-4-sorting-and-grouping)
   - [Module 5: Advanced Select and Joins (#30 – #35)](#module-5-advanced-select-and-joins)
   - [Module 6: Subqueries (#36 – #42)](#module-6-subqueries)
   - [Module 7: Advanced String Functions / Regex / Clause (#43 – #50)](#module-7-advanced-string-functions--regex--clause)
3. [Top 10 SQL Interview Pitfalls & Quick Checklist](#part-3-top-10-sql-interview-pitfalls--quick-checklist)

---

## PART 1: SQL Quick-Reference & Fundamentals

### 1. Order of Execution
Understanding the logical execution order of a SQL query is the #1 way to avoid syntax errors (e.g., using column aliases in `WHERE`).

```
1. FROM / JOIN      ──> Load & merge base tables
2. WHERE            ──> Filter individual rows (before grouping)
3. GROUP BY         ──> Group rows by key columns
4. HAVING           ──> Filter aggregated groups
5. SELECT           ──> Compute output columns & expressions
6. DISTINCT         ──> Remove duplicate rows
7. ORDER BY         ──> Sort the final result set
8. LIMIT / OFFSET   ──> Return a subset of rows
```

> [!IMPORTANT]
> - **Why you cannot use `SELECT` aliases in `WHERE`:** `WHERE` runs **before** `SELECT`.
> - **`WHERE` vs `HAVING`:** Use `WHERE` for individual row filtering; use `HAVING` for filtering results of aggregates (`COUNT`, `SUM`, `AVG`).

---

### 2. Three-Valued Logic & NULL Handling
In SQL, any comparison with `NULL` returns `UNKNOWN` (neither `TRUE` nor `FALSE`).
- `col = NULL` ❌ Always evaluates to `UNKNOWN`.
- `col IS NULL` / `col IS NOT NULL` ✅ Correct way to test for NULL.
- `col != 2` ⚠️ **Will drop rows where `col IS NULL`!** Always use `col != 2 OR col IS NULL` if you want to include NULLs.

| Function | What It Does | Example usage |
| :--- | :--- | :--- |
| `COALESCE(val1, val2, ...)` | Returns the first non-null value | `COALESCE(bonus, 0)` |
| `IFNULL(val, default)` | MySQL specific; replaces `NULL` with default | `IFNULL(price, 0)` |
| `IF(cond, true_val, false_val)` | Inline conditional (like ternary `? :`) | `IF(state='approved', 1, 0)` |
| `CASE WHEN ... THEN ... END` | Multi-branch conditional logic | `CASE WHEN age > 60 THEN 'Senior' ELSE 'Adult' END` |

---

### 3. JOIN Cheat Sheet & Visual Guide
- **`INNER JOIN`:** Returns rows that have matching values in **both** tables.
- **`LEFT JOIN`:** Returns **all** rows from the left table, and matched rows from the right table (`NULL` if no match).
- **`RIGHT JOIN`:** Returns **all** rows from the right table, and matched rows from the left table.
- **`CROSS JOIN`:** Cartesian product (every row of Table A paired with every row of Table B).
- **Self-Join:** Joining a table to itself (useful for hierarchy, consecutive days, comparing pairs).
- **Anti-Join:** Finding unmatched rows using `LEFT JOIN ... WHERE right_table.id IS NULL`.

---

### 4. Essential SQL Functions Toolkit

#### Date & Time (MySQL)
- `DATEDIFF(date1, date2)`: Returns difference in days (`date1 - date2`).
  * **Example:** `DATEDIFF('2023-10-15', '2023-10-10')` ➔ `5` (useful for finding consecutive days).
- `DATE_ADD(date1, INTERVAL 1 DAY)` / `DATE_SUB(...)`: Add or subtract time intervals.
  * **Example:** `DATE_ADD('2023-10-10', INTERVAL 1 MONTH)` ➔ `'2023-11-10'`.
- `DATE_FORMAT(date_col, '%Y-%m')`: Formats date as string.
  * **Example:** `DATE_FORMAT('2020-02-15', '%Y-%m')` ➔ `'2020-02'` (great for grouping by month).
- `YEAR(date_col)`, `MONTH(date_col)`: Extracts numeric year or month.
  * **Example:** `YEAR('2020-02-15')` ➔ `2020`.

#### String Manipulation
- `CONCAT(str1, str2, ...)`: Join strings together.
  * **Example:** `CONCAT('Leet', 'Code')` ➔ `'LeetCode'`.
- `SUBSTRING(str, start_idx, length)`: Extract substring (**1-indexed** in SQL!).
  * **Example:** `SUBSTRING('SQL is fun', 1, 3)` ➔ `'SQL'`.
- `UPPER(str)` / `LOWER(str)`: Case conversion.
  * **Example:** `UPPER('john')` ➔ `'JOHN'`.
- `CHAR_LENGTH(str)`: Number of characters in a string.
  * **Example:** `CHAR_LENGTH('Tweet')` ➔ `5`.
- `GROUP_CONCAT(col ORDER BY col SEPARATOR ',')`: Combines multiple rows of text into a single comma-separated string.
  * **Example:** `GROUP_CONCAT(product_name SEPARATOR ', ')` ➔ `'Apple, Banana, Orange'`.

#### Window Functions

The most critical analytical tools in SQL. They perform calculations across a set of table rows that are related to the current row, without collapsing them into a single output row (unlike `GROUP BY`).

| Function | What it does | Example Use Case |
| :--- | :--- | :--- |
| 🔢 **`ROW_NUMBER()`** | Gives every row a unique sequential integer. | `ROW_NUMBER() OVER(ORDER BY salary DESC)` ➔ `1, 2, 3, 4` |
| 🥈 **`RANK()`** | Ranks rows, leaving **gaps** after ties. | `RANK() OVER(ORDER BY salary DESC)` ➔ `1, 2, 2, 4` |
| 🥇 **`DENSE_RANK()`** | Ranks rows, leaving **no gaps** after ties. | `DENSE_RANK() OVER(ORDER BY salary DESC)` ➔ `1, 2, 2, 3` |
| ⏪ **`LAG()`** | Looks at the **previous** row's value. | Finding the difference in sales from yesterday to today. |
| ⏩ **`LEAD()`** | Looks at the **next** row's value. | Checking if the next login date is exactly 1 day after the current one. |
| 📈 **`SUM()`** | Calculates a **running/cumulative sum**. | `SUM(revenue) OVER(ORDER BY date)` ➔ YTD Revenue. |
| 📊 **`AVG()`** | Calculates a **moving/partition average**. | `AVG(price) OVER(ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)` ➔ 7-day moving average. |
| 🧮 **`COUNT()`** | **Counts** within a specific window. | Counting the number of transactions per user without dropping row-level details. |
| 🔼 **`MAX()`** | Finds the **maximum** within a window. | Finding the highest ever score for a player up to their current game. |
| 🔽 **`MIN()`** | Finds the **minimum** within a window. | Finding the absolute first order date for a customer (`MIN(date) OVER(PARTITION BY user)`). |
### 💡 The Ultimate Window Function Mental Model

Whenever you see: `XXX(...) OVER (...)`
Read it aloud as: *"For this row, calculate `XXX` using this specific window of rows."*

- `AVG(salary) OVER (PARTITION BY department)` ➔ *"For this employee, calculate the average salary of everyone in their department."*
- `LAG(salary) OVER (ORDER BY id)` ➔ *"For this employee, give me the salary of the previous employee."*
- `RANK() OVER (PARTITION BY department ORDER BY salary DESC)` ➔ *"For this employee, tell me their salary rank within their department."*

#### Window Function vs GROUP BY (The Most Important Distinction)

**`GROUP BY` collapses rows.**
```sql
SELECT department, MAX(salary) FROM Employee GROUP BY department;
```
Result: The individual employees vanish. You just get `IT ➔ 700`, `HR ➔ 900`.

**Window Functions keep rows intact.**
```sql
SELECT employee, department, salary, MAX(salary) OVER (PARTITION BY department) AS max_dept_salary FROM Employee;
```
Result: The rows remain! You get `A, IT, 500, (700)` and `B, IT, 700, (700)`. It simply *adds* aggregate information to the existing rows.

#### The Anatomy of `OVER()`

The `OVER()` clause is where the magic happens. It has two main sub-clauses:

1. **`PARTITION BY` (The Grouping Mechanism)**
   - Defines which rows belong together. It's like `GROUP BY`, but without collapsing them.
   - Example: `PARTITION BY department` means "calculate this metric separately for each department."

2. **`ORDER BY` (The Sorting Mechanism)**
   - Defines the order of rows *inside* that window.
   - Essential for functions that depend on sequence, like `LAG()`, `LEAD()`, or running totals (`SUM()`).
   - Example: `ORDER BY date` means "look at the previous row according to the timeline."

#### Essential Memory Tricks & Patterns

**1. The Ranking Trio:**
- `ROW_NUMBER` ➔ Everyone gets a unique number (1, 2, 3, 4).
- `RANK` ➔ Ties share rank, **gaps appear** (1, 2, 2, 4).
- `DENSE_RANK` ➔ Ties share rank, **no gaps** (1, 2, 2, 3).

**2. The Running Total Pattern:**
```sql
-- The window keeps expanding: Jan 1 (100) -> Jan 1+2 (300) -> Jan 1+2+3 (600)
SUM(amount) OVER (ORDER BY date) AS running_total
```

**3. The "Top N per Category" Pattern (Crucial for LeetCode):**
```sql
-- Combine PARTITION BY + ORDER BY to rank items within categories
RANK() OVER (PARTITION BY department ORDER BY salary DESC)
```
---

## PART 2: LeetCode SQL 50 — All Modules & Problems

---

### Module 1: Basic Select

#### 1. Recyclable and Low Fat Products (1757)
- **The 'Why':** We use the `AND` operator to strictly filter rows that satisfy both conditions simultaneously.
- **Execution Order:** `FROM` (loads the Products table) ➔ `WHERE` (filters rows row-by-row) ➔ `SELECT` (projects the `product_id` column).
- **Edge Cases:** Empty table returns an empty set. If a product has a `NULL` value for `low_fats` or `recyclable`, the `='Y'` comparison evaluates to `UNKNOWN` (not `TRUE`), and the row is safely ignored.
- **Performance:** Requires a sequential scan by default. To optimize for read-heavy systems, a composite index on `(low_fats, recyclable, product_id)` allows an **Index-Only Scan**.
- **Interviewer Follow-up:** *"If we stored these flags as `BOOLEAN` (1 and 0) instead of `VARCHAR` ('Y' and 'N'), how would that impact storage size and index performance at a scale of 10 billion rows?"*
```sql
SELECT product_id 
FROM Products 
WHERE low_fats = 'Y' AND recyclable = 'Y';
```

#### 2. Find Customer Referee (584)
- **The 'Why':** Tests understanding of **Three-Valued Logic**. `NULL` means "unknown," so `NULL != 2` evaluates to `UNKNOWN`, dropping the row. We *must* explicitly use `OR referee_id IS NULL`.
- **Execution Order:** `FROM` ➔ `WHERE` (evaluates `!= 2` first, then `IS NULL`) ➔ `SELECT`.
- **Edge Cases:** If the entire `referee_id` column is populated only with `NULL`s, the query simply returns every customer name.
- **Performance:** The `OR` operator, particularly combined with `IS NULL`, often prevents the query optimizer from using B-Tree indexes effectively, leading to a full table scan.
- **Interviewer Follow-up:** *"If this table had 500 million rows, this `OR` clause might cause a full table scan. How could you rewrite this query to ensure the database can utilize indexes optimally?" (Hint: `UNION ALL`)*.
```sql
SELECT name 
FROM Customer 
WHERE referee_id != 2 OR referee_id IS NULL;
```

#### 3. Big Countries (595)
- **The 'Why':** We evaluate two distinct criteria using the `OR` operator. If a row satisfies *at least one* condition, it gets returned.
- **Execution Order:** `FROM` ➔ `WHERE` ➔ `SELECT`.
- **Edge Cases:** If a country has a `NULL` population but an area of `4000000`, it will still be returned because `UNKNOWN OR TRUE` evaluates to `TRUE`.
- **Performance:** Using `OR` across two different columns is a performance killer. Even with separate indexes on `area` and `population`, the optimizer might still choose a full table scan.
- **Interviewer Follow-up:** *"Write a mathematically equivalent query using `UNION` instead of `OR`, and explain why `UNION` might execute faster if we have individual indexes on both columns."*
```sql
SELECT name, population, area 
FROM World 
WHERE area >= 3000000 OR population >= 25000000;
```

#### 4. Article Views I (1148)
- **The 'Why':** Self-column comparison (`author_id = viewer_id`) finds authors viewing their own work. `DISTINCT` ensures an author's ID is only listed once even if they view their article multiple times.
- **Execution Order:** `FROM` ➔ `WHERE` (author = viewer) ➔ `SELECT` (aliases author_id to id) ➔ `DISTINCT` (deduplicates) ➔ `ORDER BY` (sorts the final list).
- **Edge Cases:** If `author_id` or `viewer_id` is `NULL`, `NULL = NULL` evaluates to `UNKNOWN` and the row is excluded (correct behavior).
- **Performance:** `DISTINCT` is a heavy operation requiring hashing or sorting. Using `GROUP BY author_id` behaves the exact same way but sometimes maps better to aggregate indexes depending on the engine.
- **Interviewer Follow-up:** *"Explain the time and space complexity difference between how a database processes `DISTINCT` using a Hash set versus sorting."*
```sql
SELECT DISTINCT author_id AS id 
FROM Views 
WHERE author_id = viewer_id 
ORDER BY id ASC;
```

#### 5. Invalid Tweets (1683)
- **The 'Why':** Tests string functions. We use `CHAR_LENGTH()` (counts characters) instead of `LENGTH()` (counts bytes) to avoid false positives with emojis or multi-byte characters.
- **Execution Order:** `FROM` ➔ `WHERE` (computes character length dynamically) ➔ `SELECT`.
- **Edge Cases:** If `content` is `NULL`, `CHAR_LENGTH()` returns `NULL`, the condition `> 15` becomes `UNKNOWN`, and the row is skipped. Empty strings `''` return `0`.
- **Performance:** Applying a function directly to a column in the `WHERE` clause violates **Sargability**, blinding the database to indexes and guaranteeing a full table scan.
- **Interviewer Follow-up:** *"Since we cannot use indexes when applying functions to columns, how would you redesign the schema so we can instantly look up invalid tweets without doing a full table scan every time?"*
```sql
SELECT tweet_id 
FROM Tweets 
WHERE CHAR_LENGTH(content) > 15;
```

---

### Module 2: Basic Joins

#### 6. Replace Employee ID With The Unique Identifier (1378)
- **The 'Why':** We need all employees even if they lack a unique ID, so a `LEFT JOIN` on the primary `Employees` table is required. An `INNER JOIN` would drop employees without a unique ID.
- **Execution Order:** `FROM` (loads Employees) ➔ `LEFT JOIN` (matches with EmployeeUNI) ➔ `SELECT`.
- **Edge Cases:** If `EmployeeUNI` is empty, all unique_id values will simply be `NULL`. If `Employees` is empty, the result is empty.
- **Performance:** `LEFT JOIN`s are efficient if the joined column (`id`) is indexed in the right table (`EmployeeUNI`).
- **Interviewer Follow-up:** *"What happens if there are duplicate `id`s in the `EmployeeUNI` table? How does that affect row count?"*
```sql
SELECT eu.unique_id, e.name
FROM Employees e
LEFT JOIN EmployeeUNI eu ON e.id = eu.id;
```

#### 7. Product Sales Analysis I (1068)
- **The 'Why':** We only want sales records that have matching product info. An `INNER JOIN` securely links fact data (Sales) with dimension data (Product).
- **Execution Order:** `FROM` ➔ `JOIN` (matches on product_id) ➔ `SELECT`.
- **Edge Cases:** Sales with `product_id`s not present in the `Product` table are dropped.
- **Performance:** The engine will likely use a Hash Join or Nested Loop Join. Ensure `product_id` is indexed in `Product`.
- **Interviewer Follow-up:** *"If we needed to find Sales that had NO matching Product (data anomaly), how would you change this query?" (Answer: LEFT JOIN ... WHERE product.id IS NULL)*
```sql
SELECT p.product_name, s.year, s.price
FROM Sales s
JOIN Product p ON s.product_id = p.product_id;
```

#### 8. Customer Who Visited but Did Not Make Any Transactions (1581)
- **The 'Why':** This is the classic **Anti-Join pattern**. We `LEFT JOIN` transactions to visits and filter where `transaction_id IS NULL` to isolate visits with zero transactions.
- **Execution Order:** `FROM` ➔ `LEFT JOIN` ➔ `WHERE` (filters to NULLs) ➔ `GROUP BY` (aggregates per customer) ➔ `SELECT`.
- **Edge Cases:** If all visits have transactions, the result is an empty set.
- **Performance:** Anti-joins using `LEFT JOIN + IS NULL` are highly optimized by modern SQL engines compared to `NOT IN` (which struggles with NULLs).
- **Interviewer Follow-up:** *"Write this using `NOT EXISTS` instead of a `LEFT JOIN`, and explain when one might be faster than the other."*
```sql
SELECT v.customer_id, COUNT(v.visit_id) AS count_no_trans
FROM Visits v
LEFT JOIN Transactions t ON v.visit_id = t.visit_id
WHERE t.transaction_id IS NULL
GROUP BY v.customer_id;
```

#### 9. Rising Temperature (197)
- **The 'Why':** A **Self-Join with Date Math**. We join the table to itself offset by 1 day using `DATEDIFF()` to guarantee we strictly compare consecutive calendar days, not just consecutive IDs.
- **Execution Order:** `FROM` ➔ `JOIN` (matches tomorrow to today) ➔ `WHERE` (checks temperature) ➔ `SELECT`.
- **Edge Cases:** Missing dates (gaps in records) won't match, which is correct.
- **Performance:** `DATEDIFF(w1.recordDate, w2.recordDate) = 1` prevents index usage on dates. `w1.recordDate = DATE_ADD(w2.recordDate, INTERVAL 1 DAY)` is sargable and allows index seeks.
- **Interviewer Follow-up:** *"Rewrite the join condition to be sargable so we can utilize a B-Tree index on `recordDate`."*
```sql
SELECT w1.id
FROM Weather w1
JOIN Weather w2 ON DATEDIFF(w1.recordDate, w2.recordDate) = 1
WHERE w1.temperature > w2.temperature;
```

#### 10. Average Time of Process per Machine (1661)
- **The 'Why':** Self-Join pairs `'start'` activity rows with `'end'` activity rows on the same machine/process to compute durations before averaging.
- **Execution Order:** `FROM` ➔ `JOIN` (on machine_id, process_id, and start/end logic) ➔ `GROUP BY` (machine_id) ➔ `SELECT` (calculates AVG and ROUND).
- **Edge Cases:** If a process starts but never ends, the join condition fails, implicitly dropping orphaned events.
- **Performance:** Self-joining a massive activity log is expensive. Window functions (`LEAD` or conditional aggregation) often perform better by requiring only one scan.
- **Interviewer Follow-up:** *"Solve this in a single table scan using a `CASE WHEN` inside a `SUM()` instead of a self-join."*
```sql
SELECT a1.machine_id, ROUND(AVG(a2.timestamp - a1.timestamp), 3) AS processing_time
FROM Activity a1
JOIN Activity a2 
  ON a1.machine_id = a2.machine_id 
  AND a1.process_id = a2.process_id 
  AND a1.activity_type = 'start' 
  AND a2.activity_type = 'end'
GROUP BY a1.machine_id;
```

#### 11. Employee Bonus (577)
- **The 'Why':** We need all employees even without bonuses (`LEFT JOIN`). Filtering by `bonus < 1000` drops NULLs, so we explicitly add `OR b.bonus IS NULL`.
- **Execution Order:** `FROM` ➔ `LEFT JOIN` ➔ `WHERE` (handles condition and NULLs) ➔ `SELECT`.
- **Edge Cases:** Employees with exactly 1000 are excluded.
- **Performance:** Simple hash/merge join. `OR IS NULL` forces scanning of the filtered join results.
- **Interviewer Follow-up:** *"How would using an `INNER JOIN` subtly break the business requirements of this query?"*
```sql
SELECT e.name, b.bonus
FROM Employee e
LEFT JOIN Bonus b ON e.empId = b.empId
WHERE b.bonus < 1000 OR b.bonus IS NULL;
```

#### 12. Students and Examinations (1280)
- **The 'Why':** A `CROSS JOIN` creates a matrix of *all* students and *all* subjects. A `LEFT JOIN` onto the exams table accurately counts 0 for students who missed an exam.
- **Execution Order:** `FROM` ➔ `CROSS JOIN` (builds the grid) ➔ `LEFT JOIN` (attaches exams) ➔ `GROUP BY` ➔ `SELECT` (COUNT ignores NULLs) ➔ `ORDER BY`.
- **Edge Cases:** Empty exams table still outputs the cross-joined grid with `0` for all counts.
- **Performance:** `CROSS JOIN` explodes data volume (N * M rows). Always group and filter as early as possible on huge datasets.
- **Interviewer Follow-up:** *"Why do we use `COUNT(e.subject_name)` instead of `COUNT(*)` in the `SELECT` clause here?" (Answer: COUNT(*) would return 1 for a NULL join result, COUNT(col) returns 0).*
```sql
SELECT s.student_id, s.student_name, sub.subject_name, COUNT(e.subject_name) AS attended_exams
FROM Students s
CROSS JOIN Subjects sub
LEFT JOIN Examinations e 
  ON s.student_id = e.student_id 
  AND sub.subject_name = e.subject_name
GROUP BY s.student_id, s.student_name, sub.subject_name
ORDER BY s.student_id, sub.subject_name;
```

#### 13. Managers with at Least 5 Direct Reports (570)
- **The 'Why':** We identify manager IDs using `GROUP BY ... HAVING COUNT >= 5` in a subquery, then filter the main table using `IN` to get their names.
- **Execution Order:** Subquery (`FROM` ➔ `GROUP BY` ➔ `HAVING` ➔ `SELECT`) ➔ Outer Query (`FROM` ➔ `WHERE IN` ➔ `SELECT`).
- **Edge Cases:** If an employee manages themselves (bad data), they count towards the 5.
- **Performance:** The engine will likely optimize the `IN` subquery into a join (`Semi-Join`).
- **Interviewer Follow-up:** *"Rewrite this without a subquery by using a `JOIN` and a `GROUP BY` on the outer level. What happens to the `GROUP BY` clause?"*
```sql
SELECT name 
FROM Employee 
WHERE id IN (
    SELECT managerId 
    FROM Employee 
    GROUP BY managerId 
    HAVING COUNT(*) >= 5
);
```

#### 14. Confirmation Rate (1934)
- **The 'Why':** **Conditional Aggregation**. `c.action = 'confirmed'` yields 1 or 0 in MySQL, allowing `AVG()` to calculate the percentage. `COALESCE` handles users with no requests (returns 0 instead of NULL).
- **Execution Order:** `FROM` ➔ `LEFT JOIN` ➔ `GROUP BY` ➔ `SELECT` (calculates AVG, COALESCE, and ROUND).
- **Edge Cases:** Users with 0 signups are kept due to the `LEFT JOIN`. Their `AVG` is `NULL`, which `COALESCE` gracefully turns to `0`.
- **Performance:** Computing conditional logic inside an aggregate is efficient as it happens in a single scan.
- **Interviewer Follow-up:** *"In PostgreSQL, you can't sum/avg a boolean condition directly. How would you write this using a `FILTER` clause or `CASE WHEN`?"*
```sql
SELECT s.user_id, ROUND(COALESCE(AVG(c.action = 'confirmed'), 0), 2) AS confirmation_rate
FROM Signups s
LEFT JOIN Confirmations c ON s.user_id = c.user_id
GROUP BY s.user_id;
```

---

### Module 3: Basic Aggregate Functions

#### 15. Not Boring Movies (620)
- **The 'Why':** Modulo operator (`%`) efficiently identifies odd numbers, while `!=` eliminates the specific string.
- **Execution Order:** `FROM` ➔ `WHERE` ➔ `SELECT` ➔ `ORDER BY`.
- **Edge Cases:** If all movie IDs are even, the result is empty. If description is `NULL`, `!= 'boring'` is `UNKNOWN` and excluded.
- **Performance:** Modulo in the `WHERE` clause prevents index seeks. For a large table, you'd want a separate boolean column for `is_odd_id` or `is_boring` if this query runs frequently.
- **Interviewer Follow-up:** *"What is the difference between `!=` and `<>` in SQL?" (Answer: Functionally identical in most dialects, but `<>` is the ISO standard).*
```sql
SELECT * 
FROM cinema 
WHERE id % 2 = 1 AND description != 'boring' 
ORDER BY rating DESC;
```

#### 16. Average Selling Price (1251)
- **The 'Why':** We need a weighted average `SUM(price * units) / SUM(units)`. A `LEFT JOIN` combined with `BETWEEN` accurately ties a unit sold to the price active on that specific date.
- **Execution Order:** `FROM` ➔ `LEFT JOIN` (with date bounds) ➔ `GROUP BY` ➔ `SELECT` (calculates SUMs and IFNULL).
- **Edge Cases:** If a product hasn't sold any units, the join yields `NULL` for units. `IFNULL(..., 0)` safely catches the division by NULL/zero.
- **Performance:** `BETWEEN` joins can be slow. Ensuring a composite index on `UnitsSold(product_id, purchase_date)` is critical.
- **Interviewer Follow-up:** *"Why use `IFNULL` outside the aggregate instead of `COALESCE` inside the `SUM()`?"*
```sql
SELECT p.product_id, IFNULL(ROUND(SUM(p.price * u.units) / SUM(u.units), 2), 0) AS average_price
FROM Prices p
LEFT JOIN UnitsSold u 
  ON p.product_id = u.product_id 
  AND u.purchase_date BETWEEN p.start_date AND p.end_date
GROUP BY p.product_id;
```

#### 17. Project Employees I (1075)
- **The 'Why':** Standard `INNER JOIN` linking projects to employees to calculate a simple mathematical average using `AVG()`.
- **Execution Order:** `FROM` ➔ `JOIN` ➔ `GROUP BY` ➔ `SELECT`.
- **Edge Cases:** If a project has no employees (or only employees with `NULL` experience), `AVG` returns `NULL`.
- **Performance:** Very efficient hash join. Grouping by a primary/foreign key is highly optimized.
- **Interviewer Follow-up:** *"If we wanted the median experience years instead of the average, how would you calculate that?" (Hint: `PERCENTILE_CONT` or window functions).*
```sql
SELECT project_id, ROUND(AVG(experience_years), 2) AS average_years
FROM Project p
JOIN Employee e ON p.employee_id = e.employee_id
GROUP BY project_id;
```

#### 18. Percentage of Users Attended a Contest (1633)
- **The 'Why':** We divide the group count by a global scalar subquery `(SELECT COUNT(*) FROM Users)` to find the percentage.
- **Execution Order:** Global Subquery evaluates first ➔ Main Query `FROM` ➔ `GROUP BY` ➔ `SELECT` (calculates ratio) ➔ `ORDER BY`.
- **Edge Cases:** If the `Users` table is completely empty, it throws a divide-by-zero error.
- **Performance:** The engine executes the scalar subquery exactly once, caching the result, making this O(N) complexity overall.
- **Interviewer Follow-up:** *"Can you write this using Window Functions (`COUNT() OVER()`) instead of a scalar subquery?"*
```sql
SELECT contest_id, ROUND(COUNT(user_id) * 100.0 / (SELECT COUNT(*) FROM Users), 2) AS percentage
FROM Register
GROUP BY contest_id
ORDER BY percentage DESC, contest_id ASC;
```

#### 19. Queries Quality and Percentage (1211)
- **The 'Why':** We use multiple conditional aggregations in a single `SELECT`. `rating < 3` resolves to 1/0, allowing `AVG()` to calculate the percentage of poor queries directly.
- **Execution Order:** `FROM` ➔ `WHERE` (removes null queries) ➔ `GROUP BY` ➔ `SELECT`.
- **Edge Cases:** If `position` is 0 (bad data), `rating / position` throws a division by zero error.
- **Performance:** Doing multiple aggregations inside a single `GROUP BY` requires only one pass over the data, which is highly optimal.
- **Interviewer Follow-up:** *"How would you handle a potential divide-by-zero if `position` could be 0?" (Answer: `NULLIF(position, 0)`).*
```sql
SELECT query_name, 
       ROUND(AVG(rating / position), 2) AS quality, 
       ROUND(AVG(rating < 3) * 100, 2) AS poor_query_percentage
FROM Queries
WHERE query_name IS NOT NULL
GROUP BY query_name;
```

#### 20. Monthly Transactions I (1193)
- **The 'Why':** Multi-dimensional grouping. We extract the `'YYYY-MM'` from a full timestamp using `DATE_FORMAT` and group by that + country. `IF(state = 'approved', val, 0)` is used for conditional summation.
- **Execution Order:** `FROM` ➔ `GROUP BY` (evaluates DATE_FORMAT dynamically) ➔ `SELECT`.
- **Edge Cases:** Transactions with `NULL` country will be grouped together into a single `NULL` country bucket.
- **Performance:** `DATE_FORMAT` in the `GROUP BY` clause is not indexable. In production, it's better to maintain a `month_id` column.
- **Interviewer Follow-up:** *"What is the PostgreSQL or SQL Server equivalent of `IF()` for this query?" (Answer: `CASE WHEN state = 'approved' THEN ... END`).*
```sql
SELECT DATE_FORMAT(trans_date, '%Y-%m') AS month,
       country,
       COUNT(id) AS trans_count,
       SUM(IF(state = 'approved', 1, 0)) AS approved_count,
       SUM(amount) AS trans_total_amount,
       SUM(IF(state = 'approved', amount, 0)) AS approved_total_amount
FROM Transactions
GROUP BY month, country;
```

#### 21. Immediate Food Delivery II (1174)
- **The 'Why':** We use a **Tuple Subquery** `(customer_id, order_date) IN (...)` to isolate the absolute first order for each customer, then apply conditional aggregation on those filtered rows.
- **Execution Order:** Subquery (`FROM` ➔ `GROUP BY` ➔ `SELECT MIN()`) ➔ Main query (`FROM` ➔ `WHERE IN` ➔ `SELECT AVG()`).
- **Edge Cases:** If a customer places two orders on their very first day (same date), both are evaluated.
- **Performance:** Tuple `IN` queries can be slow in some older MySQL versions. A `ROW_NUMBER() OVER(PARTITION BY customer_id ORDER BY order_date)` is the modern, more performant alternative.
- **Interviewer Follow-up:** *"Rewrite this query using `ROW_NUMBER()` and explain why it might be faster than the tuple subquery."*
```sql
SELECT ROUND(AVG(order_date = customer_pref_delivery_date) * 100, 2) AS immediate_percentage
FROM Delivery
WHERE (customer_id, order_date) IN (
    SELECT customer_id, MIN(order_date) 
    FROM Delivery 
    GROUP BY customer_id
);
```

#### 22. Game Play Analysis IV (550)
- **The 'Why':** We find each player's first login date via a grouped subquery, then `LEFT JOIN` back to the activity table looking for `event_date = first_login + 1 DAY`.
- **Execution Order:** Subquery ➔ `LEFT JOIN` ➔ `SELECT` (COUNTs the matches vs the total).
- **Edge Cases:** Players who only logged in once will yield `NULL` from the `LEFT JOIN`, correctly making `COUNT(t2.player_id)` equal 0.
- **Performance:** Calculating `DATE_ADD()` inside the join condition is fine here because `first_login` is a computed scalar per user.
- **Interviewer Follow-up:** *"If we wanted to find players who logged in for 3 consecutive days instead of 2, how would you structure the query?" (Hint: `LEAD()` window function).*
```sql
SELECT ROUND(COUNT(t2.player_id) / COUNT(t1.player_id), 2) AS fraction
FROM (
    SELECT player_id, MIN(event_date) AS first_login
    FROM Activity
    GROUP BY player_id
) t1
LEFT JOIN Activity t2 
  ON t1.player_id = t2.player_id 
  AND t2.event_date = DATE_ADD(t1.first_login, INTERVAL 1 DAY);
```

---

### Module 4: Sorting and Grouping

#### 23. Number of Unique Subjects Taught by Each Teacher (2356)
- **The 'Why':** We use `COUNT(DISTINCT subject_id)` to find unique subjects per teacher. A simple `COUNT()` would overcount if a teacher teaches the same subject in multiple semesters.
- **Execution Order:** `FROM` ➔ `GROUP BY` ➔ `SELECT` (evaluates COUNT DISTINCT).
- **Edge Cases:** If a teacher teaches 0 subjects, they won't appear in the `Teacher` table at all.
- **Performance:** `COUNT(DISTINCT)` is memory-intensive because it requires maintaining a hash set of seen values per group.
- **Interviewer Follow-up:** *"If the table has billions of rows, `COUNT(DISTINCT)` can cause Out of Memory errors. How would you approximate this in a big data warehouse like Snowflake or BigQuery?" (Answer: `APPROX_COUNT_DISTINCT` or HyperLogLog).*
```sql
SELECT teacher_id, COUNT(DISTINCT subject_id) AS cnt
FROM Teacher
GROUP BY teacher_id;
```

#### 24. User Activity for the Past 30 Days I (1141)
- **The 'Why':** We filter for a specific 30-day window using `BETWEEN` and count unique active users per day.
- **Execution Order:** `FROM` ➔ `WHERE` (date filter) ➔ `GROUP BY` (date) ➔ `SELECT` (counts users).
- **Edge Cases:** Dates with 0 active users will not appear in the final output because there are no rows to group.
- **Performance:** Using `BETWEEN` on an indexed `activity_date` column allows a fast range scan.
- **Interviewer Follow-up:** *"How would you alter this query so that dates with 0 active users STILL show up in the output with a count of 0?" (Hint: Requires a calendar dimension table and a `LEFT JOIN`).*
```sql
SELECT activity_date AS day, COUNT(DISTINCT user_id) AS active_users
FROM Activity
WHERE activity_date BETWEEN '2019-06-28' AND '2019-07-27'
GROUP BY activity_date;
```

#### 25. Product Sales Analysis III (1070)
- **The 'Why':** We use a Tuple `IN` subquery `(product_id, year) IN (...)` to fetch the complete row (including quantity and price) for the minimum year per product.
- **Execution Order:** Subquery (`FROM` ➔ `GROUP BY` ➔ `SELECT MIN()`) ➔ Main Query (`FROM` ➔ `WHERE IN` ➔ `SELECT`).
- **Edge Cases:** If a product was sold multiple times in its first year, all those records are returned (a tie on the minimum year).
- **Performance:** Tuple matching can force sub-optimal execution plans. Window functions like `RANK()` are standard for "top-N per group" queries.
- **Interviewer Follow-up:** *"Rewrite this query using the `RANK()` window function to achieve the same result."*
```sql
SELECT product_id, year AS first_year, quantity, price
FROM Sales
WHERE (product_id, year) IN (
    SELECT product_id, MIN(year) 
    FROM Sales 
    GROUP BY product_id
);
```

#### 26. Classes More Than 5 Students (596)
- **The 'Why':** We use the `HAVING` clause to filter out classes after they have been grouped and counted. `WHERE` cannot be used with aggregate functions.
- **Execution Order:** `FROM` ➔ `GROUP BY` ➔ `HAVING` (filters groups >= 5) ➔ `SELECT`.
- **Edge Cases:** If a class has exactly 5 students, it is included (`>= 5`).
- **Performance:** Grouping and then filtering is standard. Ensure an index on `class` exists to speed up the `GROUP BY`.
- **Interviewer Follow-up:** *"What if a student accidentally enrolled in the same class twice in the table? How would you fix the query to only count unique students?" (Answer: `HAVING COUNT(DISTINCT student) >= 5`).*
```sql
SELECT class
FROM Courses
GROUP BY class
HAVING COUNT(student) >= 5;
```

#### 27. Find Followers Count (1729)
- **The 'Why':** Standard aggregation counting followers grouped by the user they follow, sorted ascending.
- **Execution Order:** `FROM` ➔ `GROUP BY` ➔ `SELECT` ➔ `ORDER BY`.
- **Edge Cases:** Users with 0 followers won't be in the table, so they won't appear in the output.
- **Performance:** `COUNT(follower_id)` skips NULLs. If `follower_id` cannot be NULL, `COUNT(*)` is slightly faster.
- **Interviewer Follow-up:** *"If we needed users with 0 followers to show up as well, which additional table would we need, and what kind of join would we use?"*
```sql
SELECT user_id, COUNT(follower_id) AS followers_count
FROM Followers
GROUP BY user_id
ORDER BY user_id;
```

#### 28. Biggest Single Number (619)
- **The 'Why':** We find all numbers that appear exactly once using `HAVING COUNT(num) = 1`. We wrap it in `SELECT MAX(num)` so that if the subquery returns an empty set, the outer query evaluates to `NULL` instead of nothing.
- **Execution Order:** Subquery (`FROM` ➔ `GROUP BY` ➔ `HAVING` ➔ `SELECT`) ➔ Outer Query (`SELECT MAX`).
- **Edge Cases:** If every number appears multiple times, the subquery is empty, and the outer query cleanly returns `NULL`.
- **Performance:** The subquery requires a full table scan and aggregation. The outer `MAX()` is trivial.
- **Interviewer Follow-up:** *"Without using an outer `SELECT MAX()`, how could you rewrite this using `ORDER BY` and `LIMIT` while still returning `NULL` if no rows are found?" (Hint: `IFNULL` or `COALESCE` with a subquery).*
```sql
SELECT MAX(num) AS num
FROM (
    SELECT num 
    FROM MyNumbers 
    GROUP BY num 
    HAVING COUNT(num) = 1
) AS single_nums;
```

#### 29. Customers Who Bought All Products (1045)
- **The 'Why':** This is **Relational Division**. We group by customer and check if their distinct count of purchased products equals the total count of products in the dimension table.
- **Execution Order:** Subquery evaluates total products ➔ Main Query `FROM` ➔ `GROUP BY` ➔ `HAVING` (compares counts) ➔ `SELECT`.
- **Edge Cases:** If there are 0 products in the `Product` table, customers who bought nothing might technically match depending on how the DB evaluates `0 = 0`.
- **Performance:** The scalar subquery `(SELECT COUNT(*) FROM Product)` is executed once and cached. The `COUNT(DISTINCT)` is the bottleneck.
- **Interviewer Follow-up:** *"If this was a banking app, and we wanted to find users who triggered EVERY type of fraud alert, how would this exact same SQL pattern apply?"*
```sql
SELECT customer_id
FROM Customer
GROUP BY customer_id
HAVING COUNT(DISTINCT product_key) = (SELECT COUNT(*) FROM Product);
```

---

### Module 5: Advanced Select and Joins

#### 30. The Number of Employees Which Report to Each Employee (1731)
- **The 'Why':** We use a **Hierarchy Self-Join**. We alias the same table as `mgr` (manager) and `emp` (employee) and join them where `mgr.employee_id = emp.reports_to`.
- **Execution Order:** `FROM` ➔ `JOIN` ➔ `GROUP BY` (manager_id) ➔ `SELECT` (counts reports, averages age).
- **Edge Cases:** If an employee has no reports, they won't appear as a `mgr` because the `INNER JOIN` eliminates them.
- **Performance:** Self-joins on large tables can be expensive. Ensuring an index exists on `reports_to` is critical for scaling this hierarchy query.
- **Interviewer Follow-up:** *"If we wanted to include ALL employees, even those with 0 reports, how would you change this query?" (Answer: Use a `LEFT JOIN` from `mgr` to `emp`).*
```sql
SELECT mgr.employee_id, mgr.name, 
       COUNT(emp.employee_id) AS reports_count, 
       ROUND(AVG(emp.age)) AS average_age
FROM Employees mgr
JOIN Employees emp ON mgr.employee_id = emp.reports_to
GROUP BY mgr.employee_id, mgr.name
ORDER BY mgr.employee_id;
```

#### 31. Triangle Judgement (610)
- **The 'Why':** We use the `IF()` function (or `CASE WHEN`) to evaluate the Triangle Inequality Theorem directly within the `SELECT` projection.
- **Execution Order:** `FROM` ➔ `SELECT` (evaluates inline logic).
- **Edge Cases:** Negative side lengths or `0` will correctly fail the `>` conditions and return `'No'`.
- **Performance:** This is an O(N) sequential scan. There's no filtering (`WHERE`), so no index is needed or used.
- **Interviewer Follow-up:** *"Can you write this using a `CASE WHEN` statement instead of `IF()` so it complies with standard ANSI SQL?"*
```sql
SELECT x, y, z, 
       IF(x + y > z AND x + z > y AND y + z > x, 'Yes', 'No') AS triangle
FROM Triangle;
```

#### 32. Consecutive Numbers (180)
- **The 'Why':** We use `LEAD(col, 1)` and `LEAD(col, 2)` to look ahead in the partition and compare the current row's number with the next two.
- **Execution Order:** Subquery (`FROM` ➔ `SELECT` with Window Functions) ➔ Outer Query (`FROM` ➔ `WHERE` checks equality ➔ `SELECT DISTINCT`).
- **Edge Cases:** If the table has fewer than 3 rows, `LEAD` returns `NULL`, the `WHERE` clause fails, and it returns an empty set.
- **Performance:** Window functions require sorting the dataset (implicit `OVER()`). If the table is large, this sort is the bottleneck.
- **Interviewer Follow-up:** *"What if we needed to find 10 consecutive numbers? Writing 9 `LEAD` statements is messy. How would you solve it using a gaps-and-islands approach (e.g., `ROW_NUMBER()`)?"*
```sql
SELECT DISTINCT num AS ConsecutiveNums
FROM (
    SELECT num, 
           LEAD(num, 1) OVER() AS num1, 
           LEAD(num, 2) OVER() AS num2
    FROM Logs
) t
WHERE num = num1 AND num = num2;
```

#### 33. Product Price at Given Date (1164)
- **The 'Why':** We combine two logic branches via `UNION`. Branch 1 finds the latest price before the date using a tuple subquery. Branch 2 handles products that had no price change before the date (defaults to 10).
- **Execution Order:** Branch 1 executes ➔ Branch 2 executes ➔ `UNION` deduplicates and merges them.
- **Edge Cases:** A product that only has price changes *after* the target date falls exclusively into the second query and gets the default price of 10.
- **Performance:** Using `UNION` requires deduplication, which is slow. `UNION ALL` is faster here because the two branches are mutually exclusive.
- **Interviewer Follow-up:** *"Rewrite this query using the `ROW_NUMBER()` window function partitioned by `product_id` instead of using a `UNION`."*
```sql
SELECT product_id, new_price AS price
FROM Products
WHERE (product_id, change_date) IN (
    SELECT product_id, MAX(change_date) 
    FROM Products 
    WHERE change_date <= '2019-08-16' 
    GROUP BY product_id
)
UNION
SELECT DISTINCT product_id, 10 AS price
FROM Products
WHERE product_id NOT IN (
    SELECT product_id 
    FROM Products 
    WHERE change_date <= '2019-08-16'
);
```

#### 34. Last Person to Fit in the Bus (1204)
- **The 'Why':** This uses a **Cumulative Sum Window Function** `SUM(weight) OVER (ORDER BY turn)` to calculate a running total row by row.
- **Execution Order:** Subquery (`FROM` ➔ `SELECT` computes running sum) ➔ Outer Query (`FROM` ➔ `WHERE` filters `<= 1000` ➔ `ORDER BY` DESC ➔ `LIMIT 1`).
- **Edge Cases:** If the very first person weighs more than 1000, the outer `WHERE` clause drops them, and the query returns empty.
- **Performance:** Calculating a running sum requires the engine to maintain a rolling accumulator over a sorted data stream. An index on `turn` makes the sort essentially free.
- **Interviewer Follow-up:** *"How would you write this cumulative sum in an older version of MySQL (e.g., v5.7) that doesn't support Window Functions?" (Answer: Correlated Subquery or Self-Join).*
```sql
SELECT person_name
FROM (
    SELECT person_name, SUM(weight) OVER (ORDER BY turn) AS running_weight
    FROM Queue
) t
WHERE running_weight <= 1000
ORDER BY running_weight DESC
LIMIT 1;
```

#### 35. Count Salary Categories (1907)
- **The 'Why':** We need all 3 categories to appear in the output, even if they have 0 accounts. A standard `GROUP BY` drops empty categories. By `UNION`ing three independent static strings, we guarantee the categories exist.
- **Execution Order:** Query 1 executes ➔ Query 2 executes ➔ Query 3 executes ➔ `UNION` merges them.
- **Edge Cases:** If `Accounts` is completely empty, the `SUM(condition)` evaluates to `NULL`. We might need to wrap it in `COALESCE(..., 0)` depending on the SQL dialect.
- **Performance:** Scanning the `Accounts` table three separate times is horribly inefficient.
- **Interviewer Follow-up:** *"Since scanning the table 3 times is bad for performance, rewrite this to scan the table only ONCE by using a `LEFT JOIN` against a hardcoded temporary table of categories."*
```sql
SELECT 'Low Salary' AS category, SUM(salary < 20000) AS accounts_count FROM Accounts
UNION
SELECT 'Average Salary', SUM(salary >= 20000 AND salary <= 50000) FROM Accounts
UNION
SELECT 'High Salary', SUM(salary > 50000) FROM Accounts;
```

---

### Module 6: Subqueries

#### 36. Employees Whose Manager Left the Company (1978)
- **The 'Why':** We use a `NOT IN` subquery to filter out employees whose manager still exists in the `Employees` table.
- **Execution Order:** Subquery (`FROM` ➔ `SELECT`) ➔ Outer Query (`FROM` ➔ `WHERE` checks salary and NOT IN ➔ `ORDER BY`).
- **Edge Cases:** If `manager_id` contains a `NULL`, `NOT IN` behaves dangerously (it will return empty if there is a NULL in the subquery result). It is safer to use `NOT EXISTS`.
- **Performance:** `NOT IN` can be slow if the subquery returns many rows and is not optimized. A `LEFT JOIN` where the right side is `NULL` is generally safer and faster.
- **Interviewer Follow-up:** *"Why is `NOT IN` dangerous when the subquery might return a `NULL` value? How would you rewrite this using `NOT EXISTS`?"*

**Visual Breakdown:**
| employee_id | salary | manager_id | (Subquery: Active Managers) | Result (Salary < 30k & Manager NOT IN Active) |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 25000 | 3 | [1, 2, 4] | Manager 3 is gone. Row Kept! |
| 2 | 20000 | 1 | [1, 2, 4] | Manager 1 exists. Row Dropped. |

```sql
SELECT employee_id
FROM Employees
WHERE salary < 30000 
  AND manager_id NOT IN (SELECT employee_id FROM Employees)
ORDER BY employee_id;
```

#### 37. Exchange Seats (626)
- **The 'Why':** We mathematically swap seats inside a `CASE` expression. Odd IDs become Even (+1), Even IDs become Odd (-1). The subquery `SELECT MAX(id)` protects the last student if the total count is odd.
- **Execution Order:** Subquery (`SELECT MAX`) ➔ Main Query (`FROM` ➔ `SELECT` evaluates CASE ➔ `ORDER BY` new id).
- **Edge Cases:** If there's only 1 student, they match `MAX(id)` and keep their seat.
- **Performance:** Calculating `MAX(id)` inline for every row could be slow, though optimizers usually execute scalar subqueries once and cache them.
- **Interviewer Follow-up:** *"Rewrite this without mathematical swapping. How could you use the `LEAD()` and `LAG()` window functions instead?"*

**Visual Breakdown (Intermediate `CASE` evaluation):**
| Original id | student | `id % 2 != 0` | is `MAX(id)`? | New `id` |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Abbot | TRUE (Odd) | FALSE | 1 + 1 = **2** |
| 2 | Doris | FALSE (Even)| FALSE | 2 - 1 = **1** |
| 3 | Emerson| TRUE (Odd) | TRUE | Stays **3** |

```sql
SELECT 
    CASE 
        WHEN id % 2 != 0 AND id = (SELECT MAX(id) FROM Seat) THEN id
        WHEN id % 2 != 0 THEN id + 1
        ELSE id - 1 
    END AS id, 
    student
FROM Seat
ORDER BY id;
```

#### 38. Movie Rating (1341)
- **The 'Why':** We need two completely different aggregations (top user vs top movie) returned in a single column. `UNION ALL` concatenates the separate queries.
- **Execution Order:** Query 1 (Top User) executes ➔ Query 2 (Top Movie in Feb) executes ➔ `UNION ALL` merges them sequentially.
- **Edge Cases:** Ties are broken lexicographically (`ORDER BY name ASC`). If there's an exact tie in counts/ratings AND names, `LIMIT 1` deterministically picks the first.
- **Performance:** `UNION ALL` is faster than `UNION` because it skips the deduplication phase.
- **Interviewer Follow-up:** *"If we used `UNION` instead of `UNION ALL`, in what extremely rare scenario would the result only contain 1 row instead of 2?" (Answer: If a user's name is identical to a movie's title).*

**Visual Breakdown (Before UNION ALL):**
| Query 1 Result (Top User) | Query 2 Result (Top Movie in Feb) |
| :--- | :--- |
| 'Daniel' (Most ratings) | 'Frozen 2' (Highest AVG rating) |

```sql
(SELECT u.name AS results
 FROM MovieRating mr JOIN Users u ON mr.user_id = u.user_id
 GROUP BY u.user_id 
 ORDER BY COUNT(*) DESC, u.name ASC LIMIT 1)
UNION ALL
(SELECT m.title
 FROM MovieRating mr JOIN Movies m ON mr.movie_id = m.movie_id
 WHERE DATE_FORMAT(mr.created_at, '%Y-%m') = '2020-02'
 GROUP BY m.movie_id 
 ORDER BY AVG(mr.rating) DESC, m.title ASC LIMIT 1);
```

#### 39. Restaurant Growth (1321)
- **The 'Why':** A **7-Day Rolling Window** uses `SUM(...) OVER(ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)` to calculate weekly metrics seamlessly without self-joining 7 times.
- **Execution Order:** Subquery (`FROM` ➔ `GROUP BY` ➔ `SELECT` applies Window sums) ➔ Outer Query (`FROM` ➔ `WHERE` filters out the first 6 days ➔ `SELECT`).
- **Edge Cases:** We use `count_days = 7` to strictly filter out the first 6 days of the dataset, which don't have a full 7-day history to average.
- **Performance:** Window functions are highly optimized for moving averages. The `GROUP BY visited_on` in the subquery first flattens multiple orders per day.
- **Interviewer Follow-up:** *"Why must we `GROUP BY visited_on` inside the subquery first before applying the Window function?" (Answer: Multiple customers can visit on the same day).*

**Visual Breakdown (Intermediate Window Function in `t`):**
| visited_on | amount (Daily) | Rolling SUM (7 days) | Rolling COUNT (Days) |
| :--- | :--- | :--- | :--- |
| 2019-01-01 | 100 | 100 | 1 (Filtered out) |
| ... | ... | ... | ... |
| 2019-01-07 | 150 | 850 | **7 (Kept!)** |

```sql
SELECT visited_on, amount, ROUND(amount/7, 2) AS average_amount
FROM (
    SELECT visited_on, 
           SUM(SUM(amount)) OVER (ORDER BY visited_on ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS amount,
           COUNT(visited_on) OVER (ORDER BY visited_on ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS count_days
    FROM Customer
    GROUP BY visited_on
) t
WHERE count_days = 7;
```

#### 40. Friend Requests II: Who Has the Most Friends (602)
- **The 'Why':** Friendships are bidirectional but stored in two columns (`requester` and `accepter`). We use `UNION ALL` to **unpivot** them into a single column, letting us safely `GROUP BY` and count total connections per ID.
- **Execution Order:** Subquery (`UNION ALL` stacks the columns) ➔ Main Query (`GROUP BY` ➔ `SELECT` counts ➔ `ORDER BY` DESC ➔ `LIMIT 1`).
- **Edge Cases:** If a user requests a friend but is never accepted, they won't appear in the `RequestAccepted` table.
- **Performance:** Unpivoting via `UNION ALL` requires reading the table twice, but it is standard for edge-list graph tables.
- **Interviewer Follow-up:** *"What if two users have the exact same maximum number of friends? Does `LIMIT 1` guarantee returning both?" (Answer: No, you would need `RANK() = 1` or `HAVING COUNT = MAX(...)`).*

**Visual Breakdown (Unpivoting via UNION ALL):**
| requester_id | accepter_id | ➔ | Stacked `id` (UNION ALL) |
| :--- | :--- | :--- | :--- |
| 1 | 2 | ➔ | 1 |
| 1 | 3 | ➔ | 1 |
| 1 | 2 | ➔ | 2 |
| 1 | 3 | ➔ | 3 |
*(User 1 now appears twice in the stacked list, meaning 2 friends).*

```sql
SELECT id, COUNT(*) AS num
FROM (
    SELECT requester_id AS id FROM RequestAccepted
    UNION ALL
    SELECT accepter_id AS id FROM RequestAccepted
) t
GROUP BY id
ORDER BY num DESC
LIMIT 1;
```

#### 41. Investments in 2016 (585)
- **The 'Why':** We use `COUNT(*) OVER(PARTITION BY ...)` to flag rows that meet the criteria (duplicate `tiv_2015` and unique `lat, lon`) without collapsing the rows using `GROUP BY`.
- **Execution Order:** Subquery (`FROM` ➔ `SELECT` calculates window counts) ➔ Outer Query (`WHERE` applies uniqueness rules ➔ `SELECT` sums the values).
- **Edge Cases:** If no policy shares a `tiv_2015` value, the result is `NULL`.
- **Performance:** Window functions execute sequentially over the partitions. A self-join approach would be much slower on large datasets.
- **Interviewer Follow-up:** *"How would you write this using a `JOIN` to a `GROUP BY` subquery instead of Window Functions?"*

**Visual Breakdown (Intermediate Window Function in `t`):**
| PID | tiv_2015 | tiv_2016 | lat, lon | `tiv_2015_cnt` (Same value?) | `loc_cnt` (Same city?) | Action |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 10 | 5 | 10, 10 | 2 (>1, Good) | 1 (Unique, Good)| Keep & Sum |
| 2 | 10 | 10 | 20, 20 | 2 (>1, Good) | 2 (Not unique!)| Drop |

```sql
SELECT ROUND(SUM(tiv_2016), 2) AS tiv_2016
FROM (
    SELECT tiv_2016,
           COUNT(*) OVER(PARTITION BY tiv_2015) AS tiv_2015_cnt,
           COUNT(*) OVER(PARTITION BY lat, lon) AS loc_cnt
    FROM Insurance
) t
WHERE tiv_2015_cnt > 1 AND loc_cnt = 1;
```

#### 42. Department Top Three Salaries (185)
- **The 'Why':** **`DENSE_RANK()`** assigns continuous ranks to ordered rows partitioned by department. If two people tie for highest salary, they both get Rank 1, and the next person gets Rank 2.
- **Execution Order:** Subquery (`FROM` ➔ `JOIN` ➔ `SELECT` calculates DENSE_RANK) ➔ Outer Query (`WHERE rnk <= 3`).
- **Edge Cases:** If a department has only 2 employees, they are both returned (ranks 1 and 2 are `<= 3`).
- **Performance:** Partitioning and sorting a massive employee table is expensive. Ensure an index on `(departmentId, salary DESC)` exists.
- **Interviewer Follow-up:** *"What would happen to the output if you used `RANK()` instead of `DENSE_RANK()` and there was a 3-way tie for the highest salary?" (Answer: The ranks would be 1, 1, 1, 4. The `<= 3` filter would completely drop the next highest salary).*

**Visual Breakdown (Intermediate `DENSE_RANK`):**
| Dept | Employee | Salary | `DENSE_RANK()` (Ties share, no gaps) |
| :--- | :--- | :--- | :--- |
| IT | Max | 90000 | **1** |
| IT | Joe | 85000 | **2** |
| IT | Randy | 85000 | **2** (Tie!) |
| IT | Will | 70000 | **3** (No gap!) |
*(All four employees are returned since their rank is `<= 3`).*

```sql
SELECT Department, Employee, Salary
FROM (
    SELECT d.name AS Department, e.name AS Employee, e.salary AS Salary,
           DENSE_RANK() OVER (PARTITION BY d.id ORDER BY e.salary DESC) AS rnk
    FROM Employee e
    JOIN Department d ON e.departmentId = d.id
) t
WHERE rnk <= 3;
```

---

### Module 7: Advanced String Functions / Regex / Clause

#### 43. Fix Names in a Table (1667)
- **The 'Why':** We use `SUBSTRING` to isolate the first letter (to uppercase it) and the rest of the name (to lowercase it), then combine them using `CONCAT`.
- **Execution Order:** `FROM` ➔ `SELECT` (evaluates string functions row by row) ➔ `ORDER BY`.
- **Edge Cases:** If a name is only 1 character long, `SUBSTRING(name, 2)` safely returns an empty string instead of crashing.
- **Performance:** String manipulation functions are generally fast, but cannot be optimized via index. The `ORDER BY user_id` is the main performance driver here.
- **Interviewer Follow-up:** *"Can you write this using `LEFT()` and `RIGHT()` or `LENGTH()` instead of `SUBSTRING()`?"*

**Visual Breakdown (String Manipulation):**
| Original `name` | `UPPER(SUBSTR(name, 1, 1))` | `LOWER(SUBSTR(name, 2))` | `CONCAT` Result |
| :--- | :--- | :--- | :--- |
| aLice | A | lice | **Alice** |
| bOB | B | ob | **Bob** |

```sql
SELECT user_id, 
       CONCAT(UPPER(SUBSTRING(name, 1, 1)), LOWER(SUBSTRING(name, 2))) AS name
FROM Users
ORDER BY user_id;
```

#### 44. Patients With a Condition (1527)
- **The 'Why':** Conditions might be the first word (`'DIAB1%'`) or appear later in a space-separated list (`'% DIAB1%'`). We need an `OR` clause with `LIKE` to catch both safely.
- **Execution Order:** `FROM` ➔ `WHERE` (evaluates LIKE patterns) ➔ `SELECT`.
- **Edge Cases:** A condition like `'SADIAB100'` won't be incorrectly matched because we specifically check for a leading space (`'% DIAB1%'`).
- **Performance:** Leading wildcards (`'% DIAB1%'`) completely disable B-Tree index seeks. The DB must do a full table scan.
- **Interviewer Follow-up:** *"Since `LIKE '% DIAB1%'` causes a full table scan, how would you redesign the database schema to make finding conditions instantly fast?" (Answer: Normalize the `conditions` into a separate mapping table).*

**Visual Breakdown (LIKE Pattern Matching):**
| conditions | Matches `'DIAB1%'`? (Starts with) | Matches `'% DIAB1%'`? (Has space before) | Result |
| :--- | :--- | :--- | :--- |
| DIAB100 MYOP | **TRUE** | FALSE | **Keep** |
| ACNE DIAB100 | FALSE | **TRUE** | **Keep** |
| SADIAB100 | FALSE | FALSE | Drop |

```sql
SELECT patient_id, patient_name, conditions
FROM Patients
WHERE conditions LIKE 'DIAB1%' OR conditions LIKE '% DIAB1%';
```

#### 45. Delete Duplicate Emails (196)
- **The 'Why':** We use a **Self-Join `DELETE`**. By joining the table to itself on email and filtering for `p1.id > p2.id`, we target only the rows with the higher (newer) IDs for deletion.
- **Execution Order:** `FROM` (Self Join) ➔ `WHERE` (Identifies duplicates with higher IDs) ➔ `DELETE` (Removes matching `p1` rows).
- **Edge Cases:** If an email appears 3 times (IDs 1, 2, 3), both 2 and 3 will match against 1 and be deleted simultaneously.
- **Performance:** Self-joining a massive table for a `DELETE` can lock the table and cause massive transaction logs. In production, it's often safer to insert distinct rows into a temp table, truncate, and re-insert.
- **Interviewer Follow-up:** *"How would you rewrite this query to use a Window Function (`ROW_NUMBER()`) instead of a self-join?"*

**Visual Breakdown (Self-Join Identification):**
| `p1` (Target) | `p2` (Comparison) | Email Match? | `p1.id > p2.id`? | Action |
| :--- | :--- | :--- | :--- | :--- |
| ID: 2 (john@a.com) | ID: 1 (john@a.com) | YES | **YES (2 > 1)** | **DELETE `p1` (ID 2)** |
| ID: 1 (john@a.com) | ID: 2 (john@a.com) | YES | NO (1 < 2) | Ignore |

```sql
DELETE p1 
FROM Person p1, Person p2
WHERE p1.email = p2.email AND p1.id > p2.id;
```

#### 46. Second Highest Salary (176)
- **The 'Why':** `LIMIT 1 OFFSET 1` skips the first row and takes the second. We wrap it in a scalar `SELECT` so that if the inner query returns empty, the outer query explicitly returns `NULL`.
- **Execution Order:** Subquery (`FROM` ➔ `ORDER BY DESC` ➔ `LIMIT/OFFSET`) ➔ Outer Query (`SELECT` wrapper).
- **Edge Cases:** If there's only 1 employee in the table, `OFFSET 1` finds nothing. The outer `SELECT` successfully converts that empty set to `NULL`.
- **Performance:** `ORDER BY DESC LIMIT 2` is very fast if `salary` is indexed. `DISTINCT` adds slight overhead.
- **Interviewer Follow-up:** *"If I asked for the Nth highest salary using a variable, why is `LIMIT 1 OFFSET N-1` better than using `MAX()` recursively?"*

**Visual Breakdown (Outer SELECT acting as NULL-coalesce):**
| Table State | Inner Query Result (`LIMIT 1 OFFSET 1`) | Outer Query Result |
| :--- | :--- | :--- |
| [100, 200, 300] | 200 | 200 |
| [300] (Only 1 row) | *Empty Set* | **NULL** |

```sql
SELECT (
    SELECT DISTINCT salary 
    FROM Employee 
    ORDER BY salary DESC 
    LIMIT 1 OFFSET 1
) AS SecondHighestSalary;
```

#### 47. Group Sold Products By The Date (1484)
- **The 'Why':** `GROUP_CONCAT()` (MySQL) or `STRING_AGG()` (PostgreSQL) is used to roll up multiple row values into a single comma-separated string.
- **Execution Order:** `FROM` ➔ `GROUP BY` ➔ `SELECT` (counts distinct, concatenates distinct strings) ➔ `ORDER BY`.
- **Edge Cases:** If a product is sold twice on the same day, `DISTINCT` inside both `COUNT` and `GROUP_CONCAT` ensures it only appears once.
- **Performance:** String aggregation is memory-heavy. MySQL has a default `group_concat_max_len` limit (1024 bytes) that can truncate long strings in production!
- **Interviewer Follow-up:** *"What happens in MySQL if the concatenated string exceeds the default `group_concat_max_len` setting? How do you fix it?"*

**Visual Breakdown (String Aggregation):**
| Date | Raw Products | `DISTINCT` + `ORDER BY` | `GROUP_CONCAT` Result |
| :--- | :--- | :--- | :--- |
| 2020-05-30 | [Mask, Mask, Pen] | [Mask, Pen] | **'Mask,Pen'** |
| 2020-06-01 | [Pencil, Book] | [Book, Pencil] | **'Book,Pencil'** |

```sql
SELECT sell_date, 
       COUNT(DISTINCT product) AS num_sold, 
       GROUP_CONCAT(DISTINCT product ORDER BY product ASC SEPARATOR ',') AS products
FROM Activities
GROUP BY sell_date
ORDER BY sell_date;
```

#### 48. List the Products Ordered in a Period (1327)
- **The 'Why':** Standard `JOIN` with a `WHERE` filter for February 2020, followed by aggregation and a `HAVING` clause to filter out total orders < 100.
- **Execution Order:** `FROM` ➔ `JOIN` ➔ `WHERE` (date filter) ➔ `GROUP BY` ➔ `HAVING` (sum filter) ➔ `SELECT`.
- **Edge Cases:** If a product gets exactly 100 orders, it is included (`>= 100`).
- **Performance:** `DATE_FORMAT(o.order_date)` prevents the use of indexes on the date column. It is significantly faster to write `WHERE o.order_date BETWEEN '2020-02-01' AND '2020-02-29'`.
- **Interviewer Follow-up:** *"Why is `DATE_FORMAT(o.order_date) = '2020-02'` bad for performance? Rewrite the `WHERE` clause to be 'Sargable' (index-friendly)."*

**Visual Breakdown (HAVING vs WHERE):**
| Product | Orders in Feb (`WHERE`) | Total Sum | Passes `HAVING >= 100`? |
| :--- | :--- | :--- | :--- |
| Book | [10, 50] | 60 | **FALSE** (Dropped) |
| Pen | [50, 60] | 110 | **TRUE** (Kept!) |

```sql
SELECT p.product_name, SUM(o.unit) AS unit
FROM Products p
JOIN Orders o ON p.product_id = o.product_id
WHERE DATE_FORMAT(o.order_date, '%Y-%m') = '2020-02'
GROUP BY p.product_id
HAVING SUM(o.unit) >= 100;
```

#### 49. Find Users With Valid E-Mails (1517)
- **The 'Why':** We use `REGEXP` to strictly enforce email validation rules: must start with letter `^[a-zA-Z]`, followed by allowed chars `[a-zA-Z0-9_.-]*`, ending strictly with `@leetcode[.]com$`.
- **Execution Order:** `FROM` ➔ `WHERE` (Regex evaluation) ➔ `SELECT`.
- **Edge Cases:** The dot in `.com` must be escaped as `[.]` or `\.`, otherwise Regex treats it as "any character" (so `@leetcodezcom` would accidentally match).
- **Performance:** Regex evaluations require CPU-intensive string parsing and cannot utilize standard B-Tree indexes.
- **Interviewer Follow-up:** *"Why did we put the period inside brackets `[.]` in the Regex string? What would happen if we just wrote `@leetcode.com$`?"*

**Visual Breakdown (Regex Validation):**
| Email String | `^[a-zA-Z]` (Starts w/ letter) | `[a-zA-Z0-9_.-]*` (Valid body) | `@leetcode[.]com$` (Valid Domain) | Result |
| :--- | :--- | :--- | :--- | :--- |
| `a@leetcode.com` | ✅ | ✅ | ✅ | **Valid** |
| `1a@leetcode.com`| ❌ (Starts w/ number) | ✅ | ✅ | Invalid |
| `a-b@leetcode.com`| ✅ | ✅ (Dash allowed) | ✅ | **Valid** |

```sql
SELECT *
FROM Users
WHERE mail REGEXP '^[a-zA-Z][a-zA-Z0-9_.-]*@leetcode[.]com$';
```

#### 50. Capital Gain/Loss (1393) *(Bonus / Core 50th Problem Pattern)*
- **The 'Why':** We use a conditional summation trick. By mapping `'Buy'` to negative price and `'Sell'` to positive price inside a single `SUM()`, we calculate net flow in one pass.
- **Execution Order:** `FROM` ➔ `GROUP BY` ➔ `SELECT` (evaluates CASE inside SUM).
- **Edge Cases:** Assuming every buy has a corresponding sell, this works perfectly. If a stock is bought but not sold yet, it will accurately show a negative unrealized loss.
- **Performance:** This is heavily optimized because it avoids self-joins or multiple subqueries. It scans the table exactly once.
- **Interviewer Follow-up:** *"Rewrite this query using a `SUM(IF(...))` shorthand instead of `CASE WHEN`."*

**Visual Breakdown (Conditional Net Flow):**
| stock_name | operation | price | `CASE` Evaluation | Cumulative `SUM` per stock |
| :--- | :--- | :--- | :--- | :--- |
| Corona | Buy | 10 | **-10** | -10 |
| Corona | Sell | 50 | **+50** | **+40** (Final Gain) |

```sql
SELECT stock_name,
       SUM(CASE WHEN operation = 'Sell' THEN price ELSE -price END) AS capital_gain_loss
FROM Stocks
GROUP BY stock_name;
```

---

## PART 3: The FAANG "Red Flag" Cheat Sheet

In a FAANG SQL interview, it's not just about getting the right output—it's about avoiding performance traps and logical errors that signal a lack of experience. Here are the **Top 10 Easiest Traps to Fall Into** based on the LeetCode 50, with 1-line examples of what NOT to do.

### 1. The `NOT IN` NULL Trap
If a subquery returns even a single `NULL` value, `NOT IN` returns an empty set for the entire query.
- 🚩 **WRONG:** `WHERE department_id NOT IN (SELECT department_id FROM Departments)`
- ✅ **RIGHT:** `WHERE NOT EXISTS (SELECT 1 FROM Departments d WHERE d.id = e.department_id)`

### 2. Accidental Cartesian Explosions (Bad Joins)
Joining on non-unique columns without grouping can cause your row count to multiply exponentially, crashing the query.
- 🚩 **WRONG:** `SELECT * FROM Orders o JOIN Users u ON o.city = u.city` (Many-to-Many on city)
- ✅ **RIGHT:** `SELECT * FROM Orders o JOIN Users u ON o.user_id = u.user_id` (Join on Primary/Foreign Keys)

### 3. `WHERE` vs `HAVING` Misplacement
Filtering aggregate results in a `WHERE` clause will throw a syntax error. `WHERE` filters rows *before* grouping; `HAVING` filters *after*.
- 🚩 **WRONG:** `WHERE COUNT(user_id) > 5`
- ✅ **RIGHT:** `HAVING COUNT(user_id) > 5`

### 4. Integer Division Zeroing
In SQL Server and PostgreSQL, dividing two integers truncates the decimal. `1 / 2` becomes `0`.
- 🚩 **WRONG:** `SELECT accepted_requests / total_requests AS rate`
- ✅ **RIGHT:** `SELECT CAST(accepted_requests AS FLOAT) / total_requests AS rate` *(Or multiply by 100.0)*

### 5. `COUNT(*)` vs `COUNT(column)`
`COUNT(*)` counts rows. `COUNT(column)` counts *non-null* values in that column. Mixing them up causes subtle reporting bugs.
- 🚩 **WRONG:** `SELECT COUNT(*) FROM Employees` *(When asked for number of employees with a known birthdate)*
- ✅ **RIGHT:** `SELECT COUNT(birthdate) FROM Employees`

### 6. The `NULL` Inequality Blindspot
Comparing a value to `NULL` using `=` or `!=` yields `UNKNOWN`, dropping the row entirely.
- 🚩 **WRONG:** `WHERE bonus != 1000` *(Drops people with NO bonus / NULL)*
- ✅ **RIGHT:** `WHERE bonus != 1000 OR bonus IS NULL` *(Or `IFNULL(bonus, 0) != 1000`)*

### 7. Non-Sargable `WHERE` Clauses
Applying functions to columns in the `WHERE` clause blinds the optimizer to indexes, causing full table scans.
- 🚩 **WRONG:** `WHERE YEAR(order_date) = 2023`
- ✅ **RIGHT:** `WHERE order_date >= '2023-01-01' AND order_date < '2024-01-01'`

### 8. `UNION` vs `UNION ALL` Performance
`UNION` performs a costly sorting and deduplication step. If you know the sets are disjoint (or you want duplicates), always use `UNION ALL`.
- 🚩 **WRONG:** `SELECT id FROM TableA UNION SELECT id FROM TableB`
- ✅ **RIGHT:** `SELECT id FROM TableA UNION ALL SELECT id FROM TableB`

### 9. `RANK()` vs `DENSE_RANK()` Gaps
`RANK()` skips numbers after a tie (1, 1, 3). `DENSE_RANK()` does not (1, 1, 2). Using `RANK()` for "Top N" queries usually breaks.
- 🚩 **WRONG:** `WHERE RANK() OVER(ORDER BY salary DESC) <= 3` *(Might only return 2 unique salaries if there's a tie)*
- ✅ **RIGHT:** `WHERE DENSE_RANK() OVER(ORDER BY salary DESC) <= 3`

### 10. Forgetting to Handle Empty Sets
If a query asks for the "Nth highest", returning nothing when the table is small is often considered a failure. You must explicitly return `NULL`.
- 🚩 **WRONG:** `SELECT salary FROM Employee ORDER BY salary DESC LIMIT 1 OFFSET 1`
- ✅ **RIGHT:** `SELECT (SELECT salary FROM Employee ORDER BY salary DESC LIMIT 1 OFFSET 1) AS SecondHighest`
