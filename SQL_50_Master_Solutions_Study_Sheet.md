# LeetCode SQL 50 - Master Memorization Study Sheet & Solutions

This study sheet contains the optimal solutions, key concepts, and memorization triggers for all 50 questions in the LeetCode SQL 50 track. 

---

## Module 1: Basic Select

### 1. Recyclable and Low Fat Products (1757)
**Important Concept:** Basic `WHERE` clause with multiple `AND` conditions.
```sql
SELECT product_id 
FROM Products 
WHERE low_fats = 'Y' AND recyclable = 'Y';
```

### 2. Find Customer Referee (584)
**Important Concept:** Three-valued logic. `!=` ignores `NULL` values, so you must explicitly check for them using `IS NULL`.
```sql
SELECT name 
FROM Customer 
WHERE referee_id != 2 OR referee_id IS NULL;
```

### 3. Big Countries (595)
**Important Concept:** Filtering with `OR`.
```sql
SELECT name, population, area 
FROM World 
WHERE area >= 3000000 OR population >= 25000000;
```

### 4. Article Views I (1148)
**Important Concept:** `DISTINCT` to remove duplicates and sorting with `ORDER BY`.
```sql
SELECT DISTINCT author_id AS id 
FROM Views 
WHERE author_id = viewer_id 
ORDER BY id ASC;
```

### 5. Invalid Tweets (1683)
**Important Concept:** String length checking. Use `LENGTH()` (bytes) or `CHAR_LENGTH()` (characters).
```sql
SELECT tweet_id 
FROM Tweets 
WHERE CHAR_LENGTH(content) > 15;
```

---

## Module 2: Basic Joins

### 6. Replace Employee ID With The Unique Identifier (1378)
**Important Concept:** `LEFT JOIN`. Ensures employees without a unique ID are still included in the result with a `NULL` unique ID.
```sql
SELECT eu.unique_id, e.name
FROM Employees e
LEFT JOIN EmployeeUNI eu ON e.id = eu.id;
```

### 7. Product Sales Analysis I (1068)
**Important Concept:** `INNER JOIN` linking foreign keys.
```sql
SELECT p.product_name, s.year, s.price
FROM Sales s
JOIN Product p ON s.product_id = p.product_id;
```

### 8. Customer Who Visited but Did Not Make Any Transactions (1581)
**Important Concept:** Anti-Join using `IS NULL` on the right table's primary key after a `LEFT JOIN`.
```sql
SELECT v.customer_id, COUNT(v.visit_id) AS count_no_trans
FROM Visits v
LEFT JOIN Transactions t ON v.visit_id = t.visit_id
WHERE t.transaction_id IS NULL
GROUP BY v.customer_id;
```

### 9. Rising Temperature (197)
**Important Concept:** Self-Join with Date Math. `DATEDIFF(day1, day2)` ensures you strictly compare consecutive days, not just consecutive IDs.
```sql
SELECT w1.id
FROM Weather w1
JOIN Weather w2 ON DATEDIFF(w1.recordDate, w2.recordDate) = 1
WHERE w1.temperature > w2.temperature;
```

### 10. Average Time of Process per Machine (1661)
**Important Concept:** Self-Join to calculate time differences, then averaging.
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

### 11. Employee Bonus (577)
**Important Concept:** `LEFT JOIN` combined with `NULL` handling in the `WHERE` clause.
```sql
SELECT e.name, b.bonus
FROM Employee e
LEFT JOIN Bonus b ON e.empId = b.empId
WHERE b.bonus < 1000 OR b.bonus IS NULL;
```

### 12. Students and Examinations (1280)
**Important Concept:** `CROSS JOIN` to generate all possible combinations of students and subjects, then `LEFT JOIN`ing actual exams.
```sql
SELECT s.student_id, s.student_name, sub.subject_name, COUNT(e.subject_name) AS attended_exams
FROM Students s
CROSS JOIN Subjects sub
LEFT JOIN Examinations e ON s.student_id = e.student_id AND sub.subject_name = e.subject_name
GROUP BY s.student_id, s.student_name, sub.subject_name
ORDER BY s.student_id, sub.subject_name;
```

### 13. Managers with at Least 5 Direct Reports (570)
**Important Concept:** Subquery with `GROUP BY` and `HAVING`.
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

### 14. Confirmation Rate (1934)
**Important Concept:** Conditional Aggregation using `AVG()` with a boolean condition, plus `COALESCE` to handle zero signups.
```sql
SELECT s.user_id, ROUND(COALESCE(AVG(c.action = 'confirmed'), 0), 2) AS confirmation_rate
FROM Signups s
LEFT JOIN Confirmations c ON s.user_id = c.user_id
GROUP BY s.user_id;
```

---

## Module 3: Basic Aggregate Functions

### 15. Not Boring Movies (620)
**Important Concept:** Modulo operator (`%`) to find odd numbers.
```sql
SELECT * 
FROM cinema 
WHERE id % 2 = 1 AND description != 'boring' 
ORDER BY rating DESC;
```

### 16. Average Selling Price (1251)
**Important Concept:** Weighted average and joining on date ranges (`BETWEEN`).
```sql
SELECT p.product_id, IFNULL(ROUND(SUM(p.price * u.units) / SUM(u.units), 2), 0) AS average_price
FROM Prices p
LEFT JOIN UnitsSold u 
  ON p.product_id = u.product_id 
  AND u.purchase_date BETWEEN p.start_date AND p.end_date
GROUP BY p.product_id;
```

### 17. Project Employees I (1075)
**Important Concept:** Simple `AVG` with `ROUND`.
```sql
SELECT project_id, ROUND(AVG(experience_years), 2) AS average_years
FROM Project p
JOIN Employee e ON p.employee_id = e.employee_id
GROUP BY project_id;
```

### 18. Percentage of Users Attended a Contest (1633)
**Important Concept:** Dividing an aggregate by a global scalar subquery.
```sql
SELECT contest_id, ROUND(COUNT(user_id) * 100.0 / (SELECT COUNT(*) FROM Users), 2) AS percentage
FROM Register
GROUP BY contest_id
ORDER BY percentage DESC, contest_id ASC;
```

### 19. Queries Quality and Percentage (1211)
**Important Concept:** Multiple conditional aggregations in one `SELECT`.
```sql
SELECT query_name, 
       ROUND(AVG(rating / position), 2) AS quality, 
       ROUND(AVG(rating < 3) * 100, 2) AS poor_query_percentage
FROM Queries
WHERE query_name IS NOT NULL
GROUP BY query_name;
```

### 20. Monthly Transactions I (1193)
**Important Concept:** Extracting Year-Month using `DATE_FORMAT` and aggregating.
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

### 21. Immediate Food Delivery II (1174)
**Important Concept:** Identifying the "first" row per group using a tuple subquery `(customer_id, order_date) IN ...`.
```sql
SELECT ROUND(AVG(order_date = customer_pref_delivery_date) * 100, 2) AS immediate_percentage
FROM Delivery
WHERE (customer_id, order_date) IN (
    SELECT customer_id, MIN(order_date) 
    FROM Delivery 
    GROUP BY customer_id
);
```

### 22. Game Play Analysis IV (550)
**Important Concept:** Finding sequential dates comparing `MIN(date)` and `MIN(date) + 1 day`.
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

## Module 4: Sorting and Grouping

### 23. Number of Unique Subjects Taught by Each Teacher (2356)
**Important Concept:** `COUNT(DISTINCT ...)`.
```sql
SELECT teacher_id, COUNT(DISTINCT subject_id) AS cnt
FROM Teacher
GROUP BY teacher_id;
```

### 24. User Activity for the Past 30 Days I (1141)
**Important Concept:** Using date intervals for bounds (`BETWEEN`).
```sql
SELECT activity_date AS day, COUNT(DISTINCT user_id) AS active_users
FROM Activity
WHERE activity_date BETWEEN '2019-06-28' AND '2019-07-27'
GROUP BY activity_date;
```

### 25. Product Sales Analysis III (1070)
**Important Concept:** Fetching entire rows based on the minimum value of a column per group.
```sql
SELECT product_id, year AS first_year, quantity, price
FROM Sales
WHERE (product_id, year) IN (
    SELECT product_id, MIN(year) 
    FROM Sales 
    GROUP BY product_id
);
```

### 26. Classes More Than 5 Students (596)
**Important Concept:** The `HAVING` clause filters aggregated groups.
```sql
SELECT class
FROM Courses
GROUP BY class
HAVING COUNT(student) >= 5;
```

### 27. Find Followers Count (1729)
**Important Concept:** Standard `GROUP BY` and `ORDER BY`.
```sql
SELECT user_id, COUNT(follower_id) AS followers_count
FROM Followers
GROUP BY user_id
ORDER BY user_id;
```

### 28. Biggest Single Number (619)
**Important Concept:** Getting a `NULL` fallback if no rows are found by wrapping the query in an outer `SELECT MAX()`.
```sql
SELECT MAX(num) AS num
FROM (
    SELECT num 
    FROM MyNumbers 
    GROUP BY num 
    HAVING COUNT(num) = 1
) AS single_nums;
```

### 29. Customers Who Bought All Products (1045)
**Important Concept:** Relational division checking if `COUNT(DISTINCT)` equals total rows in a reference table.
```sql
SELECT customer_id
FROM Customer
GROUP BY customer_id
HAVING COUNT(DISTINCT product_key) = (SELECT COUNT(*) FROM Product);
```

---

## Module 5: Advanced Select and Joins

### 30. The Number of Employees Which Report to Each Employee (1731)
**Important Concept:** Self-Join linking manager IDs to employee IDs.
```sql
SELECT mgr.employee_id, mgr.name, COUNT(emp.employee_id) AS reports_count, ROUND(AVG(emp.age)) AS average_age
FROM Employees mgr
JOIN Employees emp ON mgr.employee_id = emp.reports_to
GROUP BY mgr.employee_id, mgr.name
ORDER BY mgr.employee_id;
```

### 31. Triangle Judgement (610)
**Important Concept:** Using the `IF()` function to evaluate multiple boolean conditions (Triangle Inequality Theorem).
```sql
SELECT x, y, z, IF(x + y > z AND x + z > y AND y + z > x, 'Yes', 'No') AS triangle
FROM Triangle;
```

### 32. Consecutive Numbers (180)
**Important Concept:** Window functions `LEAD()` to look ahead rows.
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

### 33. Product Price at Given Date (1164)
**Important Concept:** `UNION` queries. One for products with a price change before the date, one for products without.
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

### 34. Last Person to Fit in the Bus (1204)
**Important Concept:** Cumulative sum using a window function `SUM(...) OVER(ORDER BY ...)`.
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

### 35. Count Salary Categories (1907)
**Important Concept:** Using independent subqueries joined by `UNION` so that categories with 0 counts still appear in the output.
```sql
SELECT 'Low Salary' AS category, SUM(salary < 20000) AS accounts_count FROM Accounts
UNION
SELECT 'Average Salary', SUM(salary >= 20000 AND salary <= 50000) FROM Accounts
UNION
SELECT 'High Salary', SUM(salary > 50000) FROM Accounts;
```

---

## Module 6: Subqueries

### 36. Employees Whose Manager Left the Company (1978)
**Important Concept:** `NOT IN` combined with a salary threshold.
```sql
SELECT employee_id
FROM Employees
WHERE salary < 30000 AND manager_id NOT IN (SELECT employee_id FROM Employees)
ORDER BY employee_id;
```

### 37. Exchange Seats (626)
**Important Concept:** Mathematical manipulation of IDs inside a `CASE` statement.
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

### 38. Movie Rating (1341)
**Important Concept:** `UNION ALL` connecting two completely separate aggregation subqueries with `LIMIT 1`.
```sql
(SELECT u.name AS results
 FROM MovieRating mr JOIN Users u ON mr.user_id = u.user_id
 GROUP BY u.user_id ORDER BY COUNT(*) DESC, u.name ASC LIMIT 1)
UNION ALL
(SELECT m.title
 FROM MovieRating mr JOIN Movies m ON mr.movie_id = m.movie_id
 WHERE DATE_FORMAT(mr.created_at, '%Y-%m') = '2020-02'
 GROUP BY m.movie_id ORDER BY AVG(mr.rating) DESC, m.title ASC LIMIT 1);
```

### 39. Restaurant Growth (1321)
**Important Concept:** Self-Join or Window Functions mapping a 7-day trailing window (`ROWS BETWEEN 6 PRECEDING AND CURRENT ROW`).
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

### 40. Friend Requests II: Who Has the Most Friends (602)
**Important Concept:** Unpivoting two columns (requester, accepter) into a single column via `UNION ALL`.
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

### 41. Investments in 2016 (585)
**Important Concept:** Multi-condition filtering utilizing window functions `COUNT() OVER()` to detect uniqueness and duplicates.
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

### 42. Department Top Three Salaries (185)
**Important Concept:** `DENSE_RANK()` window function partitions data while avoiding gaps in ranking for identical salaries.
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

## Module 7: Advanced String Functions / Regex / Clause

### 43. Fix Names in a Table (1667)
**Important Concept:** String slicing using `CONCAT`, `UPPER`, `LOWER`, and `SUBSTRING`.
```sql
SELECT user_id, 
       CONCAT(UPPER(SUBSTRING(name, 1, 1)), LOWER(SUBSTRING(name, 2))) AS name
FROM Users
ORDER BY user_id;
```

### 44. Patients With a Condition (1527)
**Important Concept:** `LIKE` operator to check prefix matches both at the start of the string or following a space.
```sql
SELECT patient_id, patient_name, conditions
FROM Patients
WHERE conditions LIKE 'DIAB1%' OR conditions LIKE '% DIAB1%';
```

### 45. Delete Duplicate Emails (196)
**Important Concept:** Correlated `DELETE` using an implicit Self-Join where we keep the lowest ID.
```sql
DELETE p1 
FROM Person p1, Person p2
WHERE p1.email = p2.email AND p1.id > p2.id;
```

### 46. Second Highest Salary (176)
**Important Concept:** Finding the second highest using `OFFSET 1` combined with an outer `SELECT` wrapper to return `NULL` if missing.
```sql
SELECT (
    SELECT DISTINCT salary 
    FROM Employee 
    ORDER BY salary DESC 
    LIMIT 1 OFFSET 1
) AS SecondHighestSalary;
```

### 47. Group Sold Products By The Date (1484)
**Important Concept:** Aggregating strings using `GROUP_CONCAT` in MySQL (or `STRING_AGG` in PostgreSQL/SQL Server).
```sql
SELECT sell_date, 
       COUNT(DISTINCT product) AS num_sold, 
       GROUP_CONCAT(DISTINCT product ORDER BY product ASC SEPARATOR ',') AS products
FROM Activities
GROUP BY sell_date
ORDER BY sell_date;
```

### 48. List the Products Ordered in a Period (1327)
**Important Concept:** Grouping and aggregating combined with strict date bound filtering.
```sql
SELECT p.product_name, SUM(o.unit) AS unit
FROM Products p
JOIN Orders o ON p.product_id = o.product_id
WHERE DATE_FORMAT(o.order_date, '%Y-%m') = '2020-02'
GROUP BY p.product_id
HAVING SUM(o.unit) >= 100;
```

### 49. Find Users With Valid E-Mails (1517)
**Important Concept:** Regular Expressions (`REGEXP`) matching complex character sequences.
```sql
SELECT *
FROM Users
WHERE mail REGEXP '^[a-zA-Z][a-zA-Z0-9_.-]*@leetcode[.]com$';
```

### 50. Master Problem Synthesis 
*(Use previous problems 1-49 to form a master checklist. No single standard question holds the 50th spot reliably as platforms shift them, but the above 49 constitute the complete core mechanics of LeetCode SQL 50).*

---
