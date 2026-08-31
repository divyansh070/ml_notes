# Data Science & Python Muscle Memory Master Guide
### Python Core • NumPy • Pandas • EDA • Feature Engineering • Complete Pipeline

> **How to Use This Document (10-Minute Daily Muscle Memory Routine):**
> * **2 min** → Scan Python Core tricks (`zip`, `enumerate`, `Counter`, dict/list comps)
> * **2 min** → Scan NumPy arrays, slicing, boolean masks, and `axis`
> * **3 min** → Scan Pandas `loc`/`iloc`, `groupby` + `agg`/`transform`, `merge`
> * **2 min** → Read 1 real-world pattern from the Pattern Library
> * **1 min** → **Close the file and type 1 code snippet from scratch on a blank terminal** (e.g., `df.groupby("dept")["salary"].transform("mean")`)

---

## Table of Contents
1. [PART 1 — PYTHON CORE FOR DATA SCIENCE & OAs](#part-1--python-core-for-data-science--oas)
   - [Data Structures: Lists, Tuples, Sets, Dictionaries, Strings](#1-core-data-structures)
   - [Indexing, Slicing & Iteration](#2-indexing-slicing--iteration)
   - [Comprehensions (List, Dict, Set)](#3-comprehensions)
   - [Functions, Lambda, Map, Filter, Sorted](#4-functional-tools--lambdas)
   - [Essential Data Science Standard Library Tools](#5-essential-ds-standard-library-tools)
2. [PART 2 — NUMPY ARRAYS & VECTORIZATION](#part-2--numpy-arrays--vectorization)
   - [Array Creation, Inspection & Dtypes](#1-array-creation-inspection--dtypes)
   - [Indexing, Slicing & 2D Row/Col Selection](#2-indexing-slicing--2d-selection)
   - [Boolean Masking & Vectorization](#3-boolean-masking--vectorization)
   - [The Axis Mental Model (axis=0 vs axis=1)](#4-the-axis-mental-model-axis0-vs-axis1)
   - [Broadcasting Rules](#5-broadcasting-rules)
   - [Reshape, Flatten, Ravel & Transpose](#6-reshape-flatten-ravel--transpose)
   - [Aggregations, Argmin/Argmax & Where](#7-aggregations-argminargmax--npwhere)
   - [Stacking, Splitting & Unique](#8-stacking-splitting--unique)
   - [Random Number Generation & Linear Algebra for ML](#9-random-generation--linear-algebra-for-ml)
3. [PART 3 — PANDAS FUNDAMENTALS](#part-3--pandas-fundamentals)
   - [Loading & Inspection](#1-loading--inspection)
   - [Selection: Bracket vs .loc vs .iloc](#2-selection-loc-vs-iloc)
   - [Filtering & Boolean Indexing](#3-filtering--boolean-indexing)
   - [Creating & Modifying Columns](#4-creating--modifying-columns)
   - [Sorting & Ranking](#5-sorting--ranking)
   - [Missing Value Handling](#6-missing-value-handling)
   - [Duplicates & Type Casting](#7-duplicates--type-casting)
   - [String Manipulation (.str)](#8-string-manipulation-str)
   - [Frequency & Cardinality](#9-frequency--cardinality)
4. [PART 4 — PANDAS ANALYTICS + EDA](#part-4--pandas-analytics--eda)
   - [Mastering GroupBy (Aggregation vs Transform vs Filter)](#1-mastering-groupby)
   - [Merging & Joining DataFrames (Inner, Left, Right, Outer)](#2-merging--joining-dataframes)
   - [Concatenation, Pivoting & Melting](#3-concatenation-pivoting--melting)
   - [Window Functions: Rolling & Cumulative](#4-window-functions-rolling--cumulative)
   - [Date & Time Analysis (.dt accessor)](#5-date--time-analysis-dt)
   - [The 12-Step Exploratory Data Analysis (EDA) Checklist](#6-the-12-step-eda-checklist)
   - [Visualization Toolkit (Matplotlib & Seaborn)](#7-visualization-toolkit)
5. [PART 5 — FEATURE ENGINEERING & PREPROCESSING](#part-5--feature-engineering--preprocessing)
   - [Anti-Leakage Cardinal Rule](#the-cardinal-rule-of-anti-leakage-in-feature-engineering)
   - [Numerical Transformations](#1-numerical-transformations)
   - [Categorical Encoding](#2-categorical-encoding)
   - [Datetime Feature Extraction](#3-datetime-feature-extraction)
   - [Binning & Discretization](#4-binning--discretization)
   - [Outlier Detection & Capping](#5-outlier-detection--capping)
   - [Scikit-Learn Preprocessing Essentials](#6-scikit-learn-preprocessing-essentials)
6. [PART 6 — COMPLETE DATA SCIENCE PIPELINE TEMPLATES](#part-6--complete-data-science-pipeline-templates)
   - [6.1 Production Scikit-Learn Pipeline (Pipeline + ColumnTransformer)](#61-production-scikit-learn-pipeline)
   - [6.2 Pure NumPy & Pandas From-Scratch End-to-End Pipeline](#62-pure-numpy--pandas-from-scratch-end-to-end-pipeline)
7. [PART 7 — MACHINE LEARNING MODELS & SCIKIT-LEARN](#part-7--machine-learning-models--scikit-learn)
   - [7.1 ML Problem Types](#71-ml-problem-types)
   - [7.2 Universal 7-Step ML Workflow](#72-universal-7-step-ml-workflow)
   - [7.3 Linear Regression (Scikit-Learn + Pure NumPy from Scratch) ⭐⭐⭐](#73-linear-regression-)
   - [7.4 Ridge & Lasso Regularization ⭐⭐](#74-ridge--lasso-regularization-)
   - [7.5 Logistic Regression (Scikit-Learn + Pure NumPy from Scratch) ⭐⭐⭐](#75-logistic-regression-)
   - [7.6 K-Nearest Neighbors (KNN) ⭐⭐](#76-k-nearest-neighbors-knn-)
   - [7.7 Decision Trees ⭐⭐⭐](#77-decision-trees-)
   - [7.8 Random Forest ⭐⭐⭐](#78-random-forest-)
   - [7.9 Gradient Boosting (GBM / XGBoost)](#79-gradient-boosting-gbm--xgboost)
   - [7.10 Support Vector Machines (SVM)](#710-support-vector-machines-svm)
   - [7.11 K-Means Clustering](#711-k-means-clustering)
8. [PART 8 — MODEL EVALUATION & VALIDATION](#part-8--model-evaluation--validation)
   - [8.1 Regression Metrics (MAE, MSE, RMSE, R²)](#81-regression-metrics-mae-mse-rmse-r)
   - [8.2 Classification Metrics (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC)](#82-classification-metrics)
   - [8.3 Confusion Matrix Visual Breakdown](#83-confusion-matrix-visual-breakdown)
   - [8.4 Cross-Validation Best Practices](#84-cross-validation-best-practices)
   - [8.5 Hyperparameter Tuning (GridSearchCV)](#85-hyperparameter-tuning-gridsearchcv)
   - [8.6 Overfitting vs. Underfitting Diagnostics](#86-overfitting-vs-underfitting-diagnostics)
9. [PART 9 — DEEP LEARNING QUICK REFERENCE](#part-9--deep-learning-quick-reference)
   - [9.1 Multi-Layer Perceptron (MLP) with Keras](#91-multi-layer-perceptron-mlp-with-keras)
   - [9.2 Convolutional Neural Networks (CNN) ⭐⭐](#92-convolutional-neural-networks-cnn-)
   - [9.3 Recurrent Neural Networks (RNN) ⭐⭐](#93-recurrent-neural-networks-rnn-)
   - [9.4 LSTM & GRU (Long-Term Sequential Memory)](#94-lstm--gru-long-term-sequential-memory)
10. [PART 10 — STATISTICS FOR DATA SCIENCE & OAs](#part-10--statistics-for-data-science--oas)
    - [10.1 Descriptive Statistics & Dispersion (Mean, Median, Mode, Variance, Std, IQR)](#101-descriptive-statistics--dispersion)
    - [10.2 Standardization, Z-Score & Outlier Detection](#102-standardization-z-score--outlier-detection)
    - [10.3 Probability & Bayes' Theorem](#103-probability--bayes-theorem)
    - [10.4 Key Statistical Distributions](#104-key-statistical-distributions)
    - [10.5 Central Limit Theorem & Confidence Intervals](#105-central-limit-theorem--confidence-intervals)
    - [10.6 Hypothesis Testing, p-values & Error Types](#106-hypothesis-testing-p-values--error-types)
    - [10.7 Covariance, Pearson vs. Spearman Correlation & VIF](#107-covariance-pearson-vs-spearman-correlation--vif)
11. [MODEL CHEAT SHEETS & DECISION GUIDES](#model-cheat-sheets--decision-guides)
    - [Model Master Cheat Sheet Table](#model-master-cheat-sheet-table)
    - [Model Selection Decision Tree ("What Model First?")](#model-selection-decision-tree-what-model-first)
    - [Universal Model Templates (Regression, Classification, Clustering)](#universal-model-templates)
12. [🧠 ACTIVE RECALL DRILLS](#-active-recall-drills)
13. [⚠️ ML HIGH-FREQUENCY FORGETTING POINTS](#-ml-high-frequency-forgetting-points)
14. [QUICK REVISION SECTIONS (TIMED DRILLS)](#quick-revision-sections)
    - [10-Minute Python Revision](#10-minute-python-revision)
    - [10-Minute NumPy Revision](#10-minute-numpy-revision)
    - [15-Minute Pandas Revision](#15-minute-pandas-revision)
    - [15-Minute EDA Fast Run](#15-minute-eda-fast-run)
    - [Pipeline One-Screen Cheat Sheet](#pipeline-one-screen-cheat-sheet)
15. [PATTERN LIBRARY (20 COPY-PASTE DATA SCIENCE PATTERNS)](#pattern-library-20-copy-paste-patterns)
16. ["WHAT DO I USE?" DECISION TABLE](#what-do-i-use-decision-table)

---

# PART 1 — PYTHON CORE FOR DATA SCIENCE & OAs

## 1. Core Data Structures

### Lists (`list`)
#### Concept
Ordered, mutable sequence. Allows duplicates. Backed by a dynamic array ($O(1)$ append/pop from end, $O(n)$ insert/delete from middle/front).
#### Syntax I Should Remember
```python
nums = [1, 2, 3]
nums.append(4)         # In-place add to end -> [1, 2, 3, 4]
nums.extend([5, 6])    # In-place concat -> [1, 2, 3, 4, 5, 6]
nums.insert(0, 99)     # Insert at index 0 -> [99, 1, 2, 3, 4, 5, 6]
val = nums.pop()       # Removes & returns last item (6)
val = nums.pop(0)      # Removes & returns index 0 (99) -> O(n)
nums.reverse()         # In-place reverse
nums.sort(reverse=True)# In-place sort
```
#### Common Patterns
```python
# Flatten list of lists
nested = [[1, 2], [3, 4], [5, 6]]
flat = [x for sub in nested for x in sub]  # [1, 2, 3, 4, 5, 6]

# Chunking a list into batches of size k
k = 2
chunks = [nums[i:i + k] for i in range(0, len(nums), k)]
```
#### Common Mistakes
* Doing `nums = nums.sort()` — `sort()` returns `None`! Use `nums.sort()` or `nums = sorted(nums)`.
* Modifying a list while iterating over it with a `for x in nums:` loop.
#### Remember
> `nums.append(x)` modifies in-place and returns `None`. `+` creates a brand new list.

---

### Tuples (`tuple`)
#### Concept
Ordered, immutable sequence. Hashable if all items inside are hashable (can be dict keys or set elements).
#### Syntax I Should Remember
```python
point = (10, 20)
single = (42,)        # Note trailing comma! (42) is just an int
x, y = point          # Unpacking
a, *rest, b = (1, 2, 3, 4, 5) # a=1, rest=[2, 3, 4], b=5
```
#### Common Patterns
```python
# Returning multiple values from a function
def min_max(arr):
    return min(arr), max(arr)

low, high = min_max([4, 1, 9, 2])
```
#### Common Mistakes
* Trying to mutate a tuple: `point[0] = 5` raises `TypeError`.
#### Remember
> Commas make the tuple, not parentheses: `t = 1, 2` is a tuple.

---

### Sets (`set`)
#### Concept
Unordered collection of unique, hashable elements. Backed by a hash table ($O(1)$ average lookup, insert, delete).
#### Syntax I Should Remember
```python
s = {1, 2, 3}
s.add(4)              # Add element
s.discard(99)         # Safe remove (no error if missing)
s.remove(2)           # Unsafe remove (KeyError if missing)

# Set algebra
a, b = {1, 2, 3}, {2, 3, 4}
union = a | b         # {1, 2, 3, 4}
inter = a & b         # {2, 3}
diff  = a - b         # {1} (elements in a but not b)
sym_d = a ^ b         # {1, 4} (elements in a or b, not both)
```
#### Common Patterns
```python
# O(1) membership check & duplicate removal
seen = set()
unique_in_order = [x for x in data if not (x in seen or seen.add(x))]
```
#### Common Mistakes
* Creating an empty set with `{}` — that creates an empty `dict`! Use `s = set()`.
* Adding unhashable items (like lists or dicts) to a set: `s.add([1, 2])` raises `TypeError`.
#### Remember
> `{}` is a dict. `set()` is a set. `in` on sets is $O(1)$, on lists is $O(n)$.

---

### Dictionaries (`dict`)
#### Concept
Key-value store mapping unique hashable keys to values. Preserves insertion order (Python 3.7+). $O(1)$ average lookup.
#### Syntax I Should Remember
```python
d = {"a": 1, "b": 2}
val = d.get("c", 0)       # Safe get with default fallback (returns 0)
d["c"] = d.get("c", 0) + 1# Frequency counter pattern
keys = d.keys()           # View of keys
vals = d.values()         # View of values
items = d.items()         # View of (key, value) pairs
d.setdefault("k", []).append(1) # Inserts [] if missing, then appends
```
#### Common Patterns
```python
# Iterating key and value together
for k, v in d.items():
    if v > 10:
        print(k, v)

# Inverting a dictionary (values must be unique)
inv_d = {v: k for k, v in d.items()}
```
#### Common Mistakes
* `d['missing_key']` raises `KeyError`. Always use `d.get(k, default)` or `defaultdict`.
#### Remember
> `d.get(key, 0)` is your shield against `KeyError`.

---

### Strings (`str`)
#### Concept
Immutable sequence of Unicode characters. Any modification creates a new string.
#### Syntax I Should Remember
```python
s = "  Data Science,Python  "
s.strip()                 # "Data Science,Python" (strips whitespace)
s.lower() / s.upper()     # Case conversion
s.split(",")              # ['  Data Science', 'Python  ']
"-".join(["A", "B", "C"]) # "A-B-C" (Fastest string concatenation)
s.replace("Python", "AI") # Replace substring
s.startswith("Data")      # Boolean check
s.find("Science")         # Returns start index or -1 if not found
```
#### Common Patterns
```python
# Clean text token list
tokens = [w.strip().lower() for w in s.split(",") if w.strip()]
```
#### Common Mistakes
* Repeated string concatenation in a loop (`s += text`) creates $O(n^2)$ memory copying. Always append to a list and `"".join(list)`.
#### Remember
> Strings are immutable. Never `+=` in loops; accumulate in a list and `"".join()`.

---

## 2. Indexing, Slicing & Iteration

### Syntax I Should Remember
```python
# arr[start : stop : step] (stop is EXCLUSIVE)
arr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

arr[2:5]      # [2, 3, 4]
arr[:4]       # [0, 1, 2, 3] (First 4 elements)
arr[6:]       # [6, 7, 8, 9] (From index 6 to end)
arr[-1]       # 9 (Last element)
arr[-3:]      # [7, 8, 9] (Last 3 elements)
arr[::2]      # [0, 2, 4, 6, 8] (Every 2nd element)
arr[::-1]     # [9, 8, 7, 6, 5, 4, 3, 2, 1, 0] (Reversed copy)
```

### Essential Iteration Tools
```python
# 1. enumerate() -> index + value
for idx, val in enumerate(["apple", "banana", "cherry"], start=0):
    print(f"Index {idx} -> {val}")

# 2. zip() -> parallel iteration (stops at shortest iterable)
names = ["Alice", "Bob", "Charlie"]
scores = [85, 92, 78]
for name, score in zip(names, scores):
    print(f"{name}: {score}")

# Build dict via zip
score_map = dict(zip(names, scores))  # {'Alice': 85, 'Bob': 92, 'Charlie': 78}
```
#### Common Mistakes
* `zip` exhausts generators; if you need to reuse it, wrap it in `list(zip(...))`.
* `arr[start:stop]` stops at `stop - 1`.

---

## 3. Comprehensions

### Syntax I Should Remember
```python
# List Comprehension: [expression for item in iterable if condition]
evens_squared = [x**2 for x in range(10) if x % 2 == 0]

# Dict Comprehension: {key_expr: val_expr for item in iterable if condition}
word_len = {w: len(w) for w in ["cat", "elephant", "dog"] if len(w) > 3}

# Set Comprehension: {expr for item in iterable if condition}
unique_lengths = {len(w) for w in ["cat", "dog", "bird", "fish"]}

# Conditional Expression inside Comprehension (if-else)
# [expr_if_true if cond else expr_if_false for item in iterable]
labels = ["even" if x % 2 == 0 else "odd" for x in range(5)]
```
#### Common Mistakes
* Putting `if-else` at the end of the comprehension. 
  - Filtering only: `[x for x in arr if x > 0]` (at the end).
  - Value transformation: `[x if x > 0 else 0 for x in arr]` (before `for`).

---

## 4. Functional Tools & Lambdas

### Syntax I Should Remember
```python
# Lambda: anonymous inline function: lambda args: expression
square = lambda x: x ** 2
add = lambda a, b: a + b

# sorted(iterable, key=..., reverse=...)
pairs = [("A", 3), ("B", 1), ("C", 2)]
sorted_by_val = sorted(pairs, key=lambda x: x[1]) # [('B', 1), ('C', 2), ('A', 3)]

# Sort dictionary by value descending
d = {"apple": 5, "banana": 2, "cherry": 8}
sorted_d = dict(sorted(d.items(), key=lambda item: item[1], reverse=True))

# map(func, iter) & filter(func, iter)
nums = [1, 2, 3, 4, 5]
sq_nums = list(map(lambda x: x**2, nums))          # [1, 4, 9, 16, 25]
gt_two  = list(filter(lambda x: x > 2, nums))       # [3, 4, 5]

# any() and all()
any([False, True, False]) # True (at least one True)
all([True, True, False])  # False (all must be True)
```

---

## 5. Essential DS Standard Library Tools

### `collections.Counter`
```python
from collections import Counter

counts = Counter(["apple", "banana", "apple", "cherry", "apple", "banana"])
# Counter({'apple': 3, 'banana': 2, 'cherry': 1})
top_2 = counts.most_common(2) # [('apple', 3), ('banana', 2)]
print(counts["missing"])      # Returns 0 (never throws KeyError!)
```

### `collections.defaultdict`
```python
from collections import defaultdict

# Grouping items by key
groups = defaultdict(list)
data = [("fruit", "apple"), ("veg", "carrot"), ("fruit", "banana")]
for category, item in data:
    groups[category].append(item)
# {'fruit': ['apple', 'banana'], 'veg': ['carrot']}
```

### `collections.deque` (Double-Ended Queue)
```python
from collections import deque

dq = deque([1, 2, 3], maxlen=3)
dq.append(4)       # [2, 3, 4] -> automatically drops from left because maxlen=3
dq.appendleft(0)   # O(1) insert at head
val = dq.popleft() # O(1) remove from head (list.pop(0) is O(n)!)
```

### `heapq` (Min-Heap / Top-K Pattern)
```python
import heapq

nums = [20, 1, 15, 3, 7, 12]
heapq.heapify(nums)       # In-place transforms into min-heap in O(n)
smallest = heapq.heappop(nums) # 1 (O(log n))
heapq.heappush(nums, 4)   # Push 4 in O(log n)

# Direct Top-K queries without full sorting:
top_3_largest = heapq.nlargest(3, [20, 1, 15, 3, 7, 12])  # [20, 15, 12]
top_3_smallest = heapq.nsmallest(3, [20, 1, 15, 3, 7, 12]) # [1, 3, 7]
```

---

# PART 2 — NUMPY ARRAYS & VECTORIZATION

NumPy is the backbone of Machine Learning and numerical computation in Python. It provides homogeneous, contiguous C-memory n-dimensional arrays (`ndarray`) enabling SIMD hardware vectorization.

```
                    NUMPY AXIS MENTAL MODEL
           
                      axis = 1 (Columns / Across horizontal)
                            ───────►
                   ┌───────┬───────┬───────┐
       axis = 0    │ (0,0) │ (0,1) │ (0,2) │
  (Rows / Down)    ├───────┼───────┼───────┤
          │        │ (1,0) │ (1,1) │ (1,2) │
          ▼        ├───────┼───────┼───────┤
                   │ (2,0) │ (2,1) │ (2,2) │
                   └───────┴───────┴───────┘
```

---

## 1. Array Creation, Inspection & Dtypes

```python
import numpy as np

# Creation functions
a = np.array([1, 2, 3], dtype=np.float32)       # 1D array
b = np.array([[1, 2, 3], [4, 5, 6]], dtype=int) # 2D array (shape: 2, 3)
zeros = np.zeros((3, 4))                        # 3x4 float zeros
ones = np.ones((2, 3), dtype=np.int32)          # 2x3 int ones
full = np.full((2, 2), fill_value=7.5)          # 2x2 filled with 7.5
eye = np.eye(3)                                 # 3x3 Identity matrix
seq = np.arange(start=0, stop=10, step=2)       # [0, 2, 4, 6, 8] (stop exclusive)
lin = np.linspace(start=0, stop=1, num=5)       # [0.0, 0.25, 0.5, 0.75, 1.0]

# Essential Properties
b.shape   # (2, 3) -> (rows, cols)
b.ndim    # 2 -> number of dimensions
b.size    # 6 -> total number of elements (2 * 3)
b.dtype   # int64 (or int32)
b.astype(np.float32) # Cast to 32-bit float
```
> **When would I use this?** Initializing weights, generating synthetic regression features, or allocating memory buffers for ML batches.

---

## 2. Indexing, Slicing & 2D Selection

```python
mat = np.array([
    [10, 20, 30, 40],
    [50, 60, 70, 80],
    [90, 100, 110, 120]
])

# Syntax: mat[row_slice, col_slice]
mat[0, 0]         # 10 (top-left single element)
mat[1, :]         # array([50, 60, 70, 80]) -> Row 1 (all cols)
mat[:, 2]         # array([30, 70, 110])    -> Column 2 (all rows)
mat[0:2, 1:3]     # Slices rows 0..1 and cols 1..2 -> [[20, 30], [60, 70]]
mat[::-1, :]      # Reverse rows
mat[:, ::-1]      # Reverse columns

# Fancy Indexing (passing lists of integer indices)
mat[[0, 2], [1, 3]] # Picks (0,1) and (2,3) -> array([20, 120])
```
> **Common Trap:** In standard Python lists `nested[0][1]` is required. In NumPy always use comma separation: `mat[0, 1]`. Slicing a NumPy array returns a **view** (modifying the slice modifies the original array!). Use `.copy()` if you need an independent copy.

---

## 3. Boolean Masking & Vectorization

Boolean masking is filtering arrays using conditional boolean vectors without writing `for` loops.

```python
x = np.array([12, -5, 0, 45, -22, 18, 9])

# Step 1: Condition creates a boolean array of same shape
mask = x > 0 # array([True, False, False, True, False, True, True])

# Step 2: Index with mask to extract elements
positives = x[mask]            # array([12, 45, 18, 9])
x[x < 0] = 0                   # In-place clipping (ReLU operation!): negative values become 0

# Multiple conditions: Use & (AND), | (OR), ~ (NOT) with PARENTHESES!
filtered = x[(x >= 10) & (x <= 40)] # array([12, 18])
```
> **When would I use this?** Removing outliers, applying threshold cutoffs, implementing activation functions (ReLU), or selecting target classes.

---

## 4. The Axis Mental Model (`axis=0` vs `axis=1`)

> [!IMPORTANT]
> **THE AXIS RULE TO REMEMBER FOREVER:**
> * `axis=0` collapses **ROWS** (operates vertically downwards across rows $\downarrow$, outputs 1 value per column).
> * `axis=1` collapses **COLUMNS** (operates horizontally across columns $\rightarrow$, outputs 1 value per row).

```python
X = np.array([
    [1, 2, 3],
    [4, 5, 6]
]) # Shape: (2, 3)

# Summing along axis=0 (collapses the 2 rows -> result length 3)
np.sum(X, axis=0) # array([5, 7, 9])   -> (1+4, 2+5, 3+6)

# Summing along axis=1 (collapses the 3 cols -> result length 2)
np.sum(X, axis=1) # array([6, 15])     -> (1+2+3, 4+5+6)

# Mean feature centering across samples (standard DS step):
col_means = np.mean(X, axis=0) # [2.5, 3.5, 4.5] (mean of each feature)
X_centered = X - col_means     # Broadcasting subtracts col_means from each row
```

---

## 5. Broadcasting Rules

Broadcasting allows arithmetic operations between arrays of different shapes without copying memory.

### The 2 Rules of Broadcasting:
Two dimensions are compatible when:
1. They are **equal**, OR
2. One of them is **1**.

NumPy compares shapes element-wise starting from the **trailing (rightmost)** dimension and works its way left.

```python
# Example 1: Matrix + Row Vector
A = np.ones((3, 4)) # shape: (3, 4)
b = np.array([1, 2, 3, 4]) # shape: (4,) -> matches trailing dim 4
C = A + b           # b is broadcast across all 3 rows. Result shape: (3, 4)

# Example 2: Column Vector + Row Vector
col = np.array([[10], [20], [30]]) # shape: (3, 1)
row = np.array([1, 2, 3, 4])        # shape: (4,) -> (1, 4)
grid = col + row                   # shape: (3, 4)

# Adding a new dimension for broadcasting
v = np.array([1, 2, 3]) # shape (3,)
v_col = v[:, np.newaxis] # shape (3, 1)
v_row = v[np.newaxis, :] # shape (1, 3)
```
> **When would I use this?** Adding neural network bias vectors to layer outputs, pairwise distance matrix calculation, scaling features by mean/std.

---

## 6. Reshape, Flatten, Ravel & Transpose

```python
arr = np.arange(12) # [0, 1, 2, ..., 11], shape (12,)

# Reshape (use -1 to let NumPy infer the remaining dimension)
mat = arr.reshape(3, 4)   # shape (3, 4)
mat = arr.reshape(3, -1)  # shape (3, 4) (automatically computes 12/3 = 4)
mat_col = arr.reshape(-1, 1) # shape (12, 1) -> Crucial for sklearn single-feature input!

# Flatten vs Ravel
flt = mat.flatten() # Returns a COPY of 1D array
rvl = mat.ravel()   # Returns a VIEW (fast, zero memory allocation)

# Transpose
mat_T = mat.T       # Swaps rows and columns -> shape (4, 3)
```
> **When would I use this?** Reshaping image tensors from $(N, H, W, C)$ to flat feature vectors $(N, H \cdot W \cdot C)$ or feeding a 1D Series into Scikit-learn via `.reshape(-1, 1)`.

---

## 7. Aggregations, Argmin/Argmax & `np.where`

```python
data = np.array([[10, 25, 5], [40, 15, 30]])

np.mean(data)          # 20.83 (overall mean)
np.median(data)        # 20.0
np.std(data, axis=0)   # Standard deviation per column
np.min(data), np.max(data)

# argmin / argmax -> Returns the INDEX of the min/max value
arr = np.array([10, 50, 20, 80, 30])
best_idx = np.argmax(arr) # 3 (because arr[3] == 80)
# In 2D:
np.argmax(data, axis=1)   # array([1, 0]) -> index of max in each row

# np.where(condition, value_if_true, value_if_false)
# Vectorized ternary if-else:
grades = np.array([85, 42, 90, 55])
pass_fail = np.where(grades >= 60, "PASS", "FAIL") 
# array(['PASS', 'FAIL', 'PASS', 'FAIL'], dtype='<U4')
```
> **When would I use this?** Finding the predicted class from neural network softmax output (`np.argmax(probs, axis=1)`), vectorized data binning/flagging with `np.where`.

---

## 8. Stacking, Splitting & Unique

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Vertical Stack (row-wise concatenation)
np.vstack([a, b])      # shape (2, 3) -> [[1, 2, 3], [4, 5, 6]]

# Horizontal Stack (col-wise concatenation)
np.hstack([a, b])      # shape (6,) -> [1, 2, 3, 4, 5, 6]

# 2D Concatenate
m1 = np.ones((2, 2))
m2 = np.zeros((2, 2))
np.concatenate([m1, m2], axis=0) # shape (4, 2)
np.concatenate([m1, m2], axis=1) # shape (2, 4)

# Unique values and counts
vals, counts = np.unique(np.array([1, 2, 2, 3, 3, 3]), return_counts=True)
# vals = [1, 2, 3], counts = [1, 2, 3]
```

---

## 9. Random Generation & Linear Algebra for ML

```python
# Modern Recommended Random Generator (NumPy 1.17+)
rng = np.random.default_rng(seed=42)

r_uniform = rng.uniform(low=0.0, high=1.0, size=(3, 2)) # Uniform [0, 1)
r_normal  = rng.normal(loc=0.0, scale=1.0, size=(100,)) # Normal (mu=0, sigma=1)
r_ints    = rng.integers(low=1, high=10, size=5)        # Discrete ints [1..9]
choices   = rng.choice(["A", "B", "C"], size=10, p=[0.7, 0.2, 0.1]) # Weighted sampling

# Linear Algebra Essentials
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix Multiplication: Use `@` or np.matmul (NOT `*` which is element-wise!)
C = A @ B                        # Matrix product (2x2)
dot_prod = np.dot(a, b)          # Vector dot product sum(a_i * b_i)
norm = np.linalg.norm(a)         # Euclidean L2 norm: sqrt(sum(a_i^2))
inv_A = np.linalg.inv(A)         # Matrix inverse
```

---

# PART 3 — PANDAS FUNDAMENTALS

Pandas is built on top of NumPy arrays. Its primary structures are 1D `Series` and 2D `DataFrame`.

---

## 1. Loading & Inspection

```python
import pandas as pd

# Load CSV with essential parameters
df = pd.read_csv(
    "dataset.csv",
    sep=",",
    header=0,
    parse_dates=["transaction_date"], # Automatically parse dates
    na_values=["?", "NA", "missing"]   # Standardize missing strings to np.nan
)

# Inspection Checklist
df.head(5)          # First 5 rows
df.tail(5)          # Last 5 rows
df.shape            # (n_rows, n_cols) tuple
df.columns          # Index object of column names
df.dtypes           # Data type of each column
df.info()           # Summary of memory usage, non-null counts, dtypes
df.describe()       # Statistical summary of numerical columns (mean, std, IQR)
df.describe(include="object") # Summary of categorical columns (count, unique, top, freq)
df.memory_usage(deep=True).sum() / 1e6 # Total memory in MB
```

---

## 2. Selection: Bracket vs `.loc` vs `.iloc`

| Method | Syntax | What it uses | Inclusion Rule |
| :--- | :--- | :--- | :--- |
| **`df["col"]`** | `df["age"]` | Column name | Returns 1D `Series` |
| **`df[["c1", "c2"]]`** | `df[["age", "salary"]]` | List of column names | Returns 2D `DataFrame` |
| **`df.loc[]`** | `df.loc[row_labels, col_labels]` | **Labels** (names / boolean mask) | **Both start and stop INCLUDED!** |
| **`df.iloc[]`** | `df.iloc[row_idx, col_idx]` | **Integer positions** (0-indexed) | **Stop index EXCLUDED!** (like Python slice) |

```python
# .loc examples (LABEL BASED)
df.loc[0:3, "age"]                # Rows with labels 0, 1, 2, 3 and column 'age'
df.loc[df["salary"] > 50000, ["name", "salary"]] # Boolean mask row filter + col select

# .iloc examples (POSITION BASED)
df.iloc[0:3, 0:2]                 # First 3 rows (0, 1, 2) and first 2 columns (0, 1)
df.iloc[-1, :]                    # Very last row across all columns
df.iloc[:, [0, 2]]                # All rows, columns at index 0 and 2
```
> **Common Mistake:** Using `df.loc[0:3]` and expecting 3 rows. `.loc` includes index 3, yielding 4 rows. Use `.iloc[0:3]` if you want pure index-range slicing.

---

## 3. Filtering & Boolean Indexing

```python
# 1. Single condition
seniors = df[df["age"] >= 60]

# 2. Multiple conditions (ALWAYS wrap each condition in parentheses!)
# & = AND, | = OR, ~ = NOT
target_group = df[(df["age"] > 25) & (df["department"] == "Engineering")]
target_or    = df[(df["salary"] > 100000) | (df["experience"] > 10)]
not_sales    = df[~(df["department"] == "Sales")]

# 3. isin() -> Filter against a list of categories
cities = ["New York", "London", "Tokyo"]
df_metro = df[df["city"].isin(cities)]

# 4. between() -> Filter values in closed range [a, b] inclusive
df_mid_age = df[df["age"].between(30, 50, inclusive="both")]

# 5. query() -> Clean string syntax for complex filters
df_fast = df.query("age > 25 and department == 'Engineering' and salary >= 80000")
```

---

## 4. Creating & Modifying Columns

```python
# Arithmetic column creation
df["total_comp"] = df["salary"] + df["bonus"]
df["tax_amount"] = df["salary"] * 0.20

# Conditional column creation with np.where
df["seniority"] = np.where(df["experience"] >= 5, "Senior", "Junior")

# Multi-condition column creation with np.select
conditions = [
    df["score"] >= 90,
    df["score"] >= 75,
    df["score"] >= 60
]
choices = ["A", "B", "C"]
df["grade"] = np.select(conditions, choices, default="F")

# map() for dictionary replacement
gender_map = {"M": "Male", "F": "Female", "O": "Other"}
df["gender_full"] = df["gender"].map(gender_map)

# apply() with lambda (Use sparingly, slow row-by-row Python loop!)
df["cleaned_title"] = df["title"].apply(lambda x: str(x).strip().title())
```

---

## 5. Sorting & Ranking

```python
# Sort by single column
df_sorted = df.sort_values(by="salary", ascending=False)

# Sort by multiple columns (e.g. Dept A-Z, then Salary Highest first)
df_multi_sort = df.sort_values(
    by=["department", "salary"],
    ascending=[True, False]
)

# Sort by index
df_idx_sorted = df.sort_index(ascending=True)

# nlargest & nsmallest (Faster than full sort if only needing Top K)
top_5_earners = df.nlargest(5, columns="salary")
bottom_3_scores = df.nsmallest(3, columns="test_score")
```

---

## 6. Missing Value Handling

```python
# Detection
df.isna().sum()                     # Count of NaN values per column
df.isna().mean() * 100              # Percentage of missing data per column
df[df["salary"].isna()]             # Rows where salary is NaN
df[df["salary"].notna()]            # Rows where salary is present

# Dropping Missing Values
df_clean = df.dropna()              # Drops any row containing at least one NaN
df_clean = df.dropna(subset=["salary", "age"]) # Drops row only if salary OR age is NaN
df_clean = df.dropna(thresh=5)      # Keep rows with at least 5 non-null values

# Imputation (Filling)
df["salary_filled"] = df["salary"].fillna(df["salary"].median()) # Fill with median
df["dept_filled"]   = df["department"].fillna("Unknown")         # Fill categorical
df["ts_forward"]    = df["stock_price"].ffill()                  # Forward-fill (time-series)
df["ts_backward"]   = df["stock_price"].bfill()                  # Backward-fill
```

---

## 7. Duplicates & Type Casting

```python
# Duplicates
df.duplicated().sum()               # Number of exact duplicate rows
df.duplicated(subset=["user_id"]).sum() # Duplicates on a specific key
df_unique = df.drop_duplicates()    # Remove identical rows
df_unique_user = df.drop_duplicates(subset=["user_id"], keep="last") # Keep latest

# Type Casting
df["age"] = df["age"].astype("int32")
df["category"] = df["category"].astype("category") # Reduces memory footprint by up to 80%!

# Safe Numeric Conversion (converts invalid strings like "N/A" to NaN)
df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")

# Date Conversion
df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
```

---

## 8. String Manipulation (`.str`)

```python
# Must use the .str accessor
df["name"] = df["name"].str.strip().str.title()
df["is_engineer"] = df["title"].str.contains("Engineer", case=False, na=False)
df["clean_phone"] = df["phone"].str.replace(r"\D+", "", regex=True)

# String splitting into multiple columns
df[["first_name", "last_name"]] = df["full_name"].str.split(" ", n=1, expand=True)

# Extract regex capture groups
df["zip_code"] = df["address"].str.extract(r"(\b\d{5}\b)")
```

---

## 9. Frequency & Cardinality

```python
# Value Counts (Frequencies)
df["department"].value_counts()                # Raw counts
df["department"].value_counts(normalize=True)  # Proportions / percentages
df["department"].value_counts(dropna=False)    # Includes NaN counts

# Unique checks
df["city"].unique()    # Array of unique values: ['NYC', 'London', 'Tokyo']
df["city"].nunique()   # Integer count of unique values: 3
```

---

# PART 4 — PANDAS ANALYTICS + EDA

---

## 1. Mastering GroupBy

```
                       GROUPBY PIPELINE
                       
   RAW DATAFRAME               SPLIT INTO GROUPS              APPLY & COMBINE
   Dept    Salary             Dept = Sales                   Sales Mean: $65k
 ┌───────┬────────┐         ┌───────┬────────┐              ┌───────┬────────┐
 │ Sales │  50000 │         │ Sales │  50000 │              │ Sales │  65000 │
 │ IT    │  80000 │ ──────► │ Sales │  80000 │ ───────────► ├───────┼────────┤
 │ Sales │  80000 │         └───────┴────────┘              │ IT    │  85000 │
 │ IT    │  90000 │           Dept = IT                     └───────┴────────┘
 └───────┴────────┘         ┌───────┬────────┐
                            │ IT    │  80000 │
                            │ IT    │  90000 │
                            └───────┴────────┘
```

### Pattern A: Standard Aggregation
```python
# Single metric
df.groupby("department")["salary"].mean()

# Multiple metrics with .agg()
dept_stats = df.groupby("department").agg(
    avg_salary=("salary", "mean"),
    max_salary=("salary", "max"),
    total_headcount=("employee_id", "count"),
    std_salary=("salary", "std")
).reset_index()
```

### Pattern B: `.transform()` (Preserves Original Row Shape)
> [!NOTE]
> `.agg()` collapses rows (outputs 1 row per group).
> `.transform()` broadcasts the group metric back to EVERY original row (shape unchanged).

```python
# Calculate group mean and attach as a column to every employee
df["dept_avg_salary"] = df.groupby("department")["salary"].transform("mean")

# Calculate deviation from department average
df["diff_from_dept_avg"] = df["salary"] - df["dept_avg_salary"]

# Group-level z-score normalization
df["salary_dept_zscore"] = df.groupby("department")["salary"].transform(
    lambda x: (x - x.mean()) / (x.std() + 1e-8)
)
```

### Pattern C: Top-N Per Group & Within-Group Rank
```python
# Rank employees within their department by salary
df["dept_salary_rank"] = df.groupby("department")["salary"].rank(
    method="dense", 
    ascending=False
)

# Get Top 2 highest earners in each department
top_2_per_dept = (
    df.sort_values(["department", "salary"], ascending=[True, False])
      .groupby("department")
      .head(2)
)
```

### Pattern D: Group Filtering (`.filter()`)
```python
# Keep only departments with more than 10 employees
large_depts = df.groupby("department").filter(lambda g: len(g) > 10)
```

---

## 2. Merging & Joining DataFrames

```
   INNER JOIN             LEFT JOIN              RIGHT JOIN             FULL OUTER JOIN
  ┌─────┬─────┐          ┌─────┬─────┐          ┌─────┬─────┐          ┌─────┬─────┐
  │  A  │  B  │          │  A  │  B  │          │  A  │  B  │          │  A  │  B  │
  │     │█████│          │█████│█████│          │     │█████│          │█████│█████│
  │     │█████│          │█████│█████│          │     │█████│          │█████│█████│
  └─────┴─────┘          └─────┴─────┘          └─────┴─────┘          └─────┴─────┘
  Only matching keys     All left + matches     All right + matches    Everything
```

```python
# Standard Merge Syntax
merged_df = pd.merge(
    left=df_orders,
    right=df_customers,
    on="customer_id",       # Common key
    how="left"              # 'inner', 'left', 'right', 'outer'
)

# Different column names on left and right
merged_df = pd.merge(
    left=df_orders,
    right=df_users,
    left_on="user_id",
    right_on="id",
    how="inner",
    suffixes=("_order", "_user") # Disambiguate duplicate col names
)

# Validate join cardinality (Catches Cartesian product explosion bugs!)
merged_df = pd.merge(df_orders, df_customers, on="customer_id", how="left", validate="many_to_one")
```

---

## 3. Concatenation, Pivoting & Melting

```python
# Concatenation (Stacking)
# Vertical Stack (Rows): Add more rows from same schema
df_all = pd.concat([df_2023, df_2024], axis=0, ignore_index=True)

# Horizontal Stack (Cols): Add side-by-side columns
df_combined = pd.concat([df_features, df_labels], axis=1)

# Pivot Table (Aggregation Grid: Reshapes Long -> Wide)
pivot = df.pivot_table(
    index="department",
    columns="gender",
    values="salary",
    aggfunc="mean",
    fill_value=0,
    margins=True # Adds 'ALL' total row/col
)

# Melt (Reshapes Wide -> Long: Unpivots columns into rows)
# Essential for tidy data & Seaborn plotting!
df_long = pd.melt(
    df_wide,
    id_vars=["student_id", "name"],
    value_vars=["math_score", "science_score", "english_score"],
    var_name="subject",
    value_name="score"
)
```

---

## 4. Window Functions: Rolling & Cumulative

```python
# Ensure data is sorted by date before windowing!
df = df.sort_values("date")

# Cumulative calculations
df["cum_revenue"] = df["revenue"].cumsum()
df["cum_max_price"] = df["stock_price"].cummax()

# Rolling Window (Moving Average)
# 7-day moving average
df["revenue_7d_ma"] = df["revenue"].rolling(window=7, min_periods=1).mean()

# Exponential Moving Average (EMA)
df["revenue_ema"] = df["revenue"].ewm(span=14, adjust=False).mean()

# Shift / Lag (Compute day-over-day growth)
df["prev_day_rev"] = df["revenue"].shift(1) # Shifts data down by 1 row
df["dod_growth_pct"] = (df["revenue"] - df["prev_day_rev"]) / df["prev_day_rev"] * 100
```

---

## 5. Date & Time Analysis (`.dt`)

```python
# Access components via .dt accessor
df["year"]        = df["date"].dt.year
df["month"]       = df["date"].dt.month
df["day"]         = df["date"].dt.day
df["day_name"]    = df["date"].dt.day_name()
df["is_weekend"]  = df["date"].dt.dayofweek.isin([5, 6]).astype(int) # 5=Sat, 6=Sun
df["quarter"]     = df["date"].dt.quarter

# Date Arithmetic
df["days_since_signup"] = (pd.Timestamp.now() - df["signup_date"]).dt.days

# Date Filtering
df_q1 = df[df["date"].between("2024-01-01", "2024-03-31")]

# Resample (Time-Series Groupby - Requires DatetimeIndex)
df_ts = df.set_index("date")
monthly_summary = df_ts.resample("ME")["revenue"].sum() # Month-End frequency
```

---

## 6. The 12-Step EDA Checklist

Run through this code block on any new dataset:

```python
def run_fast_eda(df: pd.DataFrame):
    print("=" * 60)
    print("STEP 1: SHAPE & SAMPLES")
    print(f"Rows: {df.shape[0]:,}, Columns: {df.shape[1]:,}\n")
    print(df.head(3))
    
    print("\n" + "=" * 60)
    print("STEP 2: DATA TYPES & MEMORY")
    print(df.info())
    
    print("\n" + "=" * 60)
    print("STEP 3: MISSING VALUES CHECK")
    null_counts = df.isna().sum()
    null_pct = df.isna().mean() * 100
    null_df = pd.DataFrame({"Missing Count": null_counts, "Missing %": null_pct})
    print(null_df[null_df["Missing Count"] > 0].sort_values(by="Missing %", ascending=False))
    
    print("\n" + "=" * 60)
    print("STEP 4: DUPLICATES")
    print(f"Exact Duplicate Rows: {df.duplicated().sum():,}")
    
    print("\n" + "=" * 60)
    print("STEP 5: NUMERICAL DISTRIBUTIONS")
    print(df.describe().T[["count", "mean", "std", "min", "50%", "max"]])
    
    print("\n" + "=" * 60)
    print("STEP 6: CATEGORICAL CARDINALITY")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for c in cat_cols:
        print(f"Column '{c}': {df[c].nunique()} unique values -> Top: {dict(df[c].value_counts().head(3))}")
```

---

## 7. Visualization Toolkit

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set clean whitegrid aesthetic
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. Distribution (Histplot with KDE)
sns.histplot(data=df, x="salary", kde=True, ax=axes[0, 0], color="steelblue")
axes[0, 0].set_title("Salary Distribution (Check Skewness)")

# 2. Outliers (Boxplot)
sns.boxplot(data=df, x="department", y="salary", ax=axes[0, 1], palette="Set2")
axes[0, 1].set_title("Salary by Department (Check Outliers)")

# 3. Categorical Counts (Countplot)
sns.countplot(data=df, x="department", ax=axes[0, 2], palette="viridis")
axes[0, 2].set_title("Headcount per Department (Class Balance)")

# 4. Correlation / Relationship (Scatterplot)
sns.scatterplot(data=df, x="experience", y="salary", hue="department", ax=axes[1, 0])
axes[1, 0].set_title("Experience vs Salary (Linearity check)")

# 5. Trend / Time-Series (Lineplot)
sns.lineplot(data=df, x="date", y="revenue", ax=axes[1, 1], color="darkgreen")
axes[1, 1].set_title("Revenue Trend over Time")

# 6. Feature Correlations (Heatmap)
num_df = df.select_dtypes(include=np.number)
sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=False, ax=axes[1, 2])
axes[1, 2].set_title("Feature Correlation Matrix (Multicollinearity)")

plt.tight_layout()
plt.show()
```

---

# PART 5 — FEATURE ENGINEERING & PREPROCESSING

---

> [!IMPORTANT]
> **THE CARDINAL RULE OF ANTI-LEAKAGE IN FEATURE ENGINEERING:**
> Any transformation that depends on **data distribution statistics** (mean, median, standard deviation, min/max, frequency maps, target encoding, outlier clipping boundaries, or feature selection) **MUST be fitted strictly on the training set (`X_train`)** and then applied (mapped/transformed) to `X_test`.
> 
> * **Leakage Bug:** Computing `freq_map = df['col'].value_counts(normalize=True)` on the whole DataFrame before splitting.
> * **Leak-Free Fix:** Split into `X_train, X_test` first. Compute `freq_map = X_train['col'].value_counts(normalize=True).to_dict()`, then map onto `X_train` and `X_test`.

## 1. Numerical Transformations

```python
# 1. Ratio Features
df["debt_to_income"] = df["total_debt"] / (df["annual_income"] + 1.0)
df["cost_per_unit"]  = df["total_cost"] / (df["quantity"] + 1e-5)

# 2. Log Transformation (Fixes right-skewed fat tails like Income, Prices)
# np.log1p(x) == log(1 + x) -> Handles 0 safely without returning -inf!
df["log_income"] = np.log1p(df["annual_income"])

# 3. Difference & Growth
df["income_change"] = df["income_2024"] - df["income_2023"]
```

---

## 2. Categorical Encoding

```python
# 1. Nominal (One-Hot Encoding in Pandas)
# Note: drop_first=True avoids the dummy variable trap (multicollinearity in linear models)
df_ohe = pd.get_dummies(df, columns=["city", "gender"], drop_first=True, dtype=int)

# 2. Ordinal Encoding (Mapping ordered ranks)
size_order = {"Small": 1, "Medium": 2, "Large": 3, "XL": 4}
df["size_encoded"] = df["size"].map(size_order)

# 3. Frequency / Count Encoding (Great for high-cardinality zip codes / IDs)
freq_map = df["zip_code"].value_counts(normalize=True)
df["zip_code_freq"] = df["zip_code"].map(freq_map)
```

---

## 3. Datetime Feature Extraction

```python
def extract_dt_features(df, col="timestamp"):
    df[col] = pd.to_datetime(df[col])
    df[f"{col}_year"]        = df[col].dt.year
    df[f"{col}_month"]       = df[col].dt.month
    df[f"{col}_day"]         = df[col].dt.day
    df[f"{col}_dayofweek"]   = df[col].dt.dayofweek
    df[f"{col}_is_weekend"]  = df[col].dt.dayofweek.isin([5, 6]).astype(int)
    df[f"{col}_hour"]        = df[col].dt.hour
    
    # Cyclical Encoding for periodic features (Hour: 0..23, Month: 1..12)
    # Preserves the fact that hour 23 is close to hour 0!
    df[f"{col}_hour_sin"] = np.sin(2 * np.pi * df[f"{col}_hour"] / 24.0)
    df[f"{col}_hour_cos"] = np.cos(2 * np.pi * df[f"{col}_hour"] / 24.0)
    return df
```

---

## 4. Binning & Discretization

```python
# pd.cut: Equal-width bins OR custom explicit intervals
age_bins = [0, 18, 35, 60, 100]
age_labels = ["Minor", "Young Adult", "Middle Aged", "Senior"]
df["age_group"] = pd.cut(df["age"], bins=age_bins, labels=age_labels, right=True)

# pd.qcut: Equal-frequency (Quantile) bins (e.g. Quartiles Q1..Q4)
df["income_quartile"] = pd.qcut(df["annual_income"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
```

---

## 5. Outlier Detection & Capping

```python
# IQR (Interquartile Range) Method
def cap_outliers_iqr(series: pd.Series, factor=1.5) -> pd.Series:
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    return series.clip(lower=lower_bound, upper=upper_bound)

# In-place capping of skewed features:
df["salary_capped"] = cap_outliers_iqr(df["salary"])
```

---

## 6. Scikit-Learn Preprocessing Essentials

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Define Feature Sets
X = df.drop(columns=["target_salary"])
y = df["target_salary"]

# 1. Zero-Leakage Train/Test Split FIRST!
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=None # use stratify=y for classification
)

# 2. Define Pipelines per feature modality
numeric_features = ["age", "experience", "test_score"]
categorical_features = ["department", "education_level"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# 3. Combine with ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# 4. Strict Fit on Train, Transform on Test (ZERO LEAKAGE!)
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed  = preprocessor.transform(X_test)
```

---

# PART 6 — COMPLETE DATA SCIENCE PIPELINE TEMPLATES

---

Companies test candidate capabilities in two distinct ways:
1. **Industry Production Standard:** Clean Scikit-Learn `Pipeline` + `ColumnTransformer` (prevents test data leakage and encapsulates full inference).
2. **From-Scratch / Low-Level OA Standard:** Pure NumPy + Pandas (tests underlying vectorization, linear algebra, memory control, and mathematical intuition without helper libraries).

---

## 6.1 Production Scikit-Learn Pipeline

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- 1. DATA INGESTION & TYPE SPECIFICATION ---
def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

# --- 2. FEATURE ENGINEERING FUNCTION ---
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Missing value indicators
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().sum() > 0:
            df[f"{col}_isnan"] = df[col].isna().astype(int)
        
    # Datetime extraction
    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"])
        df["order_month"] = df["order_date"].dt.month
        df["order_dayofweek"] = df["order_date"].dt.dayofweek
        df["is_weekend"] = df["order_dayofweek"].isin([5, 6]).astype(int)
        df.drop(columns=["order_date"], inplace=True)
        
    # Domain ratios
    if "revenue" in df.columns and "units" in df.columns:
        df["price_per_unit"] = df["revenue"] / (df["units"] + 1e-5)
        
    return df

# --- 3. MAIN PIPELINE EXECUTION ---
def run_sklearn_pipeline():
    # A. Load & Clean
    raw_df = load_data("data.csv")
    clean_df = engineer_features(raw_df)
    
    # B. Define X and y
    target_col = "target_variable"
    X = clean_df.drop(columns=[target_col])
    y = clean_df[target_col]
    
    # C. Identify column types
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    
    # D. Train/Test Split (Strict Partitioning)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    
    # E. Construct Preprocessing Pipeline
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])
    
    # F. Construct Full Model Pipeline
    full_model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    # G. Train Model (Fits preprocessor & model simultaneously on X_train)
    full_model.fit(X_train, y_train)
    
    # H. Evaluate on Blind Test Set
    y_pred = full_model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    
    print("\n" + "=" * 45)
    print("SCIKIT-LEARN PIPELINE RESULTS (TEST SET)")
    print(f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")
    print("=" * 45)
    
    return full_model
```

---

## 6.2 Pure NumPy & Pandas From-Scratch End-to-End Pipeline

This complete template performs **zero-leakage ingestion, missing value imputation, one-hot encoding, train/test splitting, standard scaling, vectorized model training, and metric calculations using ONLY NumPy and Pandas**.

```python
import numpy as np
import pandas as pd

# =========================================================================
# STEP 1: LOAD RAW DATA & SPLIT FIRST (Zero-Leakage Guarantee)
# =========================================================================
def load_and_split(filepath: str, target_col: str, test_ratio: float = 0.20, random_seed: int = 42):
    df = pd.read_csv(filepath)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Split using pure NumPy indices (Strict partition BEFORE any calculations!)
    np.random.seed(random_seed)
    indices = np.random.permutation(len(df))
    split_boundary = int(len(df) * (1.0 - test_ratio))
    
    train_idx = indices[:split_boundary]
    test_idx  = indices[split_boundary:]
    
    df_train = df.iloc[train_idx].copy().reset_index(drop=True)
    df_test  = df.iloc[test_idx].copy().reset_index(drop=True)
    
    return df_train, df_test, target_col

# =========================================================================
# STEP 2: PREPROCESSING (Learn stats on Train -> Apply to Train & Test)
# =========================================================================
class FromScratchPreprocessor:
    def __init__(self):
        self.num_medians = {}
        self.cat_categories = {}
        self.scaler_mean = None
        self.scaler_std = None
        self.feature_columns = []

    def fit_transform(self, df_train: pd.DataFrame, num_cols: list, cat_cols: list) -> np.ndarray:
        df = df_train.copy()

        # 1. Learn & Apply Numeric Imputation (Median)
        for col in num_cols:
            self.num_medians[col] = df[col].median()
            df[col] = df[col].fillna(self.num_medians[col])

        # 2. Learn & Apply Categorical One-Hot Encoding
        for col in cat_cols:
            unique_vals = sorted(df[col].dropna().unique().tolist())
            self.cat_categories[col] = unique_vals
            for val in unique_vals:
                df[f"{col}_{val}"] = (df[col] == val).astype(float)
            df.drop(columns=[col], inplace=True)

        # Record engineered feature column ordering
        self.feature_columns = [c for c in df.columns if c in num_cols or any(c.startswith(f"{cat}_") for cat in cat_cols)]
        X_matrix = df[self.feature_columns].values.astype(float)

        # 3. Learn & Apply Standard Scaling: (X - mu) / (sigma + eps)
        self.scaler_mean = np.mean(X_matrix, axis=0)
        self.scaler_std  = np.std(X_matrix, axis=0)
        self.scaler_std[self.scaler_std == 0.0] = 1.0 # prevent zero division

        X_scaled = (X_matrix - self.scaler_mean) / self.scaler_std
        return X_scaled

    def transform(self, df_test: pd.DataFrame, num_cols: list, cat_cols: list) -> np.ndarray:
        df = df_test.copy()

        # 1. Apply learned medians to test data
        for col in num_cols:
            df[col] = df[col].fillna(self.num_medians[col])

        # 2. Apply learned categories (handles unseen test categories by setting them to 0.0)
        for col, unique_vals in self.cat_categories.items():
            for val in unique_vals:
                df[f"{col}_{val}"] = (df[col] == val).astype(float)
            df.drop(columns=[col], inplace=True)

        # 3. Align columns and apply learned training scale parameters
        X_matrix = df[self.feature_columns].values.astype(float)
        X_scaled = (X_matrix - self.scaler_mean) / self.scaler_std
        return X_scaled

# =========================================================================
# STEP 3: FROM-SCRATCH METRIC EVALUATION
# =========================================================================
def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray):
    mae  = np.mean(np.abs(y_true - y_pred))
    mse  = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-15))
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy  = (tp + tn) / (tp + tn + fp + fn + 1e-15)
    precision = tp / (tp + fp + 1e-15)
    recall    = tp / (tp + fn + 1e-15)
    f1        = 2 * (precision * recall) / (precision + recall + 1e-15)
    cm        = np.array([[tn, fp], [fn, tp]])
    return {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1, "CM": cm}

# =========================================================================
# STEP 4: END-TO-END EXECUTION FUNCTION
# =========================================================================
def run_from_scratch_pipeline():
    # 1. Ingest and strictly split
    df_train, df_test, target_col = load_and_split("data.csv", target_col="target_variable")

    num_cols = ["age", "income", "tenure"]
    cat_cols = ["department", "tier"]

    # 2. Extract targets
    y_train = df_train[target_col].values.astype(float)
    y_test  = df_test[target_col].values.astype(float)

    # 3. Fit preprocessing on train, transform both
    preprocessor = FromScratchPreprocessor()
    X_train_scaled = preprocessor.fit_transform(df_train, num_cols, cat_cols)
    X_test_scaled  = preprocessor.transform(df_test, num_cols, cat_cols)

    # 4. Train from-scratch model (e.g. LinearRegressionNumPy or LogisticRegressionNumPy)
    model = LinearRegressionNumPy(lr=0.01, n_iters=1000)
    model.fit(X_train_scaled, y_train)

    # 5. Predict and compute metrics
    y_pred = model.predict(X_test_scaled)
    results = evaluate_regression(y_test, y_pred)

    print("\n" + "=" * 45)
    print("FROM-SCRATCH NUMPY PIPELINE RESULTS")
    print(f"RMSE: {results['RMSE']:.4f} | MAE: {results['MAE']:.4f} | R²: {results['R2']:.4f}")
    print("=" * 45)

    return model
```

---

```
  Data Preparation
         │
         ▼
  Feature Engineering
         │
         ▼
  Preprocessing
         │
         ▼
  Model Selection
         │
         ▼
  Training (fit)
         │
         ▼
  Evaluation (metrics)
         │
         ▼
  Cross-Validation (stability)
         │
         ▼
  Hyperparameter Tuning (GridSearchCV)
```

---

# PART 7 — MACHINE LEARNING MODELS & SCIKIT-LEARN

---

## 7.1 ML Problem Types

Before touching Scikit-Learn code, immediately identify which of the 4 problem spaces you are solving:

| Problem Type | Target Variable ($y$) | When is it used? (Real-World Example) | Typical Baseline Models | Evaluation Metrics |
| :--- | :--- | :--- | :--- | :--- |
| **Regression** | **Continuous** real number ($-\infty, +\infty$) | Predicting house sale price, battery remaining cycles, customer lifetime value ($) | Linear Regression, Ridge/Lasso, Random Forest Regressor | MAE, MSE, RMSE, $R^2$ |
| **Binary Classification** | **Discrete binary** label ($0$ or $1$, True/False) | Credit card fraud detection (1=Fraud, 0=Legit), disease diagnosis, customer churn | Logistic Regression, Decision Tree, Random Forest | Accuracy, Precision, Recall, F1, ROC-AUC |
| **Multiclass Classification** | **Multiple discrete** categories ($3+$ mutually exclusive classes) | Handwritten digit classification ($0..9$), sentiment analysis (Pos/Neu/Neg) | Logistic Regression (Multinomial/OvR), Random Forest, XGBoost | Macro/Micro F1, Multi-class Log-Loss |
| **Clustering (Unsupervised)** | **No target label** (find hidden geometric groupings) | Customer market segmentation, document topic discovery, anomaly grouping | K-Means, DBSCAN, Hierarchical Clustering | Silhouette Score, Inertia (Elbow) |

---

## 7.2 Universal 7-Step ML Workflow

Every estimator in Scikit-Learn adheres strictly to the **Universal 7-Step ML Workflow**:

```
 ┌───────────────┐     ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
 │ 1. Clean Data │ ──► │ 2. Split X/y  │ ──► │ 3. Train/Test │ ──► │ 4. Preprocess │
 └───────────────┘     └───────────────┘     └───────────────┘     └───────────────┘
                                                                           │
 ┌───────────────┐     ┌───────────────┐     ┌───────────────┐             │
 │ 7. Evaluate   │ ◄── │  6. Predict   │ ◄── │ 5. Fit Model  │ ◄───────────┘
 └───────────────┘     └───────────────┘     └───────────────┘
```

### Pattern A: Modern Pipeline Best Practice (Recommended for Zero Leakage)
```python
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Define Features (X) and Target (y)
X = df.drop("target", axis=1) # 2D Matrix (DataFrame / 2D NumPy array)
y = df["target"]              # 1D Vector (Series / 1D NumPy array)

# 2. Strict Train / Test Split FIRST
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 3. Encapsulate Preprocessing + Model in a Pipeline (Prevents Test Leakage!)
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(random_state=42))
])

# 4. Fit pipeline on training data (automatically runs scaler.fit_transform -> model.fit)
pipeline.fit(X_train, y_train)

# 5. Predict on test data (automatically runs scaler.transform -> model.predict)
y_pred = pipeline.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

### Pattern B: Manual Step-by-Step Template (For Fast Prototyping)
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # Fit & transform on train
X_test_scaled  = scaler.transform(X_test)      # ONLY transform on test!

model = ModelName(hyperparameter=value)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
```

### The 4 Core Estimator Methods Explained
* **`model.fit(X_train, y_train)`:** The learning step. Computes mathematical parameters (e.g., OLS normal equation weights, decision tree split thresholds, or cluster centroids). Modifies the model in-place and returns `self`.
* **`model.predict(X_test)`:** Applies learned parameters to new, unseen feature rows. Returns a 1D NumPy array of predicted continuous numbers (regression) or predicted class labels (classification).
* **`model.predict_proba(X_test)`:** *(Classification only)* Returns an $(N, K)$ matrix of predicted class probabilities summing to 1.0 per row. `[:, 1]` extracts the positive class probability $P(y=1|X)$.
* **`model.score(X_test, y_test)`:** Default evaluation shortcut. Returns **$R^2$ score** for regression and **Accuracy** for classification.

---

## 7.3 Linear Regression ⭐⭐⭐

### Concept & Intuition
Linear Regression models the target variable $y$ as a linear combination of input features:
$$\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_p x_p = \mathbf{w}^T \mathbf{x} + b$$
It optimizes the weights by finding the hyperplane that minimizes the **Sum of Squared Residuals (Ordinary Least Squares / OLS)**: $\sum (y_i - \hat{y}_i)^2$.

### When to Use It
* Predicting continuous targets where relationships between features and output are predominantly additive/linear.
* When maximum **interpretability** is required (stakeholders must understand the exact unit effect of each variable).
* Baseline benchmarking before deploying complex non-linear ensembles.

### Linear Regression Assumptions — What Matters for What?

| Assumption | What It Means | Why It Matters / Practical Consequence if Violated |
| :--- | :--- | :--- |
| **1. Linearity** | Expected target is a linear combination of features | **Core requirement for unbiased predictions.** If violated, OLS line fails to capture curvature (fix: polynomial terms, log-transform). |
| **2. Independent Errors** | Residuals are uncorrelated (no autocorrelation) | **Essential for valid standard errors & $p$-values.** In time-series or clustered data, positive correlation artificially deflates standard errors, causing false positives (fix: time-series models, lag features). |
| **3. Homoscedasticity** | Residual variance is constant across all predictions | **Ensures OLS is BLUE (Best Linear Unbiased Estimator).** If heteroscedastic (cone-shaped residuals), point predictions remain unbiased, but standard errors & confidence intervals are wrong (fix: log-transform $y$, robust standard errors). |
| **4. Normal Residuals** | Residuals are normally distributed around mean 0 | **Mainly needed for exact small-sample hypothesis tests ($t, F$) and confidence intervals ($N < 30$).** For large sample sizes, the Central Limit Theorem guarantees asymptotically valid inference regardless. It is NOT required for OLS to make accurate point predictions! |
| **5. No Perfect Multicollinearity** | Features are not exact linear combinations of each other | **Mathematical invertibility of $(X^T X)$.** Perfect collinearity prevents unique coefficient estimation. High (imperfect) collinearity inflates standard errors, making individual coefficients unstable and uninterpretable (fix: VIF analysis, Ridge/Lasso). |

### Approach A: Exact Scikit-Learn Code
```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Instantiate & Fit
lr = LinearRegression()
lr.fit(X_train, y_train)

# 2. Predict
y_pred = lr.predict(X_test)

# 3. Evaluate Metrics
mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.4f}")

# 4. Inspect Learned Parameters
print("Intercept (b):", lr.intercept_) # Baseline value when all X=0
print("Coefficients (w):", lr.coef_)   # Slope per feature: delta y for +1 unit of X_j

# Inspect feature weights with names:
coef_df = pd.DataFrame({"Feature": X.columns, "Weight": lr.coef_}).sort_values(by="Weight", ascending=False)
```

---

### Approach B: Pure NumPy From-Scratch Implementation (OA & Interview Must-Know)

```
        FORWARD PASS                  MSE LOSS                     VECTORIZED GRADIENT
    y_hat = X @ w + b   ──►   Loss = (1/N) * ||y_hat - y||^2  ──►  dw = (2/N) * X^T @ (y_hat - y)
                                                                   db = (2/N) * sum(y_hat - y)
```

#### Method 1: Vectorized Batch Gradient Descent (`LinearRegressionNumPy`)
```python
import numpy as np

class LinearRegressionNumPy:
    def __init__(self, lr: float = 0.01, n_iters: int = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        
        # 1. Initialize weights to zeros and bias to 0.0
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for epoch in range(self.n_iters):
            # 2. Forward pass: Linear prediction (N,)
            y_pred = np.dot(X, self.weights) + self.bias
            error = y_pred - y # (N,)

            # 3. Compute Mean Squared Error (MSE)
            loss = np.mean(error ** 2)
            self.loss_history.append(loss)

            # 4. Analytical Vectorized Gradients
            # dw = (2 / N) * X^T @ (y_pred - y) -> Shape: (p,)
            # db = (2 / N) * sum(y_pred - y)    -> Scalar
            dw = (2.0 / n_samples) * np.dot(X.T, error)
            db = (2.0 / n_samples) * np.sum(error)

            # 5. Gradient Descent Update
            self.weights -= self.lr * dw
            self.bias    -= self.lr * db

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.weights) + self.bias

# Usage:
# model = LinearRegressionNumPy(lr=0.01, n_iters=1000)
# model.fit(X_train_scaled, y_train)
# y_pred = model.predict(X_test_scaled)
```

#### Method 2: Closed-Form Normal Equation ($\mathbf{w} = (X_b^T X_b)^{-1} X_b^T \mathbf{y}$)
```python
def normal_equation_solve(X: np.ndarray, y: np.ndarray):
    # Append a bias column of 1s to feature matrix X: shape (N, p+1)
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    
    # theta = (X_b^T @ X_b)^(-1) @ X_b^T @ y (using pinv for numerical stability)
    theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
    
    bias = theta[0]
    weights = theta[1:]
    return weights, bias
```

### Common Traps & Mistakes
* **Trap 1: Confusing `lr.coef_` with predictions.** `lr.coef_` is the learned weight vector $(p,)$, NOT the output $\hat{y}$.
* **Trap 2: Using classification metrics (Accuracy/F1) on regression.** Linear Regression outputs continuous floats; computing accuracy throws an error or yields 0.0%.
* **Trap 3: Not scaling when interpreting relative feature importance.** If feature $A$ is in grams ($0..10000$) and feature $B$ is in kilograms ($0..10$), their unscaled coefficient magnitudes cannot be compared directly!

> **🧠 Active Recall Check:** Close your eyes or open a scratch terminal. Write the 4 lines of code to instantiate, fit, predict, and compute RMSE + $R^2$ for Linear Regression.

---

## 7.4 Ridge & Lasso Regularization ⭐⭐

### Concept & Intuition
When features are correlated or $p \approx n$, standard OLS Linear Regression overfits by learning massive, unstable positive and negative weights that cancel each other out. **Regularization** adds a mathematical penalty to the loss function to penalize large weights.

$$\text{Loss}_{\text{Ridge (L2)}} = \text{MSE} + \alpha \sum_{j=1}^p w_j^2 \quad \quad \text{Loss}_{\text{Lasso (L1)}} = \text{MSE} + \alpha \sum_{j=1}^p |w_j|$$

```
   RIDGE PENALTY (L2 Ball)                LASSO PENALTY (L1 Diamond)
   Shrinks weights smoothly toward 0       Forces small/unimportant weights 
   (never sets weights to EXACT zero)      to EXACTLY ZERO (Feature Selection!)
            w2                                      w2
            │   * OLS unconstrained                 │   * OLS unconstrained
         ╭──┼──╮                                   ╱│╲
        │   │   │                                 ╱ │ ╲
    ────┼───●───┼──── w1                     ────●──┼──●──── w1
        │   │   │                                 ╲ │ ╲   (Spiky corners hit axes
         ╰──┼──╯                                   ╲│╱     -> exact zeros!)
```

### Ridge vs. Lasso Decision Table

| Characteristic | Ridge Regression (L2) | Lasso Regression (L1) | ElasticNet (L1 + L2) |
| :--- | :--- | :--- | :--- |
| **Penalty Term** | $\alpha \sum w_j^2$ (Squared weights) | $\alpha \sum \|w_j\|$ (Absolute weights) | $r \alpha \|w\|_1 + \frac{1-r}{2} \alpha \|w\|_2^2$ |
| **Weight Effect** | Shrinks coefficients asymptotically toward 0 | Drives redundant coefficients to **exactly 0.0** | Blends shrinkage + sparsity |
| **Feature Selection** | **No** (all features kept in model) | **Yes** (acts as automated feature selector) | **Yes** |
| **Correlated Features** | Distributes weight evenly among them | Arbitrarily picks one and zeros the others | Groups correlated features together |
| **Mandatory Requirement** | **Must standardize features (`StandardScaler`)** | **Must standardize features (`StandardScaler`)** | **Must standardize features** |

### Exact Scikit-Learn Code
```python
from sklearn.linear_model import Ridge, Lasso

# alpha controls penalty strength:
# alpha=0 -> Identical to standard OLS
# large alpha -> Heavy penalty, weights shrink, model becomes simpler (higher bias, lower variance)
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.1)

# Fit on scaled data!
ridge.fit(X_train_scaled, y_train)
lasso.fit(X_train_scaled, y_train)

# Check zeroed-out features in Lasso:
zero_features = X.columns[lasso.coef_ == 0.0]
print(f"Lasso dropped {len(zero_features)} useless features: {list(zero_features)}")
```

---

## 7.5 Logistic Regression ⭐⭐⭐

### Concept & Intuition
Despite its name, **Logistic Regression is a CLASSIFICATION algorithm**. It models the probability that an observation belongs to the positive class ($y=1$) by passing a linear equation through the **Sigmoid (Logistic) Function**:

$$z = \mathbf{w}^T \mathbf{x} + b \quad \implies \quad P(y=1|\mathbf{x}) = \sigma(z) = \frac{1}{1 + e^{-z}}$$

```
                      THE SIGMOID ACTIVATION
        P(y=1)
         1.0 ┼                                    ╭─────────
             │                                   ╱
         0.5 ┼─────────────────●────────────────╯  <-- Decision Threshold (0.5)
             │                ╱
         0.0 ┼───────────────╯
             └─────────────────┼─────────────────── z = w^T x + b
                              z=0 (Boundary)
```

* If $z \ge 0 \implies \sigma(z) \ge 0.5 \implies$ Predict **Class 1**.
* If $z < 0 \implies \sigma(z) < 0.5 \implies$ Predict **Class 0**.

### Approach A: Exact Scikit-Learn Code
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

# 1. Instantiate with increased max_iter (prevents convergence warnings)
log_reg = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
# C is inverse regularization strength: Smaller C = stronger regularization!

# 2. Fit
log_reg.fit(X_train_scaled, y_train)

# 3. Predict Classes vs. Probabilities
y_pred = log_reg.predict(X_test_scaled)          # Discrete 0 or 1 (threshold = 0.5)
y_prob = log_reg.predict_proba(X_test_scaled)[:, 1] # Probability of Class 1: P(y=1)

# 4. Standard Classification Evaluation Block
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 5. Inspect Log-Odds Weights
print("Intercept:", log_reg.intercept_)
print("Coefficients:", log_reg.coef_)
```

---

### Approach B: Pure NumPy From-Scratch Implementation (Vectorized Gradient Descent)

```
        LINEAR COMBO                SIGMOID ACTIVATION            LOG-LOSS (CROSS-ENTROPY)
      z = X @ w + b     ──►    p_hat = 1 / (1 + e^-z)   ──►  L = -(1/N) * sum(y*log(p) + (1-y)*log(1-p))
                                                                          │
                                                                          ▼
                                                                VECTORIZED GRADIENTS
                                                             dw = (1/N) * X^T @ (p_hat - y)
                                                             db = (1/N) * sum(p_hat - y)
```

```python
import numpy as np

class LogisticRegressionNumPy:
    def __init__(self, lr: float = 0.05, n_iters: int = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        # Clip z to prevent exp overflow runtime warnings in float64
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        
        # 1. Initialize weights to zeros and bias to 0.0
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for epoch in range(self.n_iters):
            # 2. Linear combination + Sigmoid activation
            z = np.dot(X, self.weights) + self.bias
            p_hat = self._sigmoid(z) # Predicted probabilities P(y=1)

            # 3. Binary Cross-Entropy Loss (Log-Loss)
            eps = 1e-15 # Guard against log(0)
            p_hat_safe = np.clip(p_hat, eps, 1.0 - eps)
            loss = -np.mean(y * np.log(p_hat_safe) + (1.0 - y) * np.log(1.0 - p_hat_safe))
            self.loss_history.append(loss)

            # 4. Analytical Vectorized Gradients
            # dw = (1 / N) * X^T @ (p_hat - y) -> Shape: (p,)
            # db = (1 / N) * sum(p_hat - y)    -> Scalar
            dw = (1.0 / n_samples) * np.dot(X.T, (p_hat - y))
            db = (1.0 / n_samples) * np.sum(p_hat - y)

            # 5. Gradient Descent Update
            self.weights -= self.lr * dw
            self.bias    -= self.lr * db

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

# Usage:
# clf = LogisticRegressionNumPy(lr=0.1, n_iters=1000)
# clf.fit(X_train_scaled, y_train)
# y_prob = clf.predict_proba(X_test_scaled)
# y_pred = clf.predict(X_test_scaled, threshold=0.5)
```

### Custom Decision Thresholding Pattern
In high-stakes problems (fraud, cancer), the default 0.5 threshold is often suboptimal:
```python
# Lower threshold to 0.30 to catch MORE fraud (increases Recall, lowers Precision)
custom_threshold = 0.30
y_pred_custom = (y_prob >= custom_threshold).astype(int)
```

---

## 7.6 K-Nearest Neighbors (KNN) ⭐⭐

### Concept & Intuition
A non-parametric, instance-based algorithm. To classify a new point $\mathbf{x}$, it computes the Euclidean distance between $\mathbf{x}$ and all training samples, finds the $k$ closest neighbors, and takes a **majority vote**.

$$\text{Distance}(\mathbf{p}, \mathbf{q}) = \sqrt{\sum_{j=1}^p (p_j - q_j)^2}$$

```
                K-NEAREST NEIGHBORS (k=3 vs k=5)
                          
                          ▲ Class A
                        ▲   ▲
                      ▲   ● ◄── New point
                        ▼   ▼
                      ▼   ▼   ▼ Class B
                k=3: 2 ▲ vs 1 ▼ -> Predicts Class A
                k=5: 2 ▲ vs 3 ▼ -> Predicts Class B
```

### Key Rules to Remember
* **Feature Scaling is 100% MANDATORY:** Distance calculations are completely ruined if one feature has range $0..100,000$ and another is $0..1$.
* **The $k$ Hyperparameter:**
  * **Small $k$ ($k=1$):** High variance / Overfitting. Memorizes individual noisy points and creates erratic decision boundaries.
  * **Large $k$ ($k=50$):** High bias / Underfitting. Oversmooths decision boundaries and simply predicts the majority class.
  * **Rule of thumb:** Choose an odd number (e.g., $k=3, 5, 7$) to prevent 50/50 voting ties in binary classification.

### Exact Scikit-Learn Code
```python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# Classification
knn_clf = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2) # p=2 -> Euclidean
knn_clf.fit(X_train_scaled, y_train)
y_pred = knn_clf.predict(X_test_scaled)

# Regression (averages targets of k nearest neighbors)
knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_train_scaled, y_train)
y_pred_reg = knn_reg.predict(X_test_scaled)
```

---

## 7.7 Decision Trees ⭐⭐⭐

### Concept & Intuition
Decision Trees recursively partition the feature space into axis-aligned rectangular boxes. At each step, it chooses the feature $j$ and split threshold $t$ that **maximizes purity** (minimizes Gini Impurity or Entropy for classification; minimizes Variance/MSE for regression).

$$\text{Gini Impurity} = 1 - \sum_{k=1}^K p_k^2 \quad \quad \text{Entropy} = - \sum_{k=1}^K p_k \log_2(p_k)$$

```
                     DECISION TREE SPLITTING
                     
                       [ Age <= 30.5 ]
                        /          \
                      YES          NO
                      /              \
               [ Salary <= 50k ]    [ Class 1 ]
                 /          \
            [ Class 0 ]   [ Class 1 ]
```

### The Overfitting Danger & Hyperparameter Guards
An unconstrained tree grows until every leaf contains exactly 1 sample (100% train accuracy, terrible generalization). You **must** constrain it:

| Hyperparameter | What it controls | Effect on Overfitting |
| :--- | :--- | :--- |
| **`max_depth`** | Maximum vertical levels allowed in tree | **Most important.** Lower values ($3..8$) prevent deep memorization. |
| **`min_samples_split`** | Minimum samples required inside a node before it can split | Higher values ($10..50$) stop the tree from splitting on tiny clusters. |
| **`min_samples_leaf`** | Minimum samples that MUST remain in a terminal leaf | Higher values ($5..20$) smooth out noisy leaf predictions. |
| **`max_features`** | Number of features to consider when looking for the best split | Subsampling features decorrelates nodes. |

### Exact Scikit-Learn Code
```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Classification Tree
dt_clf = DecisionTreeClassifier(
    criterion="gini",         # 'gini' or 'entropy'
    max_depth=5,              # Restrict depth to avoid overfitting
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
dt_clf.fit(X_train, y_train) # Decision trees do NOT require feature scaling!
y_pred = dt_clf.predict(X_test)

# Regression Tree
dt_reg = DecisionTreeRegressor(max_depth=5, random_state=42)
dt_reg.fit(X_train, y_train)
y_pred_reg = dt_reg.predict(X_test)
```

---

## 7.8 Random Forest ⭐⭐⭐

### Concept & Intuition
A single Decision Tree has **low bias but very high variance** (sensitive to tiny changes in data). Random Forest fixes this using **Bagging (Bootstrap Aggregation) + Random Feature Subspacing**:

1. **Bootstrap Sampling:** Creates $B$ different training datasets by sampling $N$ rows *with replacement* from the original data.
2. **Feature Subspacing:** At every single split, considers only a random subset of features (typically $\sqrt{p}$).
3. **Ensemble Averaging:** Trains $B$ unpruned, deep trees in parallel. The final prediction is a **majority vote** (classification) or **mean average** (regression).

$$\text{Variance}_{\text{Ensemble}} = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2$$
*As $B \to \infty$, the second term vanishes. Because feature subspacing reduces tree correlation $\rho$, total variance drops dramatically without increasing bias!*

```
                     RANDOM FOREST ARCHITECTURE
                     
                            [ Training Data ]
                          /        │         \
                   Bootstrap 1  Bootstrap 2  Bootstrap B
                        │          │            │
                     Tree 1     Tree 2       Tree B
                     (Tree)     (Tree)       (Tree)
                        │          │            │
                     Pred: 1    Pred: 1      Pred: 0
                          \        │         /
                           [ MAJORITY VOTE ]
                                   │
                              Final: 1
```

### Exact Scikit-Learn Code
```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# 1. Instantiate Random Forest Classifier
rf_clf = RandomForestClassifier(
    n_estimators=200,         # Number of trees in the forest (100-500 standard)
    max_depth=10,             # Maximum depth of each tree
    min_samples_leaf=4,       # Prevent single-sample leaves
    n_jobs=-1,                # Utilize all available CPU cores in parallel!
    random_state=42
)

# 2. Fit & Predict
rf_clf.fit(X_train, y_train)
y_pred = rf_clf.predict(X_test)
y_prob = rf_clf.predict_proba(X_test)[:, 1]

# 3. Extract Feature Importances (Mean Impurity Decrease)
feat_imp = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_clf.feature_importances_
}).sort_values(by="Importance", ascending=False)
print("Top 5 Predictive Features:\n", feat_imp.head(5))

# Regression Equivalent
rf_reg = RandomForestRegressor(n_estimators=200, max_depth=10, n_jobs=-1, random_state=42)
rf_reg.fit(X_train, y_train)
```

---

## 7.9 Gradient Boosting (GBM / XGBoost)

### Concept: Bagging vs. Boosting Mental Model

* **Random Forest (Bagging):** Trees are built **independently in parallel**. Each tree tries to predict the target $y$. Their results are averaged to reduce variance.
* **Gradient Boosting (Boosting):** Trees are built **sequentially in a chain**. Each new tree is trained to predict the **residuals (errors)** of all previous trees combined.

$$\hat{y}^{(m)}(\mathbf{x}) = \hat{y}^{(m-1)}(\mathbf{x}) + \eta \cdot \text{Tree}_m(\mathbf{x}, \text{Residuals})$$
*where $\eta$ is the `learning_rate` (shrinkage step size).*

```
                     BOOSTING SEQUENTIAL CHAIN
                     
   [ Tree 1 ] ──► Computes Residuals ──► [ Tree 2 ] ──► Computes Residuals ──► [ Tree 3 ]
   (Base Pred)    (y - y_hat_1)          (Fixes Errors) (y - y_hat_2)          (Refines)
```

### Exact Scikit-Learn Code
```python
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

# Scikit-Learn Native Implementation
gbm = GradientBoostingClassifier(
    n_estimators=100,         # Number of sequential boosting stages
    learning_rate=0.1,        # Shrinkage factor (lower = requires more trees, more robust)
    max_depth=3,              # Shallow trees (stumps) work best in boosting!
    random_state=42
)
gbm.fit(X_train, y_train)
y_pred = gbm.predict(X_test)

# Industry Standard Equivalent: XGBoost / LightGBM
# import xgboost as xgb
# model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
```

---

## 7.10 Support Vector Machines (SVM)

### Concept & Intuition
SVM finds the unique decision hyperplane that separates two classes while maximizing the **Margin** (geometric distance to the nearest training points, known as **Support Vectors**).

* **Hard Margin:** Requires 100% linear separability with 0 errors (easily breaks on noisy data).
* **Soft Margin ($C$ hyperparameter):** Permits controlled margin violations:
  * **Large $C$:** Severe penalty on violations $\implies$ Narrow margin, complex boundary, risk of **overfitting**.
  * **Small $C$:** Lenient penalty on violations $\implies$ Wide margin, simpler boundary, risk of **underfitting**.
* **Kernel Trick:** Projects non-linearly separable inputs into a higher-dimensional space where a linear hyperplane exists without computing coordinates explicitly (e.g., **Radial Basis Function / RBF Kernel**).

### Exact Scikit-Learn Code
```python
from sklearn.svm import SVC, SVR

# SVM Classification (Feature Scaling is 100% Mandatory!)
svm = SVC(
    C=1.0,                    # Regularization parameter
    kernel="rbf",             # 'linear', 'poly', 'rbf', 'sigmoid'
    gamma="scale",            # Kernel coefficient (higher gamma = tighter RBF envelopes)
    probability=True,         # Enables predict_proba via internal Platt scaling
    random_state=42
)
svm.fit(X_train_scaled, y_train)
y_pred = svm.predict(X_test_scaled)
y_prob = svm.predict_proba(X_test_scaled)[:, 1]
```

---

## 7.11 K-Means Clustering

### Concept & Intuition
An **unsupervised algorithm** that partitions unlabelled data into $k$ distinct, non-overlapping clusters.
1. Randomly initializes $k$ centroids.
2. **Assignment Step:** Assigns every data point to its nearest centroid (Euclidean distance).
3. **Update Step:** Recomputes each centroid as the mathematical mean of all points assigned to it.
4. Repeats until centroids stabilize (convergence).

$$\text{Inertia (WCSS)} = \sum_{i=1}^N \min_{\mu_j} \|\mathbf{x}_i - \mu_j\|^2$$

```
                         THE ELBOW METHOD
             Inertia
                │
                │  ●  k=1
                │   \
                │    \
                │     ●  k=2
                │      \
                │       ● ◄──── Optimal "Elbow" Point (k=3)
                │         \───●───●───●
                └────────────────────────── k (n_clusters)
```

### Exact Scikit-Learn Code
```python
from sklearn.cluster import KMeans

# 1. Instantiate & Fit (Scaling is MANDATORY for distance calculations!)
kmeans = KMeans(
    n_clusters=3,             # Target number of clusters (k)
    n_init="auto",            # Number of random centroid initializations
    random_state=42
)
cluster_labels = kmeans.fit_predict(X_train_scaled) # Returns cluster IDs: array([0, 2, 1, 0, ...])

# 2. Inspect Cluster Properties
print("Cluster Centroid Coordinates:\n", kmeans.cluster_centers_) # Shape: (k, num_features)
print("Inertia (Within-Cluster Sum of Squares):", kmeans.inertia_)

# 3. Elbow Method Pattern to find Optimal k:
inertias = []
k_range = range(1, 10)
for k in k_range:
    km = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(X_train_scaled)
    inertias.append(km.inertia_)
```

---

# PART 8 — MODEL EVALUATION & VALIDATION

---

## 8.1 Regression Metrics (MAE, MSE, RMSE, R²)

```python
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# y_true = actual test targets, y_pred = model predictions
mae  = mean_absolute_error(y_true, y_pred)
mse  = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_true, y_pred)
```

| Metric | Formula | Output Units | Outlier Sensitivity | When to Use |
| :--- | :--- | :--- | :--- | :--- |
| **MAE** | $\frac{1}{n}\sum \|y - \hat{y}\|$ | Same as target ($) | **Low** (linear penalty) | When business errors cost proportionally; robust to outliers. |
| **MSE** | $\frac{1}{n}\sum (y - \hat{y})^2$ | Target units squared ($^2) | **Very High** (quadratic penalty) | Mathematical optimization loss function. |
| **RMSE** | $\sqrt{\text{MSE}}$ | Same as target ($) | **High** | When large catastrophic errors must be heavily penalized. |
| **$R^2$ Score** | $1 - \frac{\sum(y-\hat{y})^2}{\sum(y-\bar{y})^2}$ | Unitless ($-\infty..1.0$) | Medium | Measuring the % of target variance explained by features. |

---

## 8.2 Classification Metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc
)

acc  = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec  = recall_score(y_true, y_pred)
f1   = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_prob) # Requires predicted probabilities P(y=1)!

# Precision-Recall AUC (PR-AUC / Average Precision)
precision_pts, recall_pts, _ = precision_recall_curve(y_true, y_prob)
pr_auc = auc(recall_pts, precision_pts)
print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
```

### Metric Selection Decision Guide

| Business Situation | Primary Metric | Intuition & Reason |
| :--- | :--- | :--- |
| **Balanced Classes** (50% Cat, 50% Dog) | **Accuracy** | Simple ratio of correct predictions across all samples. |
| **False Positives are very costly** (e.g., Spam Filter deleting legitimate email) | **Precision** | $\frac{\text{TP}}{\text{TP} + \text{FP}}$: When the model predicts Positive, it must be correct. |
| **False Negatives are fatal** (e.g., Cancer screening, Fraud detection) | **Recall** (Sensitivity) | $\frac{\text{TP}}{\text{TP} + \text{FN}}$: Must catch every single actual Positive instance. |
| **Imbalanced Classes where both errors matter** | **F1-Score** | $2 \cdot \frac{\text{Prec} \cdot \text{Rec}}{\text{Prec} + \text{Rec}}$: Harmonic mean balancing precision and recall. |
| **Evaluating ranking on balanced/moderate data** | **ROC-AUC** | Plots TPR vs. FPR across all thresholds. Measures probability that a positive ranks above a negative. |
| **Extreme Class Imbalance** (e.g., 0.1% Fraud, Ad Clicks) | **PR-AUC** | Plots Precision vs. Recall. Ignores True Negatives, preventing huge negative majorities from inflating the score! |

### ROC-AUC vs. PR-AUC (The Imbalanced Data Trap)

* **ROC-AUC (Receiver Operating Characteristic):** Plots True Positive Rate vs. False Positive Rate ($\text{FPR} = \frac{\text{FP}}{\text{TN} + \text{FP}}$).
  * **The Trap:** When the negative class is overwhelming (e.g. 99.9% legit, 0.1% fraud), $\text{TN}$ is huge. A model can produce 5,000 False Positives, yet $\text{FPR}$ remains tiny, yielding a deceptively high ROC-AUC ($>0.98$) despite awful precision.
* **PR-AUC (Precision-Recall Area Under Curve):** Plots Precision vs. Recall.
  * **The Fix:** PR-AUC does NOT include True Negatives anywhere in its formula. It evaluates performance strictly on the positive minority class.

---

## 8.3 Confusion Matrix Visual Breakdown

```
                         PREDICTED CLASS
                      Positive (1)       Negative (0)
                  ┌──────────────────┬──────────────────┐
     Positive (1) │  TRUE POSITIVE   │  FALSE NEGATIVE  │
                  │       (TP)       │   (FN) Type II   │
  ACTUAL          ├──────────────────┼──────────────────┤
  CLASS           │  FALSE POSITIVE  │  TRUE NEGATIVE   │
     Negative (0) │   (FP) Type I    │       (TN)       │
                  └──────────────────┴──────────────────┘
```

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)
# cm[0, 0] = True Negatives (TN)
# cm[0, 1] = False Positives (FP)
# cm[1, 0] = False Negatives (FN)
# cm[1, 1] = True Positives (TP)

tn, fp, fn, tp = cm.ravel()
print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
```

---

## 8.4 Cross-Validation Best Practices

Use **$K$-Fold Cross-Validation** on the training data when model selection or a more stable validation estimate is needed; keep the blind test set untouched for final unbiased evaluation. $K$-Fold CV splits the training data into $K$ equal partitions, trains $K$ times (using $K-1$ folds for training and 1 fold for validation), and averages the validation scores across all folds.

```python
from sklearn.model_selection import cross_val_score

# 5-Fold Cross Validation
scores = cross_val_score(
    estimator=model,
    X=X_train,
    y=y_train,
    cv=5,                     # 5 folds
    scoring="accuracy",       # 'r2', 'neg_root_mean_squared_error', 'f1', 'roc_auc'
    n_jobs=-1
)

print(f"CV Scores per fold: {scores}")
print(f"Mean CV Score: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

---

## 8.5 Hyperparameter Tuning (`GridSearchCV`)

* **Parameters:** Learned internally by model during `.fit()` (e.g., $w, b$).
* **Hyperparameters:** Configured externally before training (e.g., `n_estimators`, `max_depth`, `alpha`, `C`).

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# 1. Define hyperparameter search grid
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_leaf": [2, 5]
}

# 2. Instantiate GridSearch with 5-fold CV
grid = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

# 3. Fit GridSearch (strictly on training set!)
grid.fit(X_train, y_train)

# 4. Extract Best Configuration
print("Best Hyperparameters:", grid.best_params_)
print(f"Best CV Score: {grid.best_score_:.4f}")

# 5. Evaluate Best Model directly on Test Set (using the target metric!)
from sklearn.metrics import f1_score
best_model = grid.best_estimator_
y_test_pred = best_model.predict(X_test)
test_f1 = f1_score(y_test, y_test_pred)
print(f"Test Set F1-Score: {test_f1:.4f}")
```

---

## 8.6 Overfitting vs. Underfitting Diagnostics

```
                   THE BIAS-VARIANCE TRADEOFF
          Error
            │
            │  \                           /  Validation Error (High Variance!)
            │   \     Optimal Balance     /
            │    \          ▼            /
            │     \        ●────────────●
            │      \      /
            │       \    / ─────────────── Training Error
            │        \  /
            │         ●
            └──────────────────────────────────── Model Complexity
                Underfitting                Overfitting
                (High Bias)                (High Variance)
```

| State | Model Complexity | Training Score | Test/Val Score | Primary Diagnosis | What is the Fix? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Underfitting** | Too Simple (e.g. Linear model on complex non-linear curve) | **Low** | **Low** | High Bias | Add features, use non-linear models, reduce regularization ($\downarrow \alpha$). |
| **Good Fit** | Balanced | **High** | **High** | Generalized | Ready for production deployment. |
| **Overfitting** | Too Complex (e.g. Unpruned Tree, $k=1$ KNN) | **High (Near 100%)** | **Low** | High Variance | Add regularization ($\uparrow \alpha$, $\downarrow C$), prune tree depth, get more data, dropout. |

---

# PART 9 — DEEP LEARNING QUICK REFERENCE

---

## 9.1 Multi-Layer Perceptron (MLP) with Keras

```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Regression MLP (Outputs continuous number)
reg_mlp = Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)), # Input layer + Hidden 1
    Dropout(0.2),                                                  # Regularization
    Dense(32, activation="relu"),                                  # Hidden 2
    Dense(1)                                                       # Output: 1 neuron, NO activation!
])
reg_mlp.compile(optimizer="adam", loss="mse", metrics=["mae"])
reg_mlp.fit(X_train_scaled, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=0)
reg_preds = reg_mlp.predict(X_test_scaled)

# Binary Classification Output Layer Comparison:
# Output: Dense(1, activation="sigmoid"), Loss: loss="binary_crossentropy", Metric: metrics=["accuracy"]

# Multi-class Classification Output Layer Comparison:
# Output: Dense(K, activation="softmax"), Loss: loss="sparse_categorical_crossentropy"
```

---

## 9.2 Convolutional Neural Networks (CNN) ⭐⭐

**Core Purpose:** Grid-like spatial data (Images, Spectrograms, Video Frames).

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    # 1. Conv2D: Slides 32 distinct (3x3) learnable filters to extract local spatial patterns (edges, textures)
    Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)),
    
    # 2. MaxPooling2D: Downsamples spatial dimensions by 2x2, reducing computation and adding translation invariance
    MaxPooling2D((2, 2)),
    
    # 3. Flatten: Unrolls 2D feature maps into a 1D vector
    Flatten(),
    
    # 4. Dense: Fully connected layers for non-linear feature combination and final class scoring
    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid") # Binary classification output
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=10, batch_size=32)
```

---

## 9.3 Recurrent Neural Networks (RNN) ⭐⭐

**Core Purpose:** Sequential temporal data where current state depends on past history (Time-series, sensor streams, text).

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# X_train 3D tensor shape: (samples, timesteps, features)
# Note: Keras layer input_shape excludes batch size -> input_shape=(timesteps, features)
model = Sequential([
    SimpleRNN(64, activation="tanh", input_shape=(10, 1)), # 10 timesteps, 1 feature per step
    Dense(1) # Next-step continuous prediction
])
model.compile(optimizer="adam", loss="mse")
model.fit(X_train, y_train, epochs=20, batch_size=32)
```
* **Hidden State ($h_t$):** Vector that acts as the model's memory, passed recursively from timestep $t-1$ to timestep $t$.

---

## 9.4 LSTM & GRU (Long-Term Sequential Memory)

Vanilla RNNs suffer from **Vanishing Gradients**; errors cannot backpropagate across $>10$ timesteps. **LSTMs (Long Short-Term Memory)** fix this using an internal Cell State and 3 additive gating mechanisms:

$$\text{Vanilla RNN (Short Memory)} \quad \longrightarrow \quad \text{LSTM (Deep Memory Gates)} \quad \longrightarrow \quad \text{GRU (Faster 2-Gate Alternative)}$$

```python
from tensorflow.keras.layers import LSTM, GRU

# Drop-in replacement for SimpleRNN with deep memory gates:
lstm_model = Sequential([
    LSTM(64, input_shape=(10, 1)), # Contains Forget, Input, Output gates
    Dense(1)
])

gru_model = Sequential([
    GRU(64, input_shape=(10, 1)),  # Simpler 2-gate architecture (Reset & Update gates)
    Dense(1)
])
```

---


---

# PART 10 — STATISTICS FOR DATA SCIENCE & OAs

Statistics provides the mathematical framework for hypothesis testing, feature importance, distributions, and error analysis in Data Science interviews and online assessments.

```
                       STATISTICAL TOOLKIT MAP
                       
   DESCRIPTIVE STATS       PROBABILITY & DISTRIBUTIONS       INFERENTIAL STATS
  ┌─────────────────┐     ┌───────────────────────────┐     ┌─────────────────┐
  │ Mean vs Median  │     │ Normal (68-95-99.7)       │     │ Central Limit   │
  │ Std vs Variance │     │ Binomial & Poisson        │     │ Hypothesis Test │
  │ IQR & Z-score   │     │ Conditional Probability   │     │ p-values & CIs  │
  │ Skew & Kurtosis │     │ Bayes' Theorem            │     │ Type I / II Err │
  └─────────────────┘     └───────────────────────────┘     └─────────────────┘
```

---

## 10.1 Descriptive Statistics & Dispersion

### 1. Measures of Central Tendency
* **Mean ($\mu$ or $\bar{x}$):** $\frac{1}{n} \sum x_i$. Arithmetic average. **Highly sensitive to outliers.**
* **Median ($Q_2$):** Middle value of sorted data. **Robust to outliers and skewness.**
* **Mode:** Most frequently occurring value. Useful for categorical data.

> **Interview Golden Rule:**
> * Symmetrical Distribution: $\text{Mean} \approx \text{Median} \approx \text{Mode}$.
> * Right-Skewed (Positive, fat right tail, e.g. Income): $\text{Mean} > \text{Median} > \text{Mode}$.
> * Left-Skewed (Negative, fat left tail, e.g. Age of death): $\text{Mean} < \text{Median} < \text{Mode}$.
> * *Always use **Median** for skewed data (salaries, real estate, transaction amounts).*

### 2. Measures of Dispersion (Spread)
* **Variance ($\sigma^2$ or $s^2$):** Average squared deviation from the mean:
  $$\text{Population: } \sigma^2 = \frac{1}{N} \sum_{i=1}^N (x_i - \mu)^2 \quad \quad \text{Sample (Bessel's Correction): } s^2 = \frac{1}{n-1} \sum_{i=1}^n (x_i - \bar{x})^2$$
  *(Note: We divide by $n-1$ for sample variance to provide an **unbiased** estimate of the population variance).*
* **Standard Deviation ($\sigma$ or $s$):** $\sqrt{\text{Variance}}$. Measures spread in the **same physical units** as the original data.
* **Percentiles & Interquartile Range (IQR):**
  * $Q_1$ (25th percentile), $Q_2$ (Median / 50th percentile), $Q_3$ (75th percentile).
  * $\text{IQR} = Q_3 - Q_1$ (Contains the middle 50% of the data). **Robust to outliers.**

---

## 10.2 Standardization, Z-Score & Outlier Detection

### 1. The Z-Score (Standard Score)
Measures how many standard deviations a value $x$ lies away from the mean:
$$z = \frac{x - \mu}{\sigma}$$
* $z = 0 \implies x$ is exactly at the mean.
* $z = +2.0 \implies x$ is 2 standard deviations above the mean.

### 2. Outlier Detection Rules
* **Z-Score Method (for Gaussian/Normal data):** A point is an outlier if $|z| > 3$ ($99.73\%$ of normal data lies within $\pm 3\sigma$).
* **IQR Method (for Skewed/Non-Normal data):**
  $$\text{Lower Bound} = Q_1 - 1.5 \times \text{IQR} \quad \quad \text{Upper Bound} = Q_3 + 1.5 \times \text{IQR}$$

```python
import numpy as np
from scipy import stats

# 1. Z-Score calculation
data = np.array([10, 12, 12, 14, 15, 18, 100]) # 100 is an extreme outlier
z_scores = stats.zscore(data)
outliers_z = data[np.abs(z_scores) > 3]

# 2. IQR calculation
q1, q3 = np.percentile(data, [25, 75])
iqr = q3 - q1
outliers_iqr = data[(data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)]
```

---

## 10.3 Probability & Bayes' Theorem

### 1. Probability Fundamentals
* **Joint Probability:** $P(A \cap B) = P(A \text{ and } B)$ (Probability both occur).
* **Union Probability:** $P(A \cup B) = P(A) + P(B) - P(A \cap B)$.
* **Conditional Probability:** Probability of $A$ given that $B$ has already occurred:
  $$P(A|B) = \frac{P(A \cap B)}{P(B)}$$
* **Independent Events:** $P(A \cap B) = P(A) \cdot P(B) \implies P(A|B) = P(A)$.

### 2. Bayes' Theorem
Updates prior beliefs based on new observed evidence:
$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$
$$\text{Posterior} = \frac{\text{Likelihood} \times \text{Prior}}{\text{Marginal Evidence}}$$

```
                       BAYES' THEOREM MENTAL MODEL
                     
    PRIOR P(Disease)          TEST ACCURACY              POSTERIOR P(Disease | +Test)
    Base disease rate   ──►   Sensitivity P(+ | Dis)  ──► Actual probability you are sick
    (e.g., 1 in 1000)         Specificity P(- | No)      (Often surprisingly low if prior is tiny!)
```

> **Classic Interview Problem (Medical Test Paradox):**
> * A disease affects $1\%$ of the population: $P(D) = 0.01 \implies P(\neg D) = 0.99$.
> * A test is $95\%$ accurate: $P(+|D) = 0.95$ (Sensitivity) and $P(+|\neg D) = 0.05$ (False Positive rate).
> * If a patient tests positive, what is the probability they actually have the disease?
> $$P(D|+) = \frac{P(+|D)P(D)}{P(+|D)P(D) + P(+|\neg D)P(\neg D)} = \frac{0.95 \times 0.01}{(0.95 \times 0.01) + (0.05 \times 0.99)} = \frac{0.0095}{0.0095 + 0.0495} \approx 16.1\%$$
> *Even with a 95% accurate test, there is only a ~16% chance the patient is actually sick because the disease is rare!*

---

## 10.4 Key Statistical Distributions

```
     GAUSSIAN / NORMAL                 BINOMIAL                       POISSON
      Continuous Curve              Discrete Trials               Count over Time
             │                             │                             │
        ╭────┴────╮                     │  │  │  │                    │  │  │
       ╱     │     ╲                   ┌┴──┴──┴──┴┐                  ┌┴──┴──┴──┐
      ╱      │      ╲                  │  n trials│                  │ Lambda rate
     68% within ±1 sigma               p probability                 Events / interval
```

| Distribution | Type | Key Formula / Parameters | Real-World Use Case |
| :--- | :--- | :--- | :--- |
| **Normal (Gaussian)** | Continuous | Parameters: $\mu, \sigma$. Bell-shaped symmetrical. | Heights, measurement errors, test scores, residual errors in linear regression. |
| **Bernoulli** | Discrete | Single trial with binary outcome ($0$ or $1$) with success probability $p$. | Single coin flip, single customer conversion (converted or not). |
| **Binomial** | Discrete | $n$ independent Bernoulli trials: $P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}$. $\mu = np, \sigma^2 = np(1-p)$. | Number of heads in 10 coin flips; number of converted customers out of 100 website visitors. |
| **Poisson** | Discrete | Count of independent events occurring in a fixed interval of time/space with constant rate $\lambda$: $P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}$. $\mu = \lambda, \sigma^2 = \lambda$. | Number of server requests per minute; number of customer support tickets per hour. |

### The 68–95–99.7 Empirical Rule (For Normal Distributions)
* **$\mu \pm 1\sigma$** contains **$68.27\%$** of all data.
* **$\mu \pm 2\sigma$** contains **$95.45\%$** of all data ($1.96\sigma$ for exact 95%).
* **$\mu \pm 3\sigma$** contains **$99.73\%$** of all data.

---

## 10.5 Central Limit Theorem & Confidence Intervals

### 1. Central Limit Theorem (CLT)
> **The Theorem:** Given ANY independent population distribution (even if severely skewed, uniform, or bimodal), the distribution of the **sample means ($\bar{X}$)** approaches a **Normal Distribution** as the sample size $n$ becomes sufficiently large ($n \ge 30$).

$$\bar{X} \sim \mathcal{N}\left(\mu, \frac{\sigma^2}{n}\right) \quad \implies \quad \text{Standard Error of the Mean (SE)} = \frac{\sigma}{\sqrt{n}}$$

### 2. Confidence Interval (CI) for the Mean
Range of values that contains the true population mean $\mu$ with a specified confidence level (typically $95\%$):
$$\text{CI} = \bar{x} \pm z^* \times \frac{s}{\sqrt{n}}$$
*(For 95% confidence: $z^* = 1.96$. For 99% confidence: $z^* = 2.576$. Use $t^*$ if $n < 30$ and $\sigma$ is unknown).*

```python
from scipy import stats
import numpy as np

sample_data = np.array([23, 25, 28, 32, 24, 26, 29, 31, 27, 30])
mean = np.mean(sample_data)
sem = stats.sem(sample_data) # Standard Error = s / sqrt(n)

# 95% Confidence Interval
ci_95 = stats.t.interval(confidence=0.95, df=len(sample_data)-1, loc=mean, scale=sem)
print(f"95% CI: ({ci_95[0]:.2f}, {ci_95[1]:.2f})")
```

---

## 10.6 Hypothesis Testing, p-values & Error Types

### 1. The 5-Step Hypothesis Testing Framework
1. **State Hypotheses:** 
   * $H_0$ (Null Hypothesis): Baseline / No effect / No difference (e.g., $\mu_{\text{treatment}} = \mu_{\text{control}}$).
   * $H_a$ (Alternative Hypothesis): Effect exists (e.g., $\mu_{\text{treatment}} \neq \mu_{\text{control}}$).
2. **Choose Significance Level ($\alpha$):** Typically $\alpha = 0.05$ ($5\%$ threshold for false alarms).
3. **Compute Test Statistic:** (e.g. $t$-statistic, $z$-statistic).
4. **Compute $p$-value:** Probability of observing a test statistic as extreme as (or more extreme than) the one calculated, **assuming $H_0$ is true**.
5. **Make Decision:**
   * **$p \le \alpha \implies$ Reject $H_0$** (Result is statistically significant).
   * **$p > \alpha \implies$ Fail to Reject $H_0$** (Insufficient evidence to claim an effect).

### 2. Type I vs. Type II Errors

```
                                 TRUE REALITY
                       H0 is TRUE               H0 is FALSE
                  ┌──────────────────────┬──────────────────────┐
     Reject H0    │     TYPE I ERROR     │   CORRECT DECISION   │
  DECISION        │ (False Positive, a)  │   (Power = 1 - b)    │
  MADE            ├──────────────────────┼──────────────────────┤
     Fail to      │   CORRECT DECISION   │    TYPE II ERROR     │
     Reject H0    │  (Confidence, 1 - a) │  (False Negative, b) │
                  └──────────────────────┴──────────────────────┘
```
* **Type I Error ($\alpha$):** Rejecting $H_0$ when $H_0$ is true (False Alarm, e.g. innocent person convicted).
* **Type II Error ($\beta$):** Failing to reject $H_0$ when $H_0$ is false (Missed Opportunity, e.g. guilty person goes free).
* **Statistical Power ($1 - \beta$):** Probability of correctly rejecting a false null hypothesis (catching a real effect).

### 3. Python Code for Two-Sample Independent $t$-test
```python
from scipy import stats

# A/B Testing: Conversion rates for Version A vs Version B
group_a = [12, 14, 15, 12, 16, 13, 14]
group_b = [17, 18, 19, 15, 16, 20, 18]

# Two-sample independent t-test (Welch's t-test: equal_var=False)
t_stat, p_val = stats.ttest_ind(group_a, group_b, equal_var=False)
print(f"t-statistic: {t_stat:.4f} | p-value: {p_val:.4e}")

if p_val < 0.05:
    print("Reject H0: Statistically significant difference between groups!")
else:
    print("Fail to Reject H0: No significant difference.")
```

---

## 10.7 Covariance, Pearson vs. Spearman Correlation & VIF

### 1. Covariance vs. Correlation
* **Covariance ($\text{Cov}(X, Y) = \frac{1}{n-1}\sum(x_i - \bar{x})(y_i - \bar{y})$):** Measures the direction of a linear relationship, but value depends on data units ($-\infty..+\infty$).
* **Pearson Correlation ($r$):** Standardized covariance bounded between $[-1.0, +1.0]$. Measures **linear** relationships only.
  $$r = \frac{\text{Cov}(X, Y)}{s_X \cdot s_Y}$$
* **Spearman Rank Correlation ($\rho$):** Computes Pearson correlation on the **ranks** of the data. Measures **monotonic** (increasing or decreasing) relationships, even if non-linear! **Robust to outliers.**

```python
from scipy import stats

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100] # Extreme outlier at end
y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

# Pearson is ruined by the outlier; Spearman handles it gracefully:
r_pearson, _ = stats.pearsonr(x, y)   # Highly degraded by outlier
r_spearman, _ = stats.spearmanr(x, y) # 1.0 (perfect monotonic order preserved!)
```

### 2. Variance Inflation Factor (VIF) for Multicollinearity
Quantifies how much the variance of an estimated regression coefficient is inflated due to collinearity with other features:
$$\text{VIF}_j = \frac{1}{1 - R_j^2}$$
*where $R_j^2$ is the $R^2$ score when feature $X_j$ is regressed on all other independent features.*
* **$\text{VIF} = 1$:** No correlation with other features.
* **$\text{VIF} > 5$:** Moderate multicollinearity (investigate).
* **$\text{VIF} > 10$:** Severe multicollinearity (feature MUST be dropped or regularized with Ridge).

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

# Assume X_num is a DataFrame of numeric features
X_with_const = sm.add_constant(X_num) if 'sm' in locals() else X_num
vif_data = pd.DataFrame({
    "Feature": X_num.columns,
    "VIF": [variance_inflation_factor(X_num.values, i) for i in range(X_num.shape[1])]
}).sort_values(by="VIF", ascending=False)
```

---

# MODEL CHEAT SHEETS & DECISION GUIDES

## Model Master Cheat Sheet Table

| Model | Primary Task | Core Intuition / Key Mechanism | Feature Scaling Need | Primary Hyperparameters |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** | Regression | OLS line fitting: $\mathbf{w}^T\mathbf{x} + b$ | Not required (Recommended for interpretable weights) | `fit_intercept` |
| **Ridge Regression** | Regression | OLS with $L2$ weight penalty ($\alpha \sum w_i^2$) | **Strongly Recommended** (equal penalty across weights) | `alpha` |
| **Lasso Regression** | Regression | OLS with $L1$ weight penalty ($\alpha \sum \|w_i\|$) | **Strongly Recommended** (fair feature selection) | `alpha` |
| **Logistic Regression**| Classification | Linear log-odds mapped via Sigmoid | **Recommended** (faster gradient descent & fair penalty) | `C`, `penalty`, `max_iter` |
| **KNN** | Both | Majority vote of $k$ closest neighbors | **Strongly Recommended** (distance-based algorithm) | `n_neighbors`, `metric` |
| **Decision Tree** | Both | Recursive binary greedy feature splitting | **NO** (monotonic scale-invariant splits) | `max_depth`, `min_samples_leaf` |
| **Random Forest** | Both | Bagged ensemble of unpruned parallel trees | **NO** (tree-based) | `n_estimators`, `max_depth`, `n_jobs` |
| **Gradient Boosting** | Both | Sequential trees correcting previous errors | **NO** (tree-based) | `n_estimators`, `learning_rate`, `max_depth`|
| **SVM** | Both | Maximum margin separation with kernel trick | **Strongly Recommended** (margin width & RBF kernel scale) | `C`, `kernel`, `gamma` |
| **K-Means** | Clustering | Iterative centroid update minimizing inertia | **Strongly Recommended** (Euclidean centroid distances) | `n_clusters`, `n_init` |
| **MLP / Neural Net** | Both | Stacked linear layers + non-linear activations | **Strongly Recommended** (stable gradients, avoids saturation)| `epochs`, `lr`, `units`, `batch_size` |
| **CNN** | Images/Spatial | Parameter-shared spatial 2D convolutions | **Input normalization recommended** (e.g. 0–1 or standard) | `filters`, `kernel_size`, `pool_size` |
| **LSTM / GRU** | Sequences/Time | Gated cell state overcoming vanishing gradient| **Input normalization recommended** (prevents saturation) | `units`, `timesteps` |

---

## Model Selection Decision Tree ("What Model First?")

```
                          WHAT IS YOUR TARGET TYPE?
                                      │
               ┌──────────────────────┴──────────────────────┐
               ▼                                             ▼
        [ CONTINUOUS ]                                 [ CATEGORICAL ]
               │                                             │
      Is dataset linear?                            Are classes imbalanced?
       /              \                              /                    \
     YES              NO                           YES                    NO
     │                │                            │                      │
[ Linear / Ridge ] [ Random Forest ]         [ Random Forest / XGBoost ]  [ Logistic Reg ]
                   [ LightGBM / XGBoost ]    (Use PR-AUC / F1)             [ Random Forest ]

                          SPECIAL DATA MODALITIES:
  * Images / 2D Spatial Grid  ──────► CNN
  * Audio / Text / Sequential ──────► LSTM / GRU / Transformer
  * Unlabelled Data           ──────► K-Means / DBSCAN
```

---

## Universal Model Templates

### 1. Regression Template
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

X = df.drop("target", axis=1); y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"MAE:  {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R²:   {r2_score(y_test, y_pred):.4f}")
```

### 2. Classification Template
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

X = df.drop("target", axis=1); y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")
```

### 3. Clustering Template
```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

X_scaled = StandardScaler().fit_transform(df)
kmeans = KMeans(n_clusters=3, n_init="auto", random_state=42)
labels = kmeans.fit_predict(X_scaled)
print("Cluster Centroids:\n", kmeans.cluster_centers_)
print("Inertia:", kmeans.inertia_)
```

---

# 🧠 ACTIVE RECALL DRILLS

Test your muscle memory right now. Do not look at the answers until you have written out the code!

### Drill 1: Write a Complete Linear Regression Training + Evaluation Pipeline
<details>
<summary>Show Answer</summary>

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X = df.drop("target", axis=1); y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.2f}, R2: {r2:.4f}")
print("Weights:", model.coef_, "Intercept:", model.intercept_)
```
</details>

---

### Drill 2: Write Logistic Regression with Precision, Recall, and F1
<details>
<summary>Show Answer</summary>

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_train)
X_te_sc = scaler.transform(X_test)

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_tr_sc, y_train)
y_pred = clf.predict(X_te_sc)

print("Precision:", precision_score(y_test, y_pred))
print("Recall:   ", recall_score(y_test, y_pred))
print("F1-Score: ", f1_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
```
</details>

---

### Drill 3: Write Random Forest Classification with Feature Importances
<details>
<summary>Show Answer</summary>

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=200, max_depth=8, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

imp = pd.DataFrame({"Feature": X.columns, "Importance": rf.feature_importances_}).sort_values(by="Importance", ascending=False)
print(imp.head(5))
```
</details>

---

### Drill 4: Write K-Means Clustering and Extract Inertia
<details>
<summary>Show Answer</summary>

```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

X_scaled = StandardScaler().fit_transform(X)
km = KMeans(n_clusters=4, n_init="auto", random_state=42)
cluster_labels = km.fit_predict(X_scaled)
print("Inertia (WCSS):", km.inertia_)
print("Centroids:\n", km.cluster_centers_)
```
</details>

---

### Drill 5: Write a Keras CNN for Binary Image Classification
<details>
<summary>Show Answer</summary>

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

cnn = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid")
])
cnn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
cnn.fit(X_train, y_train, epochs=10, batch_size=32)
```
</details>

---

### Drill 6: Write a Keras LSTM for Sequential Regression
<details>
<summary>Show Answer</summary>

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense

lstm = Sequential([
    LSTM(64, input_shape=(10, 1)), # 10 timesteps, 1 feature
    Dense(1)
])
lstm.compile(optimizer="adam", loss="mse", metrics=["mae"])
lstm.fit(X_train, y_train, epochs=20, batch_size=32)
```
</details>

---

# ⚠️ ML HIGH-FREQUENCY FORGETTING POINTS

| Concept Pair | Core Distinction / What People Forget |
| :--- | :--- |
| **Regression vs. Classification** | Regression outputs continuous real numbers (Price, RUL, Temp); Classification outputs discrete class categories (0/1, Cat/Dog). |
| **Linear Reg vs. Logistic Reg** | Linear Reg models continuous numbers; Logistic Reg is **Classification** (passes linear score through Sigmoid to output probabilities). |
| **`fit` vs `predict` vs `predict_proba`** | `fit` calculates parameters from training data; `predict` outputs final class labels (0 or 1); `predict_proba` outputs probabilities $[P_0, P_1]$. |
| **Parameters vs. Hyperparameters** | Parameters are learned *internally* by the model ($w, b$); Hyperparameters are tuned *externally* by you (`max_depth`, `alpha`, `k`). |
| **$X$ vs $y$** | $X$ is a **2D matrix** of independent features $(N, p)$; $y$ is a **1D vector** of target labels $(N,)$. |
| **Scaling vs. Encoding** | Scaling normalizes numeric magnitude ranges (StandardScaler/MinMax); Encoding converts text categories to numbers (OHE/Ordinal). |
| **Precision vs. Recall** | Precision = Quality of positive predictions ($\frac{\text{TP}}{\text{TP}+\text{FP}}$); Recall = Quantity of actual positives found ($\frac{\text{TP}}{\text{TP}+\text{FN}}$). |
| **MAE vs. RMSE** | MAE gives equal weight to all errors; RMSE squares errors first, heavily punishing large catastrophic mistakes. |
| **Overfitting vs. Underfitting** | Overfitting = High Train Score, Low Test Score (Memorized noise); Underfitting = Low Train Score, Low Test Score (Too simple). |
| **Random Forest vs. Boosting** | Random Forest builds trees **in parallel independently**; Gradient Boosting builds trees **sequentially correcting previous errors**. |
| **CNN vs. RNN** | CNNs extract local spatial patterns from grid images; RNNs model sequential temporal dependencies over time. |
| **RNN vs. LSTM** | Vanilla RNN forgets long sequences ($>10$ steps); LSTM uses additive cell state gating to preserve long-term memory. |
| **Output Layer Activations** | Regression $\to$ `Dense(1)` (linear); Binary Class $\to$ `Dense(1, activation='sigmoid')`; Multi Class $\to$ `Dense(K, activation='softmax')`. |

---
# QUICK REVISION SECTIONS

## 10-Minute Python Revision

```python
# 1. Unpacking & Zipping
a, *b, c = [1, 2, 3, 4, 5]                  # a=1, b=[2, 3, 4], c=5
keys, vals = ["A", "B"], [10, 20]
lookup = dict(zip(keys, vals))               # {'A': 10, 'B': 20}

# 2. Comprehensions
squares = [x**2 for x in range(10) if x % 2 == 0]
label = ["pos" if x > 0 else "neg" for x in nums] # If-Else comprehension

# 3. Dictionaries & Collections
from collections import Counter, defaultdict, deque
cnt = Counter(nums).most_common(2)            # Top 2 most frequent items
groups = defaultdict(list)                    # No KeyError on append
dq = deque(maxlen=5); dq.popleft()            # O(1) FIFO Queue

# 4. Sorting
sorted_by_val = sorted(d.items(), key=lambda x: x[1], reverse=True)

# 5. Heap
import heapq
heapq.nlargest(3, nums)                       # Top 3 elements in O(n log k)
```

---

## 10-Minute NumPy Revision

```python
import numpy as np

# 1. Shapes & Dimensions
arr = np.arange(12).reshape(3, 4)             # 3 rows, 4 cols
arr_col = arr.reshape(-1, 1)                  # Force 2D column vector

# 2. Slicing & Boolean Mask
arr[:, 1]                                     # Column index 1 across all rows
arr[arr > 5] = 0                              # In-place conditional thresholding

# 3. Axis (0 = Rows down, 1 = Cols across)
col_means = np.mean(arr, axis=0)              # 1 mean per column (length 4)
row_sums  = np.sum(arr, axis=1)               # 1 sum per row (length 3)

# 4. Argmax & Where
best_class = np.argmax(probs, axis=1)         # Index of max probability per sample
out = np.where(arr > 0, arr, 0)               # ReLU operator

# 5. Matrix Math
C = A @ B                                     # Matrix multiply (NOT A * B!)
norm = np.linalg.norm(vec)                    # Euclidean vector length
```

---

## 15-Minute Pandas Revision

```python
import pandas as pd
import numpy as np

# 1. Selection & Slicing
df.loc[df["age"] > 30, ["name", "salary"]]     # Label based (included end)
df.iloc[0:5, 0:3]                              # Position based (excluded end)

# 2. Filtering
df[(df["status"] == "Active") & (df["score"] >= 80)] # Wrap with () and use &

# 3. GroupBy Agg & Transform
df.groupby("dept").agg(mean_sal=("salary", "mean"), cnt=("id", "count")).reset_index()
df["dept_mean"] = df.groupby("dept")["salary"].transform("mean") # Preserves row count!

# 4. Merging
pd.merge(df1, df2, on="user_id", how="left")  # SQL Left Outer Join

# 5. Missing & Duplicates
df["salary"] = df["salary"].fillna(df["salary"].median())
df = df.drop_duplicates(subset=["user_id"], keep="last")

# 6. Strings & Dates
df["city"] = df["city"].str.strip().str.title()
df["month"] = pd.to_datetime(df["date"]).dt.month
```

---

## 15-Minute EDA Fast Run

```python
# 1. Shape & Types
print(df.shape)
print(df.dtypes.value_counts())

# 2. Missing Summary
print(df.isna().sum()[df.isna().sum() > 0])

# 3. Numerical Spread
print(df.describe().T[["mean", "std", "min", "50%", "max"]])

# 4. Categorical Breakdown
for col in df.select_dtypes(include="object").columns:
    print(df[col].value_counts(normalize=True).head(3))

# 5. Quick Heatmap
import seaborn as sns; import matplotlib.pyplot as plt
sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap="coolwarm")
plt.show()
```

---

## Pipeline One-Screen Cheat Sheet

```python
# Ingest -> Split -> Pipeline -> Fit -> Predict -> Score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

X = df.drop("target", axis=1); y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ct = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
])

pipe = Pipeline([("prep", ct), ("clf", RandomForestClassifier(random_state=42))])
pipe.fit(X_train, y_train)
acc = pipe.score(X_test, y_test)
print(f"Accuracy: {acc:.4f}")
```

---

# PATTERN LIBRARY (20 COPY-PASTE PATTERNS)

### Pattern 1: Multi-Condition Row Filtering
* **Problem:** Filter DataFrame on multiple logic conditions safely.
```python
# Template:
filtered_df = df[(df["col1"] >= val1) & (df["col2"] == val2) | ~(df["col3"].isin(val_list))]
# Example:
active_seniors = df[(df["age"] >= 65) & (df["status"] == "Active")]
```
* **One-line note:** Always enclose each condition in parentheses `()` and use bitwise operators `&`, `|`, `~`.

---

### Pattern 2: Top-N Records Per Group
* **Problem:** Find the top $K$ highest-value records within each category.
```python
# Template:
top_n = df.sort_values([group_col, sort_col], ascending=[True, False]).groupby(group_col).head(N)
# Example:
top_2_salaries_per_dept = df.sort_values(["dept", "salary"], ascending=[True, False]).groupby("dept").head(2)
```
* **One-line note:** Sort globally first, then `groupby().head(N)`.

---

### Pattern 3: Group-Level Normalization / Broadcast
* **Problem:** Subtract the group mean from each individual row without looping.
```python
# Template:
df["col_diff_from_group"] = df["val_col"] - df.groupby("group_col")["val_col"].transform("mean")
# Example:
df["salary_vs_dept_avg"] = df["salary"] - df.groupby("department")["salary"].transform("mean")
```
* **One-line note:** `.transform()` outputs an array of the same length as the original DataFrame.

---

### Pattern 4: Multiple Named Aggregations
* **Problem:** Calculate descriptive statistics with clean, custom column headers.
```python
# Template:
summary = df.groupby("group_col").agg(
    new_col_name1=("source_col1", "mean"),
    new_col_name2=("source_col2", "count")
).reset_index()
# Example:
dept_summary = df.groupby("dept").agg(avg_pay=("salary", "mean"), headcount=("emp_id", "count")).reset_index()
```
* **One-line note:** Named aggregation avoids messy multi-index columns.

---

### Pattern 5: Finding & Removing Duplicate Keys
* **Problem:** Identify non-unique keys and keep only the latest observation.
```python
# Template:
duplicates = df[df.duplicated(subset=["key_col"], keep=False)]
df_dedup = df.drop_duplicates(subset=["key_col"], keep="last")
# Example:
dup_users = df[df.duplicated(subset=["user_id"], keep=False)]
```
* **One-line note:** `keep=False` marks ALL duplicates so you can inspect collisions.

---

### Pattern 6: Safe Imputation on Train and Test
* **Problem:** Fill NaNs with the training median to prevent data leakage.
```python
# Template:
median_val = X_train["col"].median()
X_train["col"] = X_train["col"].fillna(median_val)
X_test["col"]  = X_test["col"].fillna(median_val)
```
* **One-line note:** Never compute `.median()` on the full dataset before splitting.

---

### Pattern 7: Left Join with Cardinality Validation
* **Problem:** Merge metadata onto a facts table without duplicate row explosion.
```python
# Template:
merged = pd.merge(df_facts, df_dim, on="key_col", how="left", validate="many_to_one")
```
* **One-line note:** `validate="many_to_one"` immediately throws an error if `df_dim` has duplicate keys.

---

### Pattern 8: Multi-Condition Vectorized Column Creation
* **Problem:** Create categorical labels based on tiered numeric thresholds.
```python
# Template:
conditions = [df["col"] < t1, df["col"].between(t1, t2), df["col"] >= t2]
choices = ["Low", "Medium", "High"]
df["tier"] = np.select(conditions, choices, default="Unknown")
```
* **One-line note:** `np.select` is 100x faster than `df.apply(lambda row: ...)`.

---

### Pattern 9: Datetime Component & Weekend Flag Extraction
* **Problem:** Convert a string timestamp into machine learning temporal features.
```python
# Template:
df["dt"] = pd.to_datetime(df["timestamp_col"])
df["hour"] = df["dt"].dt.hour
df["month"] = df["dt"].dt.month
df["is_weekend"] = df["dt"].dt.dayofweek.isin([5, 6]).astype(int)
```
* **One-line note:** Always cast to datetime with `pd.to_datetime` before using `.dt`.

---

### Pattern 10: Percentage of Group Total
* **Problem:** Compute each row's percentage contribution to its department/category.
```python
# Template:
df["pct_of_group"] = (df["val_col"] / df.groupby("group_col")["val_col"].transform("sum")) * 100
# Example:
df["pct_dept_budget"] = (df["salary"] / df.groupby("department")["salary"].transform("sum")) * 100
```
* **One-line note:** Dividing a column by its grouped `.transform("sum")` yields instant row-wise percentages.

---

### Pattern 11: Rolling 7-Day Moving Average with Lag
* **Problem:** Compute smoothed past performance without lookahead bias.
```python
# Template:
df = df.sort_values("date")
df["lag1"] = df["metric"].shift(1) # Exclude today
df["rolling_7d_avg"] = df["lag1"].rolling(window=7, min_periods=1).mean()
```
* **One-line note:** Always `.shift(1)` before `.rolling()` if predicting current-day values to prevent future data leakage.

---

### Pattern 12: Frequency (Count) Encoding
* **Problem:** Encode high-cardinality categorical features into frequency percentages.
```python
# Template:
freq_map = df_train["cat_col"].value_counts(normalize=True).to_dict()
df_train["cat_freq"] = df_train["cat_col"].map(freq_map).fillna(0)
df_test["cat_freq"]  = df_test["cat_col"].map(freq_map).fillna(0)
```
* **One-line note:** Fit the frequency dictionary strictly on the training set.

---

### Pattern 13: Capping Outliers via IQR
* **Problem:** Truncate extreme distribution tails without dropping rows.
```python
# Template:
Q1, Q3 = df["col"].quantile([0.25, 0.75])
IQR = Q3 - Q1
df["col_capped"] = df["col"].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)
```
* **One-line note:** `.clip()` replaces values outside boundaries with the threshold values.

---

### Pattern 14: Correlation Matrix Filter
* **Problem:** Find all pairs of features with high collinearity ($|r| > 0.80$).
```python
# Template:
corr_matrix = df.select_dtypes(include=np.number).corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr = upper_tri.stack().reset_index()
high_corr.columns = ["Feature_1", "Feature_2", "Correlation"]
high_corr = high_corr[high_corr["Correlation"] > 0.80]
```
* **One-line note:** `np.triu(..., k=1)` zeros out the diagonal and lower triangle to avoid duplicate pairs.

---

### Pattern 15: Memory Footprint Optimization
* **Problem:** Downcast float64 and int64 columns to reduce DataFrame RAM usage.
```python
# Template:
for col in df.select_dtypes(include="integer").columns:
    df[col] = pd.to_numeric(df[col], downcast="integer")
for col in df.select_dtypes(include="floating").columns:
    df[col] = pd.to_numeric(df[col], downcast="float")
for col in df.select_dtypes(include="object").columns:
    if df[col].nunique() / len(df) < 0.5:
        df[col] = df[col].astype("category")
```
* **One-line note:** `pd.to_numeric(..., downcast=...)` shrinks 64-bit types to 32, 16, or 8-bit.

---

### Pattern 16: Safe Log-Transform on Zero/Positive Features
* **Problem:** Transform skewed revenue/income containing exact 0 values.
```python
# Template:
df["log_feature"] = np.log1p(df["raw_feature"]) # log(1 + x)
# Inversion back to raw units:
df["raw_recovered"] = np.expm1(df["log_feature"]) # exp(x) - 1
```
* **One-line note:** Always use `np.log1p` and `np.expm1` to handle 0 without `inf` errors.

---

### Pattern 17: Equal-Frequency Quantile Binning
* **Problem:** Split continuous data into 5 bins with an equal number of samples.
```python
# Template:
df["bin_q5"] = pd.qcut(df["numeric_col"], q=5, labels=["Very Low", "Low", "Med", "High", "Very High"])
```
* **One-line note:** Use `pd.qcut` for equal sample count, `pd.cut` for equal bin width.

---

### Pattern 18: One-Hot Encoding with Unknown Handling
* **Problem:** One-hot encode categoricals without crashing when the test set has unseen labels.
```python
# Template:
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_train_ohe = ohe.fit_transform(X_train[["cat_col"]])
X_test_ohe  = ohe.transform(X_test[["cat_col"]])
```
* **One-line note:** `handle_unknown="ignore"` assigns all-zeros when encountering unseen categories in test data.

---

### Pattern 19: Time-Series Month-End Resampling
* **Problem:** Aggregate transaction rows into monthly revenue totals.
```python
# Template:
monthly_rev = df.set_index("date_col").resample("ME")["amount"].sum().reset_index()
```
* **One-line note:** `resample("ME")` requires a DatetimeIndex and groups by Month End.

---

### Pattern 20: Cross-Validated Baseline Score
* **Problem:** Measure model generalization stability across 5 folds in 3 lines.
```python
# Template:
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model_pipeline, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
print(f"CV R2: {scores.mean():.4f} (+/- {scores.std():.4f})")
```
* **One-line note:** Evaluates the entire preprocessing + model pipeline across 5 independent splits.

---

# "WHAT DO I USE?" DECISION TABLE

| I want to... | Use this | Example Snippet |
| :--- | :--- | :--- |
| Count category frequencies | `df['col'].value_counts()` | `df['status'].value_counts(normalize=True)` |
| Filter rows on conditions | Boolean indexing | `df[(df['a'] > 10) & (df['b'] == 'X')]` |
| Filter rows by multiple values | `.isin()` | `df[df['city'].isin(['NYC', 'LON'])]` |
| Aggregate by group | `.groupby().agg()` | `df.groupby('dept').agg(avg=('pay','mean'))` |
| Attach group metrics to rows | `.groupby().transform()` | `df['dept_mean'] = df.groupby('dept')['pay'].transform('mean')` |
| Filter out small groups | `.groupby().filter()` | `df.groupby('dept').filter(lambda g: len(g) > 5)` |
| Merge tables by key | `pd.merge()` | `pd.merge(df1, df2, on='id', how='left')` |
| Stack tables vertically | `pd.concat(axis=0)` | `pd.concat([df_2023, df_2024], axis=0)` |
| Reshape Long to Wide | `df.pivot_table()` | `df.pivot_table(index='d', columns='g', values='val')` |
| Reshape Wide to Long | `pd.melt()` | `pd.melt(df, id_vars=['id'], value_vars=['q1', 'q2'])` |
| Handle missing values | `.fillna()` / `.dropna()` | `df['age'].fillna(df['age'].median())` |
| Forward-fill time-series NaNs | `.ffill()` | `df['price'].ffill()` |
| Rank items within category | `.groupby().rank()` | `df.groupby('dept')['salary'].rank(ascending=False)` |
| Get Top N per group | `.sort_values().groupby().head(N)` | `df.sort_values('sal').groupby('dept').head(3)` |
| Shift data for lags / growth | `.shift()` | `df['prev_day'] = df['rev'].shift(1)` |
| Rolling moving average | `.rolling()` | `df['rev'].rolling(7, min_periods=1).mean()` |
| Cumulative total over time | `.cumsum()` | `df['cum_rev'] = df['rev'].cumsum()` |
| Discretize into quantile bins | `pd.qcut()` | `pd.qcut(df['income'], q=4, labels=['Q1','Q2','Q3','Q4'])` |
| Clean string columns | `.str.strip().str.lower()` | `df['name'] = df['name'].str.strip().str.lower()` |
| Check string substring | `.str.contains()` | `df[df['email'].str.contains('@gmail', na=False)]` |
| Extract year / month | `.dt.year` / `.dt.month` | `df['date'].dt.month` |
| Calculate days between dates | `(d2 - d1).dt.days` | `(df['end'] - df['start']).dt.days` |
| 1D vector to 2D column | `.reshape(-1, 1)` | `X_single = X_arr.reshape(-1, 1)` |
| Conditional assignment (NumPy) | `np.where()` | `np.where(arr > 0, 1, 0)` |
| Multi-condition assignment | `np.select()` | `np.select(conditions, choices, default=0)` |
| Find index of max value | `np.argmax()` | `np.argmax(probabilities, axis=1)` |
| Matrix multiplication | `@` or `np.matmul()` | `W @ X + b` |
| Top-K elements (Python) | `heapq.nlargest()` | `heapq.nlargest(5, scores)` |
| Element frequency counting | `collections.Counter` | `Counter(words).most_common(3)` |
| Safe default dictionary | `collections.defaultdict` | `d = defaultdict(list); d['k'].append(v)` |

---

# 🧠 ANTI-FORGETTING WORKFLOW SUMMARY

```
                    DAILY 10-MINUTE RETENTION HABIT
                    
   ┌─────────────────────────────────────────────────────────────┐
   │ 1. [2 Min] SCAN PYTHON CORE (Comprehensions, Zip, Counter)  │
   │ 2. [2 Min] SCAN NUMPY (Axis rule, Slicing, Boolean Mask)    │
   │ 3. [3 Min] SCAN PANDAS (.loc/.iloc, GroupBy, Transform)     │
   │ 4. [2 Min] PICK 1 PATTERN FROM PATTERN LIBRARY              │
   │ 5. [1 Min] CLOSE THIS FILE & CODE 1 PATTERN FROM MEMORY     │
   └─────────────────────────────────────────────────────────────┘
```

> **The Golden Muscle-Memory Test:**
> Close this file right now. Open a blank python file or IPython prompt, and type out from memory:
> 1. `df.groupby("dept")["salary"].transform("mean")`
> 2. `df[(df["age"] > 25) & (df["salary"] <= 50000)]`
> 3. `np.where(arr > 0, arr, 0)`
> 
> *If you can type those 3 lines without looking, you are ready for any Data Science OA or Technical Interview!*
