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
   - [Numerical Transformations](#1-numerical-transformations)
   - [Categorical Encoding](#2-categorical-encoding)
   - [Datetime Feature Extraction](#3-datetime-feature-extraction)
   - [Binning & Discretization](#4-binning--discretization)
   - [Outlier Detection & Capping](#5-outlier-detection--capping)
   - [Scikit-Learn Preprocessing Essentials](#6-scikit-learn-preprocessing-essentials)
6. [PART 6 — COMPLETE DATA SCIENCE PIPELINE TEMPLATE](#part-6--complete-data-science-pipeline-template)
7. [QUICK REVISION SECTIONS (TIMED DRILLS)](#quick-revision-sections)
   - [10-Minute Python Revision](#10-minute-python-revision)
   - [10-Minute NumPy Revision](#10-minute-numpy-revision)
   - [15-Minute Pandas Revision](#15-minute-pandas-revision)
   - [15-Minute EDA Fast Run](#15-minute-eda-fast-run)
   - [Pipeline One-Screen Cheat Sheet](#pipeline-one-screen-cheat-sheet)
8. [PATTERN LIBRARY (20 COPY-PASTE DATA SCIENCE PATTERNS)](#pattern-library-20-copy-paste-patterns)
9. ["WHAT DO I USE?" DECISION TABLE](#what-do-i-use-decision-table)

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

# PART 6 — COMPLETE DATA SCIENCE PIPELINE TEMPLATE

Copy-paste this turnkey template at the beginning of any project:

```python
# ==============================================================================
# END-TO-END DATA SCIENCE PIPELINE TEMPLATE
# ==============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# --- 1. LOAD DATA ---
def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return df

# --- 2. CLEAN & BASIC FEATURE ENGINEERING ---
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Clean string columns
    if "customer_name" in df.columns:
        df["customer_name"] = df["customer_name"].str.strip().str.title()
        
    # Date extraction
    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"])
        df["order_month"] = df["order_date"].dt.month
        df["order_dayofweek"] = df["order_date"].dt.dayofweek
        df["is_weekend"] = df["order_dayofweek"].isin([5, 6]).astype(int)
        df.drop(columns=["order_date"], inplace=True)
        
    # Ratios
    if "revenue" in df.columns and "units" in df.columns:
        df["price_per_unit"] = df["revenue"] / (df["units"] + 1e-5)
        
    return df

# --- 3. MAIN PIPELINE EXECUTION ---
def run_pipeline():
    # A. Ingestion
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
    
    # G. Train Model
    full_model.fit(X_train, y_train)
    
    # H. Evaluate
    y_pred = full_model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    
    print("\n" + "=" * 45)
    print("MODEL EVALUATION RESULTS (TEST SET)")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")
    print("=" * 45)
    
    return full_model

if __name__ == "__main__":
    model = run_pipeline()
```

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
