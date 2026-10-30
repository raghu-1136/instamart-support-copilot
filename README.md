# Overview

A comprehensive Customer Intelligence & Support Copilot built for Swiggy Instamart quick-commerce operations across six metro cities. As order volume scales, quick-commerce operators encounter three systemic challenges: raw transactional exports suffer from duplicates, missing fields, and unparseable values; retention teams lack visibility into which customers are lapsing; and customer support queues are overwhelmed by thousands of free-text tickets mixing urgent fraud, late deliveries, app crashes, and billing disputes.

All data in this project is strictly synthetic and fictional, generated deterministically to model realistic quick-commerce scenarios. The brand name is used solely to frame the business context; no real Instamart customer records, policies, or pricing are used.

This repository provides an end-to-end, deterministic solution across four connected subsystems:
1. **Data Foundation (Part A)**: Cleans dirty transactional exports with explicit documented rules, loads a relational SQLite database (`data/instamart.db`), verifies data integrity against reference checkpoints, and answers business questions via SQL and exploratory data analysis.
2. **Customer Segmentation & Lapse Prediction (Part B)**: Constructs RFM profiles for 1,175 active customers, identifies 4 actionable segments via K-Means and 2D PCA, trains leak-free supervised classifiers (Logistic Regression and Random Forest) to predict customer churn, and exports a prioritized retention list.
3. **Support Ticket Triage with Guardrails (Part C)**: Implements regex-based PII redaction, token-length input guardrails, zero-shot and few-shot classification with exponential backoff retries, output schema validation, token budgeting, and customer-segment-aware queue routing.
4. **Policy RAG Assistant & Autonomous Refund Agent (Part D)**: Chunks company policy documents with sliding-window overlap, implements TF-IDF vector retrieval to ground a policy question-answering assistant, and runs an autonomous ReAct refund agent equipped with deterministic business rules, safety stop boundaries, and crash-resilient state recovery.

The entire copilot operates **100% offline, deterministically, and without paid API dependencies** using `mock_llm.py`. All monetary figures are reported strictly in **Indian Rupees (INR)**.

---

# Reproduce

The complete pipeline can be reproduced from a clean clone using the following numbered commands:

1. Create and activate an isolated Python virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate
   ```
   *(On Windows: `env\Scripts\activate`)*

2. Upgrade pip and install exact project dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. Generate deterministic synthetic raw data and policy text files:
   ```bash
   python generate_data.py
   ```

4. Audit raw datasets for anomalies and schema integrity:
   ```bash
   python -m src.part_a.audit
   ```

5. Clean raw data and populate the analytical SQLite database:
   ```bash
   python -m src.part_a.clean
   ```

6. Validate Part A cleaning checkpoints against exact references:
   ```bash
   python -m src.part_a.validate
   ```

7. Execute business SQL queries against the database:
   ```bash
   python -m src.part_a.run_sql
   ```

8. Execute exploratory data analysis and distribution checks:
   ```bash
   python -m src.part_a.eda
   ```

9. Generate RFM features, evaluate K-Means clusters, and profile segments:
   ```bash
   python -m src.part_b.segments
   ```

10. Train, tune, and evaluate lapse prediction models and export retention targets:
    ```bash
    python -m src.part_b.lapse_model
    ```

11. Run support ticket triage with input guardrails, PII redaction, and routing:
    ```bash
    python -m src.part_c.triage
    ```

12. Evaluate policy document chunking, TF-IDF retrieval, and grounding:
    ```bash
    python -m src.part_d.rag
    ```

13. Run autonomous refund agent scenarios, safety limit fallbacks, and crash recovery:
    ```bash
    python -m src.part_d.run_scenarios
    ```

---

# Part A
## Data Foundation: Clean, Store, Query

### 1. Raw Data Audit
The raw exports contained substantial noise, unparseable entries, inconsistent formatting, and duplicate records. `src/part_a/audit.py` inspected shapes, data types, and missingness:

| Dataset | Shape | Column | Data Type | Missing Count | Missing Pct (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **orders.csv** | (13084, 9) | `order_id` | str | 0 | 0.00% |
| | | `customer_id` | str | 0 | 0.00% |
| | | `product_id` | str | 0 | 0.00% |
| | | `quantity` | int64 | 0 | 0.00% |
| | | `order_ts` | str | 0 | 0.00% |
| | | `status` | str | 0 | 0.00% |
| | | `discount_pct` | int64 | 0 | 0.00% |
| | | `order_amount` | str | 53 | 0.41% |
| | | `delivery_minutes`| float64 | 2,541 | 19.42% |
| **customers.csv**| (1200, 5) | `customer_id` | str | 0 | 0.00% |
| | | `city` | str | 0 | 0.00% |
| | | `signup_date` | str | 0 | 0.00% |
| | | `age` | float64 | 61 | 5.08% |
| | | `gender` | str | 30 | 2.50% |
| **products.csv** | (40, 4) | `product_id` | str | 0 | 0.00% |
| | | `product_name` | str | 0 | 0.00% |
| | | `category` | str | 0 | 0.00% |
| | | `price_inr` | int64 | 0 | 0.00% |

*(Note: String columns are represented as `str` in pandas >= 3.0, corresponding to `object` in earlier versions.)*

#### Specific Audit Checkpoints
- **Raw Orders**: 13,084 total rows; 119 exact duplicates; 656 older/pending records; 12,309 distinct `order_id` values.
- **Order Anomalies**: 120 non-numeric `order_amount` entries (53 `"N/A"` plus 67 `"error"`); 106 negative `delivery_minutes` durations; 2,541 blank `delivery_minutes`.
- **Customer Anomalies**: 61 blank ages; 6 impossible ages outside range 10–100 (-4, 0, 150, 200, 280, 999); 24 raw city spellings (mixed upper/lower case and whitespace padding); 10 distinct non-blank raw gender spellings; 30 blank genders.

---

### 2. Cleaning Rules & Transformations
Cleaning was executed deterministically via `src/part_a/clean.py` according to explicit business logic:

#### Customers Table Transformations
1. **City Standardization**: Stripped leading/trailing whitespace and converted to title case. Reduced 24 raw variations down to exactly 6 metro cities:
   - Pune: 227 | Delhi: 210 | Bengaluru: 197 | Chennai: 191 | Hyderabad: 191 | Mumbai: 184.
2. **Gender Mapping**: Mapped 10 raw spellings (`male, Male, M, m, MALE, female, Female, F, f, FEMALE`) into canonical values, and mapped blanks to `unknown`:
   - Female: 610 | Male: 560 | Unknown: 30.
3. **Age Imputation**: Replaced 6 out-of-range ages (< 10 or > 100) with `NaN`. All 67 missing ages were imputed using the customer's city median (`groupby('city')['age'].transform('median')`). Post-cleaning: 0 missing ages, all values strictly bounded between 18 and 60.
4. **Dates**: Converted `signup_date` to standard ISO `YYYY-MM-DD`.

#### Orders Table Transformations
1. **Timestamp Conversion**: Parsed `order_ts` into datetime objects.
2. **Exact Duplicate Removal**: Dropped 119 exact duplicate rows (row count: 13,084 → 12,965).
3. **Deduplication by Status/Timestamp**: Retained only the latest record per `order_id` by sorting by `order_ts` and keeping the last entry. This eliminated 656 older/pending records (row count: 12,965 → 12,309).
4. **Order Amount Repair**: Converted `order_amount` to numeric, coercing unparseable entries to missing values. Of the 120 non-numeric entries identified in the raw audit, 8 were on rows removed during deduplication. The remaining 112 missing amounts were filled from catalogue product price:
   `order_amount = round(price_inr * quantity * (1 - discount_pct / 100), 2)`
   Post-cleaning: 0 missing order amounts.
5. **Delivery Duration Imputation**: Set negative `delivery_minutes` to missing (105 negative delivery times set to NaN, as 1 of the 106 was on a row removed during deduplication). For completed orders, filled the 774 missing `delivery_minutes` with the median delivery time of that order's product category among completed orders. For the 1,199 cancelled orders, delivery durations were left missing as cancelled orders were never delivered.
6. **Reference Checkpoints after Cleaning**:
   - Total rows: 12,309 (one per `order_id`).
   - Completed orders: 11,110 | Cancelled orders: 1,199 | Pending orders: 0.
   - Missing `order_amount`: 0.
   - Missing `delivery_minutes`: 0 among completed orders; exactly 1,199 among cancelled orders.
   - Total revenue of completed orders: **INR 2,998,668.80**.

---

### 3. Database Storage & Checkpoint Validation
Cleaned datasets were saved to `data/clean/` (`customers.csv`, `products.csv`, `orders.csv`) and loaded into SQLite at `data/instamart.db`. 

The test script `src/part_a/validate.py` evaluated all 14 assertions:
1. `orders` table count in `data/instamart.db` equals 12,309.
2. Cleaned customer city value counts match expected distribution across all 6 cities.
3. Cleaned customer gender counts match expected values (female: 610, male: 560, unknown: 30).
4. Missing age count equals 0.
5. All customer ages fall between 18 and 60.
6. Total orders row count equals 12,309.
7. Completed orders count equals 11,110.
8. Cancelled orders count equals 1,199.
9. All `order_id`s are strictly unique.
10. Pending status count equals 0.
11. Missing `order_amount` count equals 0.
12. Completed orders missing `delivery_minutes` equals 0.
13. Cancelled orders missing `delivery_minutes` equals 1,199.
14. Total completed revenue equals INR 2,998,668.80 (tolerance < 0.01).

Execution output:
```text
ALL CHECKPOINTS PASSED
```

---

### 4. SQL Analytical Queries
Seven business queries in `sql/queries.sql` were executed against `data/instamart.db` via `src/part_a/run_sql.py`:

| Query | Focus & Concept | Key Results & Checkpoints |
| :--- | :--- | :--- |
| **Q1** | Filter high-value completed orders (> INR 500) and aggregate spend > INR 8,000 (`WHERE` then `HAVING`). | **30 customers** qualified. |
| **Q2** | Three-table join (`orders`, `customers`, `products`) aggregating revenue by city and category. | **30 rows**; total revenue: **INR 2,998,668.80**. Top cell: **Bengaluru / Electronics at INR 233,909.45**. |
| **Q3** | Scalar subquery identifying catalogue products priced strictly above average catalogue price (catalogue prices sum to 11,573 across 40 items, average is **INR 289.33**). | **11 products** priced above catalogue average. |
| **Q4** | `LEFT JOIN` identifying registered customers with zero completed orders. | **25 customers** with no completed orders. |
| **Q5** | Dual Common Table Expressions (CTEs) computing customer completed spend and filtering those above average. | **550 customers** spend above the mean customer spend of **INR 2,552.06**. |
| **Q6** | Window functions (`RANK()`, `DENSE_RANK()`, `ROW_NUMBER()`) partitioned by city ordered by total spend. | Top per city: **Bengaluru C0082** (INR 12,856.20), **Chennai C0930** (INR 10,151.40), **Delhi C0291** (INR 13,742.60), **Hyderabad C0743** (INR 12,730.05), **Mumbai C0693** (INR 14,766.55), **Pune C0426** (INR 10,377.10). Exactly **8 rows** have `RANK() != ROW_NUMBER()`. |
| **Q7** | Daily completed revenue and cumulative running total using `SUM(revenue) OVER (ORDER BY day)`. | **183 daily rows** (2025-04-01 to 2025-09-30); final cumulative revenue reaches **INR 2,998,668.80**. |

> **Reason for Q6 Rank Discrepancy**: Ties in customer `total_spend` receive identical ranking values with `RANK()` (causing subsequent ranks to skip), whereas `ROW_NUMBER()` assigns distinct sequential integers arbitrarily regardless of ties.

---

### 5. Pandas Cross-Check & Exploratory Data Analysis

#### City-Level Aggregation Cross-Check
Rebuilding Q2 via `pd.pivot_table(..., margins=True)` confirmed an identical grand total of **INR 2,998,668.80**. Grouping completed orders by city yielded:

| City | Total Spend (INR) | Mean Order Amount (INR) | Completed Order Count |
| :--- | :--- | :--- | :--- |
| **Bengaluru** | 523,643.40 | 276.47 | 1,894 |
| **Chennai** | 463,629.15 | 272.72 | 1,700 |
| **Delhi** | 502,829.80 | 243.62 | 2,064 |
| **Hyderabad** | 528,589.90 | 284.95 | 1,855 |
| **Mumbai** | 479,058.50 | 297.00 | 1,613 |
| **Pune** | 500,918.05 | 252.48 | 1,984 |

#### Distribution & Outlier Statistics
- **Order Amount Distribution**: Mean = INR 269.91 | Median = INR 135.00 | Skewness = 4.61 (heavily right-skewed).
- **Quartiles & IQR**: Q1 = INR 64.80 | Q3 = INR 297.00 | IQR = INR 232.20.
- **Tukey 1.5×IQR Fences**: Lower Fence = INR -283.50 | Upper Fence = INR 645.30.
- **Outliers**: Exactly 897 upper outliers above INR 645.30; 0 outliers below the lower fence. Upper outliers concentrate in high-ticket Electronics (635 orders) and bulk Dairy items (166 orders of Ghee 500ml).

#### Correlation Matrix (Completed Orders)
| Feature | `order_amount` | `discount_pct` | `quantity` | `delivery_minutes` |
| :--- | :--- | :--- | :--- | :--- |
| `order_amount` | 1.000 | -0.095 | +0.087 | -0.003 |
| `discount_pct` | -0.095 | 1.000 | -0.000 | -0.010 |
| `quantity` | +0.087 | -0.000 | 1.000 | -0.013 |
| `delivery_minutes` | -0.003 | -0.010 | -0.013 | 1.000 |

#### Four Exploratory Questions
1. **Data Scope**: The dataset models fast-moving consumer goods (grocery, dairy, snacks, personal care) and small electronics orders across 6 Indian metro cities over a 6-month window (April 1 to September 30, 2025).
2. **Distributions**: Completed order amounts are heavily right-skewed (mean INR 269.91 vs. median INR 135.00, skewness 4.61), dominated by low-value grocery/dairy baskets alongside occasional high-value electronic purchases.
3. **Anomalies & Outliers**: Exactly 897 transactions exceed the upper fence of INR 645.30, primarily representing premium electronics (e.g., Bluetooth speakers, power banks) and high-value pantry staples (Ghee 500ml).
4. **Relationships & Correlation**: The correlation between `order_amount` and `discount_pct` has the largest absolute magnitude at −0.095 (followed by `order_amount` and `quantity` at +0.087). Crucially, a correlation of this minor magnitude does not establish causation, as promotional discounts and order values are influenced by product pricing and purchasing contexts.

---

# Part B
## Customer Segments and Lapse Prediction

### 1. RFM Feature Engineering
As of snapshot **2025-10-01 00:00:00**, RFM metrics were computed across all 1,175 customers with completed orders:
- **Recency** (days since last completed order): Median = 19 days | Minimum = 0 days | Maximum = 180 days.
- **Frequency** (count of distinct completed orders): Mean = 9.46 orders | Maximum = 33 orders.
- **Monetary** (sum of completed spend): Mean = INR 2,552.06 | Maximum = INR 14,766.55.

---

### 2. K-Means Clustering & K Selection
The three RFM features were standardised using `StandardScaler`. `KMeans(n_init=10, random_state=42)` was fitted for K from 2 to 8:

| Number of Clusters (K) | Inertia | Silhouette Score |
| :--- | :--- | :--- |
| K = 2 | 1820.761 | 0.465 |
| K = 3 | 1268.235 | 0.445 |
| **K = 4 (Selected)** | **743.071** | **0.526** |
| K = 5 | 617.367 | 0.480 |
| K = 6 | 503.171 | 0.482 |
| K = 7 | 408.460 | 0.416 |
| K = 8 | 366.503 | 0.412 |

> **Why Inertia Alone Cannot Pick K**: Inertia strictly measures the within-cluster sum of squared distances to the nearest centroid and decreases monotonically toward zero as K approaches the sample size. Because inertia does not penalize model complexity or measure separation between clusters, selecting K requires identifying the point of diminishing returns (the elbow inflection) balanced against silhouette separation and business interpretability.
> 
> **Selection of K=4**: K=4 was selected because it delivers the global maximum silhouette score (0.526), coincides with a clear elbow reduction, and isolates operationally distinct cohorts (dormant, occasional, high-frequency, and high-spend customers).

---

### 3. Customer Segment Profiles & PCA Projection
Cluster profiles for the selected K=4 model:

| Segment ID | Segment Name | Size | Mean Recency (days) | Mean Frequency | Mean Monetary (INR) | `is_premium` | Recommended Action |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **Premium Big Spenders** | 138 | 24.12 | 6.77 | **INR 7,129.70** | **True** | Dedicated VIP support desk and preview access to high-value electronics. |
| **2** | **Loyal Regulars** | 419 | 8.77 | 19.40 | INR 3,410.02 | False | Grocery subscription passes, bulk-saving bundles, and loyalty rewards. |
| **0** | **Occasional Shoppers** | 436 | 28.86 | 3.77 | INR 1,057.62 | False | Category discount coupons and re-order reminders for pantry essentials. |
| **3** | **Lapsing / Dormant** | 182 | 109.04 | 2.20 | INR 685.99 | False | Win-back push notifications featuring deep flat-rate promotional discounts. |

*(Note: Cluster IDs are arbitrary algorithm labels with no natural ordering. The segment with the highest mean monetary value is designated `is_premium = True`.)*

- **PCA Projection**: Two principal components explain **63.4%** and **21.8%** of the variance respectively, capturing **85.2%** of total variance.
- Output exported to `outputs/customer_segments.csv` (1,175 rows).

---

### 4. Lapse Problem Formulation & Leak-Free Cohort
- **Cutoff Date**: `2025-09-01`.
- **Target Definition**: A customer has lapsed (`lapsed = 1`) if they placed **zero** completed orders between 2025-09-01 and 2025-09-30, and `0` otherwise.
- **Eligible Cohort**: 1,156 customers placed at least one completed order prior to the cutoff.
- **Class Distribution**: 443 lapsed customers (**38.32%**) and 713 active customers (**61.68%**).
- **Features (Pre-cutoff Only)**: `recency`, `frequency`, `monetary`, `avg_discount`, `cancel_rate`, `avg_delivery`, `fav_category`, `city`, `age`, `gender`, `tenure`.
- **Split**: Stratified train/test split (80/20, `random_state=42`) produced 924 training rows and 232 test rows.
- **Naive Baseline**: Predicting the majority class ("not lapsed") on the test set yields an accuracy of **61.6%** and a lapsed-class F1 score of **0.000**.

> **Leakage Prevention Statement**: No encoder, scaler, or resampler saw test rows. Oversampling touched the training set only (the test set remained untouched and un-resampled). No data from 2025-09-01 onwards entered the features, and segment labels from Part B clustering were not used as features.

---

### 5. Preprocessing Pipeline & Feature Scaling
- Built with `ColumnTransformer`: numeric features standardized via `StandardScaler()`; nominal categoricals (`city`, `gender`, `fav_category`) encoded using `OneHotEncoder(drop='first', handle_unknown='ignore')`.
- **Learned Training Mean vs. Full Cohort**: Scaler learned training mean for `frequency` = **8.116** vs. full-cohort mean = **8.029**. They differ because the scaler was fitted exclusively on the 924 training rows to avoid test-set data snooping.
- Zero missing values were verified across all feature matrices.

---

### 6. Model Tuning & Test-Set Evaluation
Both models were tuned using 5-fold cross-validation (`scoring='f1'`) over the training partition:
- **Logistic Regression**: `GridSearchCV` over C in {0.01, 0.1, 1, 10, 100}. Best parameter: C = 0.1 (CV F1 = 0.639).
- **Random Forest**: `GridSearchCV` over `max_depth` in {3, 5, 8} and `min_samples_leaf` in {5, 10, 20} (45 fits). Best parameters: `max_depth = 3`, `min_samples_leaf = 5` (CV F1 = 0.675).

#### Untouched Test-Set Performance
| Model | Test Accuracy | Precision (Lapsed) | Recall (Lapsed) | F1 Score (Lapsed) | Confusion Matrix `[[TN, FP], [FN, TP]]` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (Majority)**| 61.6% | 0.000 | 0.000 | 0.000 | `[[143, 0], [89, 0]]` |
| **Logistic Regression**| **76.3%** | 0.655 | **0.809** | **0.724** | `[[105, 38], [17, 72]]` |
| **Random Forest** | 76.7% | **0.680** | 0.742 | 0.710 | `[[112, 31], [23, 66]]` |

*(Note: Both tuned test F1 scores (0.724 and 0.710) exceed the brief's reference values of ~0.62 and ~0.65, comfortably surpassing the minimum requirement threshold of F1 >= 0.55.)*

> **Model Selection Justification**: In customer retention, the business cost of a false negative (missing a customer who actually lapses and stops ordering permanently) is far higher than the cost of a false positive (sending an unnecessary retention offer or discount to an active customer). Logistic Regression was selected as the final model because it achieves the higher recall on the lapsed class (0.809 vs. 0.742) and a higher overall F1 score (0.724 vs. 0.710), successfully identifying 72 of 89 test-set churners. Its linear structure also offers direct, transparent feature interpretability.

---

### 7. Class-Balance Experiment
Oversampling the minority class (`lapsed = 1`) on the training partition brought the training set to an even 570 / 570 balance. Evaluating the retrained Logistic Regression model against the unchanged test set produced:

| Model Split Configuration | Precision (Lapsed) | Recall (Lapsed) | F1 Score (Lapsed) |
| :--- | :--- | :--- | :--- |
| **Unbalanced Logistic Regression** | **0.655** | 0.809 | **0.724** |
| **Balanced Logistic Regression** | 0.587 | **0.910** | 0.714 |

> **Retention Impact**: Recall moved most dramatically, increasing from 0.809 to 0.910 (+10.1 percentage points), while precision fell from 0.655 to 0.587. For the retention team, this trade-off means capturing 91.0% of all at-risk accounts, providing nearly complete churn interception at the cost of a modest increase in discount outreach to loyal customers.

---

### 8. Interpretability, Persistence & High-Risk Retention List
- **Top 5 Random Forest Features**:
  1. `frequency` (0.352) — Order frequency accounts for the largest share of tree splits.
  2. `monetary` (0.289) — Total spend is the second most heavily used feature.
  3. `recency` (0.115) — Recency contributes moderate tree split importance.
  4. `cancel_rate` (0.053) — Carries little predictive signal in this data.
  5. `avg_discount` (0.041) — Carries little predictive signal in this data.
  **(Together, frequency and monetary dominate feature importances, accounting for over 64% of total feature importance.)**
- **Logistic Regression Coefficients & Business Readings**:
  - `frequency`: **−1.281** — Order frequency is the sole strong driver; each standard-deviation increase in order frequency significantly reduces churn likelihood.
  - `recency`: **−0.028** — Near zero; has no meaningful effect on churn probability once frequency is in the model.
  - `cancel_rate`: **−0.101** — Near zero; has no meaningful effect on churn probability once frequency is in the model.
- **Model Persistence**: Saved to `outputs/lapse_model.joblib`. Reloading with `joblib.load` reproduced original test predictions with 100% exact match.
- **High-Risk Retention List (`outputs/retention_list.csv`)**: Evaluated among the 29 test-set customers identified as Premium, ranked by predicted lapse probability:

| Customer ID | Predicted Lapse Probability | Actually Lapsed |
| :--- | :--- | :--- |
| **C0577** | 0.5750 | 0 |
| **C0102** | 0.5704 | 1 |
| **C0697** | 0.5475 | 0 |
| **C0942** | 0.5099 | 0 |
| **C0831** | 0.4959 | 0 |
| **C0509** | 0.4765 | 0 |
| **C0162** | 0.4764 | 0 |
| **C0324** | 0.4763 | 0 |
| **C0977** | 0.4449 | 1 |
| **C0299** | 0.4447 | 0 |

> **Caveat on Retention List**: Note that the Premium flag was derived from the post-hoc whole-window RFM segmentation (April–September 2025), which overlaps with the September 2025 lapse observation window; thus, this retention list serves as an illustrative operational demonstration rather than a leakage-free evaluation. Exactly 2 of the top 10 ranked customers truly lapsed.

---

# Part C
## Support Ticket Triage with Guardrails

### 1. Input Guardrails
Pre-model checks evaluated all 164 raw support tickets in `data/raw/tickets.csv`:
- **Empty Tickets**: 4 tickets (`T0157, T0158, T0159, T0160`) contained empty/whitespace text -> status `skipped_empty` (0 tokens used; zero LLM calls).
- **Excessively Long Tickets**: 4 tickets (`T0161, T0162, T0163, T0164`) exceeded `MAX_INPUT_TOKENS = 400` (measured at exactly 734 estimated tokens each) -> status `flagged_long` (0 tokens used; zero LLM calls).
- Exactly 156 tickets met criteria and proceeded to the LLM pipeline.

---

### 2. PII Redaction
Three regex patterns sanitized customer personal information before prompt assembly:
- Email: `[\w.+-]+@[\w-]+\.[\w.]+` -> `[EMAIL]`
- Phone Number: `\b[6-9]\d{9}\b` -> `[PHONE]`
- Order ID: `ORD-\d{6}` -> `[ORDER_ID]`

| Redaction Entity | Substitutions Across 164 Tickets |
| :--- | :--- |
| **Email Addresses** | 55 |
| **Phone Numbers** | 91 |
| **Order IDs** | 132 |
| **Total Substitutions** | **278** |

Code assertions confirmed that **0 PII patterns remained** across all sanitized texts. Prompts sent to `mock_llm.chat` contained strictly the redacted ticket body.

---

### 3. Prompt Engineering
- **Zero-Shot Prompt** (36 tokens): Instructs model with keyword `CLASSIFY`, restricts categories to `billing, delivery, technical, fraud`, requires pure JSON response `{"category": "...", "reason": "..."}`, and contains no `EXAMPLES:`.
- **Few-Shot Prompt** (128 tokens): Identical baseline instructions plus an `EXAMPLES:` block containing four labeled demonstration tickets spanning all valid classes, explicitly highlighting fraud detection.

---

### 4. Resilience & Exponential Backoff Retries
Transient network and server issues were simulated via `TransientLLMError`. A retry wrapper executed with exponential backoff and jitter:
`delay = BASE_DELAY * (2 ** n) * uniform(0.9, 1.15) (with BASE_DELAY = 0.05s)`
Across 156 eligible tickets, exactly **19 TransientLLMErrors** occurred. All 19 errors were successfully recovered on retry 0 with zero ticket terminations or permanent failures.

---

### 5. Output Validation Guardrails
Model outputs were strictly verified without assuming valid JSON formatting:
- **`classified_json`**: Valid JSON dictionary matching one of four allowable categories (**135 tickets**).
- **`classified_keyword`**: Non-JSON responses containing chatty assistant phrasing (e.g., `"The customer appears to have a billing-related problem."`), parsed via `re.findall(r"[a-z]+", text.lower())` to extract single valid categories (**12 tickets**).
- **`flagged_invalid_output`**: Invalid JSON or categories outside allowable values (e.g., `"shipping"`) routed to human agents (**9 tickets**).

> **Verification and Escalation Protocol**: In this triage pipeline, `classified_json` and `classified_keyword` tickets were rule-validated against the schema, while empty, long, invalid, and budget-exhausted tickets were routed directly to human review.

---

### 6. Zero-Shot vs. Few-Shot Comparison

| Pipeline Configuration | Classified Tickets | Correct Classifications | Accuracy (%) | Total Tokens Used | Tokens / Classified Ticket |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Zero-Shot** | 147 | 126 | 85.7% | 10,024 | 68.2 |
| **Few-Shot** | 147 | 147 | **100.0%** | 24,376 | 165.8 |

#### Zero-Shot Error Breakdown
| True Category | Predicted: Billing | Predicted: Delivery | Predicted: Fraud | Predicted: Technical |
| :--- | :--- | :--- | :--- | :--- |
| **Billing** | **38** | 0 | 0 | 0 |
| **Delivery** | 0 | **36** | 0 | 0 |
| **Fraud** | **11** | 0 | **23** | 0 |
| **Technical** | **10** | 0 | 0 | **29** |

> **Cost vs. Accuracy Analysis**: The few-shot prompt consumes an additional 97.6 tokens per ticket (165.8 vs. 68.2 tokens) and 24,376 total tokens versus 10,024 for zero-shot, representing a 2.43x token cost increase. In return, few-shot classification completely eliminates all 21 zero-shot misclassifications (11 fraud tickets and 10 technical tickets misrouted to billing), lifting accuracy from 85.7% to 100.0%. In quick-commerce customer support, misrouting unauthorized transactions to billing queues causes critical security response delays and compliance exposure. Therefore, paying for the few-shot demonstration context is justified for high-stakes customer workflows, while zero-shot remains suitable only for low-cost exploratory tasks.

---

### 7. Token Budgeting & Customer-Segment Routing
- **Token Budget Demo (`outputs/triage_budget_demo.csv`)**: Evaluated zero-shot with `TOKEN_BUDGET = 6000`. Once the token threshold was reached, the pipeline halted model calls and marked remaining eligible tickets as `budget_exhausted` (**62 tickets exhausted**).
- **Final Segment-Aware Routing (`outputs/triage_results.csv`)**: Generated using the 100% accurate few-shot pipeline (`TOKEN_BUDGET = 100000`) joined against Part B's customer segments:
  - Guardrail flags (`skipped_empty`, `flagged_long`, `flagged_invalid_output`, `budget_exhausted`) -> `human_review` (NORMAL priority).
  - Category `fraud` -> `security_team` (HIGH priority).
  - Customer `is_premium == True` -> `premium_desk` (HIGH priority).
  - Standard queue -> `support_queue` (NORMAL priority).

#### Final Triage Routing Summary (164 Total Tickets)
| Assigned Route | Assigned Priority | Ticket Count | Description |
| :--- | :--- | :--- | :--- |
| **`support_queue`** | NORMAL | 102 | Standard non-urgent delivery, billing, and technical tickets. |
| **`security_team`** | HIGH | 34 | Fraud and unauthorized-transaction tickets. |
| **`human_review`** | NORMAL | 17 | Edge cases: 4 empty, 4 long, 9 invalid LLM responses. |
| **`premium_desk`** | HIGH | 11 | Tickets from Premium customers. |

---

# Part D
## Policy RAG Assistant and Refund Agent

### 1. Document Chunking
The 5 policy documents in `data/policies/` were segmented using sliding-window chunking (chunk size = 40 words, overlap = 10 words, step = 30 words):
- Total chunks created: **15 chunks** (`refund_policy.txt`: 4 chunks | `delivery_policy.txt`: 3 chunks | `escalation_policy.txt`: 3 chunks | `payment_policy.txt`: 3 chunks | `privacy_policy.txt`: 2 chunks).

---

### 2. TF-IDF Retrieval Architecture
A `TfidfVectorizer` was fitted across all 15 chunks, and query-to-chunk relevance was ranked via `cosine_similarity`.
- For query `"refund policy late delivery"`, all top 3 retrieved passages originate from `refund_policy.txt` (top similarity score = 0.272).

> **TF and IDF Explanation**: Term Frequency (TF) measures how frequently a term appears within an individual chunk, while Inverse Document Frequency (IDF) scales down ubiquitous terms across all documents to highlight distinctive, high-information keywords.
> 
> **Limitation of TF-IDF**: Because TF-IDF relies purely on lexical word matching, it fails to capture semantic equivalence when queries use synonyms or colloquial phrasing not literally present in the document text (e.g., matching "reimbursement" with "refund").

---

### 3. Grounded Policy Assistant QA
The `answer(question)` pipeline retrieves top-3 chunks, injects source tags `[<filename>]`, and prompts `mock_llm.chat` with system role `POLICY ASSISTANT` to restrict responses to provided context.

---

### 4. Retrieval & Grounding Evaluation
Evaluated across all 14 questions in `data/raw/policy_eval.csv` (12 answerable, 2 out-of-domain):

| Chunking Configuration | Answerable Doc Hits (Top-3) | Answerable Fact Hits | Unanswerable Fallback Hits | Failed Questions |
| :--- | :--- | :--- | :--- | :--- |
| **Overlap = 10 (40 words, step 30)** | **12 / 12** | **12 / 12** | **2 / 2** | **None (100% Pass)** |
| **Overlap = 0 (40 words, step 40)** | **12 / 12** | 10 / 12 | **2 / 2** | 2 fact failures (see below) |

> **Analysis of Zero-Overlap Failures**: While top-3 doc hits remained 12/12, two answerable questions failed fact verification under zero overlap because facts were split across boundaries:
> 1. *"What delivery time makes a completed order eligible for a full refund?"*: The condition "more than 45 minutes" was severed across consecutive chunk boundaries.
> 2. *"How soon is money returned when a payment fails but is deducted?"*: The phrase "48 hours" was split across chunks, and the severed fragment was retrieved alongside an unrelated passage.

---

### 5. Tool Specifications & Execution Map
The agent operates over 5 tools defined in `src/part_d/tools.py`:
- `get_order(order_id)`: Queries `data/instamart.db` for status, amount, and delivery latency.
- `search_policy(query)`: Executes TF-IDF retrieval over policy chunks.
- `calculate_refund(order_amount, delivery_minutes)`: Enforces business logic (delivery > 45 mins on completed orders yields 100% refund, <= 45 mins yields 0).
- `issue_refund(order_id, refund_inr)`: Appends approved refund (<= INR 1000) to `outputs/refund_ledger.jsonl`.
- `escalate_to_human(order_id, refund_inr)`: Logs high-value refund (> INR 1000) to `outputs/refund_ledger.jsonl` for manual review.

---

### 6. ReAct Agent Scenario Results
Executed across all 7 scenarios in `data/raw/agent_scenarios.csv`:

| Scenario ID | Status | Refund Amount (INR) | Model Calls | Tokens Used | Outcome Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **S1** | **REFUND** | **INR 128.25** | 5 | 917 | Late delivery refund auto-issued below INR 1000 limit. |
| **S2** | **REFUND** | **INR 135.00** | 5 | 917 | Late delivery refund auto-issued below INR 1000 limit. |
| **S3** | **ESCALATE** | **INR 1999.00** | 5 | 904 | Late delivery refund above INR 1000 escalated to human lead. |
| **S4** | **DENIED** | **INR 0.00** | 4 | 621 | Order delivered in <= 45 minutes; refund policy denied. |
| **S5** | **DENIED** | **INR 0.00** | 3 | 349 | Cancelled order was never charged; refund denied. |
| **S6** | **NOT_FOUND** | **INR 0.00** | 2 | 80 | Non-existent order ORD-999999; halted at lookup. |
| **S7** | **FALLBACK_MAX_ITERATIONS** | **INR 0.00** | 6 | 3,024 | Looping synthetic prompt; safely halted at 6 iterations. |

All scenario outputs and step histories were recorded in `outputs/agent_runs.jsonl`. All financial operations were appended to `outputs/refund_ledger.jsonl`.

---

### 7. Stop-Safely Demonstrations
The agent enforces three independent runtime guardrails:
1. **Max Iterations Limit**: Tested on scenario S7 (`max_iterations = 6`). Agent halts after 6 model calls without infinite looping.
   - Status: **`FALLBACK_MAX_ITERATIONS`**
2. **Token Limit Fallback**: Tested on scenario S1 with `token_limit = 200`. The agent made 3 model calls and executed `get_order` and `search_policy`. Upon making the third call requesting `calculate_refund`, accumulated tokens exceeded 200, so `calculate_refund` was not executed and the agent halted immediately.
   - Status: **`FALLBACK_TOKEN_LIMIT`**
3. **Timeout Fallback**: Tested on scenario S1 with `time_limit_seconds = 0`. Agent halts immediately before making any LLM call.
   - Status: **`FALLBACK_TIMEOUT`**

---

### 8. Crash-and-Resume Demonstration
To guarantee zero duplicate disbursements upon runtime failures, state is persisted to `state/<run_id>.json` after every tool execution. A simulated crash was injected during scenario S1 immediately following tool 2 (`search_policy`):
- **Tools Executed in Crash Phase**: `get_order`, `search_policy` -> `SimulatedCrash` raised.
- **Resumption**: `resume_agent("S1_demo")` loaded state from `state/S1_demo.json` and continued the loop.
- **Tools Executed in Resume Phase**: `calculate_refund`, `issue_refund`.
- **Ordered Tool Execution Across Both Phases**:
  1. `get_order`
  2. `search_policy`
  *(Crash & Recovery)*
  3. `calculate_refund`
  4. `issue_refund`
- Exactly **4 tools executed**, **0 duplicated executions**, with final status **`REFUND: INR 128.25 issued.`**
- The demonstration appends a fourth line to `outputs/refund_ledger.jsonl`. Directory `state/` holds 9 checkpoint files: S1–S7 runs (7 files), the token-limit demo (1 file, which executed two tools before stopping), and the crash-and-resume S1 demo (1 file).

---

### 9. ReAct versus Plan-and-Execute Architectural Trade-Offs

Comparing operational paradigms across our seven refund scenarios highlights the clear trade-offs between step-by-step reactivity and upfront planning. Our ReAct agent dynamically requested tools based on incremental environment feedback: S1 completed in 5 model calls (917 tokens), S5 resolved in 3 calls (349 tokens), and S6 finished in just 2 calls (80 tokens). In contrast, a plan-and-execute design would ask the model for a complete sequential plan upfront, requiring only about 1–2 model calls where ReAct needed 5 calls for S1. While upfront planning saves model calls and tokens on linear happy paths, it struggles with conditional branching and unexpected observations: on S6 (`NOT_FOUND`) or S5 (`status: cancelled`), a fixed plan would continue executing useless steps, such as computing an unnecessary refund or erroneously logging a zero-amount refund to the ledger, unless an explicit replanning cycle is triggered. For an automated customer refund workflow, ReAct is the superior architecture. Because every tool execution depends strictly on prior validation, ReAct prevents unintended ledger writes, halts early on missing or cancelled orders, and guarantees that refunds above INR 1000 escalate immediately to human leads.

---

# Limitations and Next Steps

1. **Post-Hoc Premium Labeling Caveat**: The `is_premium` flag was generated from an RFM segmentation spanning the full dataset (April 1 to September 30, 2025), which overlaps with the September lapse evaluation period. As stated in Part B, the retention list (`outputs/retention_list.csv`) is an operational demonstration rather than a clean out-of-time evaluation. In production, segmentation models should be fitted strictly prior to the observation window.
2. **Lexical Keyword Retrieval Limitations**: The current policy RAG assistant uses TF-IDF and cosine similarity. While effective on exact terms, it lacks semantic awareness for synonyms (e.g., "reimbursement" vs. "refund") and complex multilingual inquiries. Next steps include implementing dense vector embeddings backed by a lightweight vector index (e.g., FAISS or Chroma).
3. **Offline Mock LLM to Production Models**: All LLM interactions were simulated using the offline `mock_llm.py`. Future iterations can integrate hosted foundation models with API keys managed securely via `.env` files.
4. **Concurrency & Distributed State Storage**: The current state checkpointing writes individual JSON files to local disk (`state/`). In high-concurrency microservice architectures, this should transition to Redis or PostgreSQL with distributed locking and ACID transaction guarantees to prevent race conditions during refund issuance.
5. **Dynamic Token & Budget Management**: The token budget in Part C uses a hard-coded cumulative threshold. Upgrading to rolling per-ticket complexity estimators and token-streaming timeouts will optimize resource allocation without dropping eligible customer tickets during peak traffic spikes.
6. **Python 3.14 Environment Considerations**: This project was developed and evaluated on a system running Python 3.14.8. When replicating in environments with older or strictly pinned C-extension packages, standard Python 3.11 or 3.12 virtual environments provide reliable pre-built wheel compatibility for dependencies such as `scikit-learn` and `pandas`.
