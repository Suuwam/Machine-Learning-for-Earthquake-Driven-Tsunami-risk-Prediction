# Tsunami Trigger Prediction with Random Forests

This is a machine learning project that predicts whether a given earthquake is likely to trigger a **tsunami**.  
Using a **Random Forest classifier** trained on global earthquake data (1995–2023), the model learns from seismic and spatio‑temporal features to classify events as **tsunami** or **non‑tsunami**, providing a data‑driven decision-support component for early warning systems.

---

##  Project Overview

- **Goal:** Predict tsunami occurrence from earthquake characteristics.
- **Data Source:** Historical global earthquake records (1995–2023) merged from two CSV datasets.
- **Target Variable:** `tsunami` (binary: 0 = no tsunami, 1 = tsunami).
- **Tech Stack:**  
  - Python, Jupyter Notebook  
  - pandas, NumPy  
  - scikit-learn (RandomForestClassifier, GridSearchCV, train_test_split, metrics)  
  - Matplotlib, Seaborn  

---

##  Data Processing & Feature Engineering

1. **Data Loading & Merging**
   - Load:
     - `earthquake_1995-2023.csv`
     - `earthquake_data.csv`
   - Vertically concatenate into a single DataFrame: `complete_quake_data` (1,782 records, 19+ columns).

2. **Datetime Parsing & Temporal Features**
   - Convert `date_time` to `datetime`:
     ```python
     pd.to_datetime(..., format='%d-%m-%Y %H:%M', errors='coerce', utc=True)
     ```
   - Extract rich temporal features:
     - `year`, `month_num`, `month_name`
     - `day`, `hour`
     - `day_of_week_num`, `day_of_week_name`
     - `quarter`
     - `is_weekend`
     - `hour_category` (Night, Morning, Afternoon, Evening)

3. **Missing Value Handling**
   - **Numeric columns:** fill with column **mean**.
   - **Categorical columns:** fill with `'Unknown'`.
   - Final check confirms **no missing values** remain.

4. **Feature Cleanup**
   - Drop non‑model columns:
     - `date_time`
     - `title`
   - Final dataset: **1,782 rows × 29 columns** (mixed numeric & categorical).

---

##  Feature Preparation

- **Target:**  
  ```python
  target_col = 'tsunami'
  y = complete_quake_data[target_col]
  ```

- **Features:**  
  ```python
  X = complete_quake_data.drop(columns=[target_col])
  ```

- **Numeric features (examples):**
  - `magnitude`, `cdi`, `mmi`, `sig`, `nst`, `dmin`, `gap`, `depth`
  - `latitude`, `longitude`
  - `year`, `hour`, `month_num`, `day`, `day_of_week_num`, `quarter`

- **Categorical features (examples):**
  - `alert`, `net`, `magType`, `location`
  - `continent`, `country`
  - `month`, `day_of_week`, `month_name`, `day_of_week_name`

- **Encoding:**
  - One‑hot encoding with `pd.get_dummies`:
    ```python
    X_encoded = pd.get_dummies(
        X,
        columns=categorical_features,
        drop_first=True,
        dtype='int8'
    )
    ```
  - Final feature matrix: **(1782, 638)**

---

##  Train–Test Split

- Split with stratification to preserve class balance:
  ```python
  X_train, X_test, y_train, y_test = train_test_split(
      X_encoded,
      y,
      test_size=0.2,
      random_state=42,
      stratify=y
  )
  ```
- Training set: **1,425 samples (80%)**  
- Test set: **357 samples (20%)**  
- Verified:
  - Similar class distribution in train/test
  - Feature columns consistent across splits

---

##  Model & Hyperparameter Tuning

- **Base model:**  
  ```python
  rf_model = RandomForestClassifier(
      random_state=42,
      class_weight='balanced',
      n_jobs=-1
  )
  ```

- **GridSearchCV parameter grid (example from notebook):**
  ```python
  param_grid = {
      'n_estimators': [100, 200, 300],
      'max_depth': [None, 10, 20],
      'min_samples_split': [2, 5],
      'max_features': ['sqrt', 0.8]
  }
  ```

- **Tuning Strategy:**
  - Use `GridSearchCV` to find the best combination of hyperparameters.
  - Cross‑validation on training data.
  - Fit best estimator and evaluate on the held‑out test set.

---

##  Model Evaluation

The notebook evaluates performance using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **Confusion matrix**

These metrics provide a balanced view of performance on both tsunami and non‑tsunami classes, which is important for an imbalanced classification problem.

---

##  Repository Contents

- `mlpc_INDIVIDUAL.ipynb` – Main Jupyter notebook containing:
  - Data loading & merging  
  - Preprocessing and feature engineering  
  - Train–test split  
  - Random Forest model training & tuning  
  - Evaluation & summaries  

(Additional plots and outputs are generated inside the notebook.)

---

##  What This Project Demonstrates

- Practical **end‑to‑end ML pipeline**:
  - Data cleaning, feature engineering, encoding, splitting, modeling, evaluation.
- Use of **tree‑based models** (Random Forest) for a **binary classification** problem.
- Handling of:
  - Mixed numerical and categorical features
  - Time‑based feature extraction
  - Class imbalance via `class_weight='balanced'`
- Application of **GridSearchCV** to systematically tune model hyperparameters.

---

##  Disclaimer

This project is an **academic / experimental prototype** and **not** an operational early warning system.  
Predictions should **not** be used for real‑world disaster management decisions without validation by domain experts and integration into official warning frameworks.

---
