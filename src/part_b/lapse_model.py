import pandas as pd
import os
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone

orders = pd.read_csv("data/clean/orders.csv")
customers = pd.read_csv("data/clean/customers.csv")
products = pd.read_csv("data/clean/products.csv")

cutoff = pd.Timestamp("2025-09-01")
end = pd.Timestamp("2025-10-01")

completed = orders[orders["status"] == "completed"].copy()
completed["order_ts"] = pd.to_datetime(completed["order_ts"])

# Strictly filter orders before 2025-09-01 cutoff to prevent future-data leakage into churn features
before = completed[completed["order_ts"] < cutoff].copy()
september = completed[(completed["order_ts"] >= cutoff) & (completed["order_ts"] < end)]

cohort = before[["customer_id"]].drop_duplicates().reset_index(drop=True)
print("Cohort size:", len(cohort))
assert len(cohort) == 1156

cohort["lapsed"] = (~cohort["customer_id"].isin(september["customer_id"])).astype(int)

print("Lapsed:", cohort["lapsed"].sum(), f"({cohort['lapsed'].mean() * 100:.2f}%)")
assert cohort["lapsed"].sum() == 443

orders["order_ts"] = pd.to_datetime(orders["order_ts"])

print(cohort.shape)
print(cohort.isna().sum())

order_feats = before.groupby("customer_id").agg(
    last_ts=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum"),
    avg_discount=("discount_pct", "mean"),
    avg_delivery=("delivery_minutes", "mean"),
).reset_index()
order_feats["recency"] = (cutoff - order_feats["last_ts"]).dt.days
order_feats = order_feats.drop(columns="last_ts")

cohort = cohort.merge(order_feats, on="customer_id", how="left")
print(cohort.shape)

all_before = orders[orders["order_ts"] < cutoff]
cancel = (all_before.assign(c=all_before["status"] == "cancelled")
          .groupby("customer_id")["c"].mean().rename("cancel_rate").reset_index())
cohort = cohort.merge(cancel, on="customer_id", how="left")

fav = (before.merge(products[["product_id", "category"]], on="product_id")
       .groupby("customer_id")["category"]
       .agg(lambda s: s.mode().iloc[0]).rename("fav_category").reset_index())
cohort = cohort.merge(fav, on="customer_id", how="left")

cust = customers[["customer_id", "city", "age", "gender", "signup_date"]].copy()
cust["signup_date"] = pd.to_datetime(cust["signup_date"])
cust["tenure"] = (cutoff - cust["signup_date"]).dt.days
cohort = cohort.merge(cust.drop(columns="signup_date"), on="customer_id", how="left")

print(cohort["recency"].min(), cohort["recency"].max())
print(cohort["cancel_rate"].between(0, 1).all())
print(cohort["tenure"].min())
print(cohort["fav_category"].value_counts())

y = cohort["lapsed"]
X = cohort.drop(columns=["lapsed", "customer_id"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("Train/test sizes:", len(X_train), len(X_test))
assert (len(X_train), len(X_test)) == (924, 232)

pred0 = [0] * len(y_test)
print("Baseline accuracy / F1:", accuracy_score(y_test, pred0), f1_score(y_test, pred0, zero_division=0))

num_cols = ["recency", "frequency", "monetary", "avg_discount",
            "cancel_rate", "avg_delivery", "age", "tenure"]
cat_cols = ["city", "gender", "fav_category"]

preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
])

# Fit scalers and OneHotEncoders exclusively on training data (X_train) to avoid test-set data snooping
preprocess.fit(X_train)
scaler = preprocess.named_transformers_["num"]
freq_idx = num_cols.index("frequency")
print("scaler learned frequency mean:", scaler.mean_[freq_idx])
print("full-cohort frequency mean:", cohort["frequency"].mean())

assert X.isna().sum().sum() == 0
print(y_train.value_counts())

lr_pipe = Pipeline([("prep", preprocess),
                    ("model", LogisticRegression(max_iter=1000))])
lr_grid = GridSearchCV(lr_pipe,
                       param_grid={"model__C": [0.01, 0.1, 1, 10, 100]},
                       cv=5, scoring="f1")
lr_grid.fit(X_train, y_train)
print("Logistic regression best:", lr_grid.best_params_, round(lr_grid.best_score_, 4))

rf_pipe = Pipeline([("prep", preprocess),
                    ("model", RandomForestClassifier(n_estimators=100, random_state=42))])
rf_grid = GridSearchCV(rf_pipe,
                       param_grid={"model__max_depth": [3, 5, 8],
                                   "model__min_samples_leaf": [5, 10, 20]},
                       cv=5, scoring="f1")
rf_grid.fit(X_train, y_train)
print("Random forest best:", rf_grid.best_params_, round(rf_grid.best_score_, 4))

for name, g in [("LogReg", lr_grid), ("RandomForest", rf_grid)]:
    pred = g.best_estimator_.predict(X_test)
    print(name)
    print(confusion_matrix(y_test, pred))
    print("precision", round(precision_score(y_test, pred, zero_division=0), 3),
          "recall", round(recall_score(y_test, pred, zero_division=0), 3),
          "F1", round(f1_score(y_test, pred, zero_division=0), 3))
final_f1 = f1_score(y_test, lr_grid.best_estimator_.predict(X_test), zero_division=0)
assert final_f1 >= 0.55

# Oversample lapsed minority class only within the training partition (570/570 balanced); test partition remains untouched
lapsed_rows = X_train[y_train == 1]
extra = lapsed_rows.sample(n=570 - 354, replace=True, random_state=42)

X_bal = pd.concat([X_train, extra])
y_bal = pd.concat([y_train, y_train.loc[extra.index]])
print(y_bal.value_counts())          # 570 / 570


lr_bal = clone(lr_grid.best_estimator_).fit(X_bal, y_bal)
pred_bal = lr_bal.predict(X_test)

print("Balanced logistic regression:")
print(confusion_matrix(y_test, pred_bal))
print("precision", round(precision_score(y_test, pred_bal, zero_division=0), 3),
      "recall", round(recall_score(y_test, pred_bal, zero_division=0), 3),
      "F1", round(f1_score(y_test, pred_bal, zero_division=0), 3))

print("Top 5 forest features:")
names = rf_grid.best_estimator_.named_steps["prep"].get_feature_names_out()
imp = rf_grid.best_estimator_.named_steps["model"].feature_importances_
print(pd.Series(imp, index=names).sort_values(ascending=False).head(5))

print("Logistic coefficients:")
lr_names = lr_grid.best_estimator_.named_steps["prep"].get_feature_names_out()
coef = lr_grid.best_estimator_.named_steps["model"].coef_[0]
print(pd.Series(coef, index=lr_names)[["num__recency", "num__frequency", "num__cancel_rate"]])

final_model = lr_grid.best_estimator_

os.makedirs("outputs", exist_ok=True)
joblib.dump(final_model, "outputs/lapse_model.joblib")
loaded = joblib.load("outputs/lapse_model.joblib")
assert (loaded.predict(X_test) == final_model.predict(X_test)).all()
print("Reload check passed: predictions identical")

proba = final_model.predict_proba(X_test)[:, 1]
res = pd.DataFrame({
    "customer_id": cohort.loc[X_test.index, "customer_id"].values,
# Disclaimer: Premium status is derived from whole-window RFM and is post-hoc; retention list is illustrative
    "lapse_probability": proba,
    "actually_lapsed": y_test.values,
})
seg = pd.read_csv("outputs/customer_segments.csv")[["customer_id", "is_premium"]]
res = res.merge(seg, on="customer_id", how="inner")
print("Test customers after merge:", len(res), "| Premium:", int(res["is_premium"].sum()))
res = res[res["is_premium"]].sort_values("lapse_probability", ascending=False).head(10)
res[["customer_id", "lapse_probability", "actually_lapsed"]].to_csv(
    "outputs/retention_list.csv", index=False)
print(res[["customer_id", "lapse_probability", "actually_lapsed"]])
print("truly lapsed:", res["actually_lapsed"].sum(), "of", len(res))
