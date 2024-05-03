import pandas as pd
import numpy as np

file_location = "train.csv"
df = pd.read_csv(file_location)
include = ["Age", "Sex", "Embarked", "Survived"]
df_ = df[include]

categoricals = []
for col, col_type in df_.dtypes.items():
    if col_type == "object":
        categoricals.append(col)
    else:
        df_[col].fillna(0, inplace=True)

df_ohe = pd.get_dummies(df_, columns=categoricals, dummy_na=True)

from sklearn.linear_model import LogisticRegression

dependent_variable = "Survived"
x = df_ohe[df_ohe.columns.difference([dependent_variable])]
y = df_ohe[dependent_variable]
lr = LogisticRegression(
    C=1.0,
    class_weight=None,
    dual=False,
    fit_intercept=True,
    intercept_scaling=1,
    max_iter=100,
    multi_class="ovr",
    n_jobs=1,
    penalty="l2",
    random_state=None,
    solver="liblinear",
    tol=0.0001,
    verbose=0,
    warm_start=False,
)
lr.fit(x, y)

import joblib

joblib.dump(lr, "model.pkl")
print("Model dumped!")

lr = joblib.load("model.pkl")

model_columns = list(x.columns)
joblib.dump(model_columns, "model_columns.pkl")
print("Models columns dumped!")
