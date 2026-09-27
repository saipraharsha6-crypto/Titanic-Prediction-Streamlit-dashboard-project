import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from xgboost import XGBClassifier

import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    Conv1D,
    MaxPooling1D,
    Flatten
)


print("\nLoading Titanic dataset...")

df = pd.read_csv("Titanic-Dataset.csv")

print("\nDataset loaded successfully!")
print("Shape:", df.shape)


print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())


features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
]

X = df[features]
y = df["Survived"]

print("\nFeatures:")
print(features)

print("\nTarget:")
print("Survived")


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data :", X_test.shape)

numeric_features = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare"
]

categorical_features = [
    "Sex",
    "Embarked"
]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss"
    )
}

ml_results = []

trained_ml_models = {}

print("\n")
print("=" * 60)
print("TRAINING MACHINE LEARNING MODELS")
print("=" * 60)

for name, model in models.items():

    print("\nTraining:", name)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    y_prob = pipeline.predict_proba(X_test)[:, 1]

    results = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    }

    ml_results.append(results)

    trained_ml_models[name] = pipeline

    print("Accuracy :", round(results["Accuracy"], 4))
    print("Precision:", round(results["Precision"], 4))
    print("Recall   :", round(results["Recall"], 4))
    print("F1 Score :", round(results["F1"], 4))
    print("ROC-AUC  :", round(results["ROC-AUC"], 4))

print("\n")
print("=" * 60)
print("PREPARING DATA FOR DEEP LEARNING")
print("=" * 60)

X_train_dl = preprocessor.fit_transform(X_train)
X_test_dl = preprocessor.transform(X_test)

if hasattr(X_train_dl, "toarray"):
    X_train_dl = X_train_dl.toarray()

if hasattr(X_test_dl, "toarray"):
    X_test_dl = X_test_dl.toarray()

print("\nDeep Learning training shape:", X_train_dl.shape)
print("Deep Learning testing shape :", X_test_dl.shape)

print("\n")
print("=" * 60)
print("TRAINING ANN")
print("=" * 60)

ann = Sequential([
    Input(shape=(X_train_dl.shape[1],)),
    Dense(64, activation="relu"),
    Dropout(0.30),
    Dense(32, activation="relu"),
    Dropout(0.20),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])

ann.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

ann.summary()

# Train ANN
ann.fit(
    X_train_dl,
    y_train,
    validation_split=0.20,
    epochs=50,
    batch_size=32,
    verbose=1
)

print("\nEvaluating ANN...")

ann_prob = ann.predict(X_test_dl).ravel()

ann_pred = (ann_prob >= 0.5).astype(int)

ann_results = {
    "Model": "ANN",
    "Accuracy": accuracy_score(y_test, ann_pred),
    "Precision": precision_score(y_test, ann_pred),
    "Recall": recall_score(y_test, ann_pred),
    "F1": f1_score(y_test, ann_pred),
    "ROC-AUC": roc_auc_score(y_test, ann_prob)
}

print("\nANN Results:")
print("Accuracy :", round(ann_results["Accuracy"], 4))
print("Precision:", round(ann_results["Precision"], 4))
print("Recall   :", round(ann_results["Recall"], 4))
print("F1 Score :", round(ann_results["F1"], 4))
print("ROC-AUC  :", round(ann_results["ROC-AUC"], 4))

print("\n")
print("=" * 60)
print("PREPARING DATA FOR 1D CNN")
print("=" * 60)

X_train_cnn = X_train_dl.reshape(
    X_train_dl.shape[0],
    X_train_dl.shape[1],
    1
)

X_test_cnn = X_test_dl.reshape(
    X_test_dl.shape[0],
    X_test_dl.shape[1],
    1
)

print("CNN training shape:", X_train_cnn.shape)
print("CNN testing shape :", X_test_cnn.shape)

print("\n")
print("=" * 60)
print("TRAINING 1D CNN")
print("=" * 60)

cnn = Sequential([
    Input(shape=(X_train_cnn.shape[1], 1)),
    Conv1D(32, 3, activation="relu"),
    MaxPooling1D(2),
    Conv1D(64, 3, activation="relu"),
    Flatten(),
    Dense(32, activation="relu"),
    Dropout(0.30),
    Dense(1, activation="sigmoid")
])

cnn.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

cnn.summary()

cnn.fit(
    X_train_cnn,
    y_train,
    validation_split=0.20,
    epochs=50,
    batch_size=32,
    verbose=1
)

print("\nEvaluating 1D CNN...")

cnn_prob = cnn.predict(X_test_cnn).ravel()

cnn_pred = (cnn_prob >= 0.5).astype(int)

cnn_results = {
    "Model": "1D CNN",
    "Accuracy": accuracy_score(y_test, cnn_pred),
    "Precision": precision_score(y_test, cnn_pred),
    "Recall": recall_score(y_test, cnn_pred),
    "F1": f1_score(y_test, cnn_pred),
    "ROC-AUC": roc_auc_score(y_test, cnn_prob)
}

print("\nCNN Results:")
print("Accuracy :", round(cnn_results["Accuracy"], 4))
print("Precision:", round(cnn_results["Precision"], 4))
print("Recall   :", round(cnn_results["Recall"], 4))
print("F1 Score :", round(cnn_results["F1"], 4))
print("ROC-AUC  :", round(cnn_results["ROC-AUC"], 4))

print("\n")
print("=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

all_results = ml_results + [
    ann_results,
    cnn_results
]

results_df = pd.DataFrame(all_results)

results_df = results_df.sort_values(
    by="F1",
    ascending=False
).reset_index(drop=True)

print("\n")
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]["Model"]

print("\n")
print("=" * 60)
print("BEST MODEL BASED ON F1 SCORE")
print("=" * 60)

print("\nBest model:", best_model_name)
print("F1 Score:", round(results_df.iloc[0]["F1"], 4))

if best_model_name in trained_ml_models:

    os.makedirs("models", exist_ok=True)

    best_pipeline = trained_ml_models[best_model_name]

    joblib.dump(
        best_pipeline,
        "models/titanic_best_model.pkl"
    )

    print("\nBest ML model saved successfully!")

else:

    print("\nThe best model is a Deep Learning model.")
    print("The 5 ML pipelines are available in memory.")
    print("For Streamlit deployment, select a saved ML pipeline.")

os.makedirs("models", exist_ok=True)

results_df.to_csv(
    "models/model_comparison.csv",
    index=False
)

print("\nModel comparison saved to:")
print("models/model_comparison.csv")

print("\n")
print("=" * 60)
print("PROJECT TRAINING COMPLETED")
print("=" * 60)

print("\nCompleted:")
print("1. Data loading")
print("2. Data preprocessing")
print("3. Train-test split")
print("4. Logistic Regression")
print("5. Decision Tree")
print("6. Random Forest")
print("7. Gradient Boosting")
print("8. XGBoost")
print("9. ANN")
print("10. 1D CNN")
print("11. Model evaluation")
print("12. Model comparison")
print("13. Best model selection")
print("14. Model saving")

print("\nDone!")