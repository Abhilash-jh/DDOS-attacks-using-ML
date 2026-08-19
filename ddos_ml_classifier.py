"""
=============================================================================
ML-Based Classification and Prediction Technique for DDoS Attacks
=============================================================================
Based on: "A Machine Learning-Based Classification and Prediction Technique
           for DDoS Attacks"

Features:
  - Data preprocessing & feature engineering
  - Multiple ML classifiers (Random Forest, XGBoost, SVM, Neural Network)
  - Ensemble voting classifier
  - Real-time prediction pipeline
  - Performance evaluation & visualization
  - Model persistence

Dataset: Compatible with CIC-DDoS2019, NSL-KDD, or any network traffic CSV
         with numeric features and a label column.
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import json
import os

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, auc
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, AdaBoostClassifier
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────
CONFIG = {
    "test_size": 0.20,
    "val_size": 0.10,
    "random_state": 42,
    "n_features": 20,          # top-K features to select
    "cv_folds": 5,
    "model_dir": "saved_models",
    "results_dir": "results",
    "label_column": "Label",   # change to your dataset's label column
}

os.makedirs(CONFIG["model_dir"], exist_ok=True)
os.makedirs(CONFIG["results_dir"], exist_ok=True)


# ─────────────────────────────────────────────
# 2. SYNTHETIC DATASET GENERATOR
#    (Replace with real dataset loader in production)
# ─────────────────────────────────────────────
def generate_synthetic_dataset(n_samples: int = 20_000, n_features: int = 40,
                                random_state: int = 42) -> pd.DataFrame:
    """
    Generates a synthetic network traffic dataset that mimics
    CIC-DDoS2019 feature distributions for demonstration purposes.
    """
    rng = np.random.RandomState(random_state)

    ATTACK_TYPES = {
        "BENIGN":       (0.35, 0),
        "UDP-Flood":    (0.15, 1),
        "SYN-Flood":    (0.12, 2),
        "HTTP-Flood":   (0.10, 3),
        "ICMP-Flood":   (0.08, 4),
        "DNS-Amplify":  (0.08, 5),
        "NTP-Amplify":  (0.07, 6),
        "SlowLoris":    (0.05, 7),
    }

    feature_names = [
        "Flow_Duration", "Total_Fwd_Packets", "Total_Bwd_Packets",
        "Total_Length_Fwd", "Total_Length_Bwd", "Fwd_Packet_Len_Max",
        "Fwd_Packet_Len_Min", "Fwd_Packet_Len_Mean", "Fwd_Packet_Len_Std",
        "Bwd_Packet_Len_Max", "Bwd_Packet_Len_Min", "Bwd_Packet_Len_Mean",
        "Bwd_Packet_Len_Std", "Flow_Bytes_per_s", "Flow_Packets_per_s",
        "Flow_IAT_Mean", "Flow_IAT_Std", "Flow_IAT_Max", "Flow_IAT_Min",
        "Fwd_IAT_Total", "Fwd_IAT_Mean", "Fwd_IAT_Std", "Fwd_IAT_Max",
        "Fwd_IAT_Min", "Bwd_IAT_Total", "Bwd_IAT_Mean", "Bwd_IAT_Std",
        "Fwd_PSH_Flags", "Bwd_PSH_Flags", "Fwd_URG_Flags",
        "Bwd_URG_Flags", "Fwd_Header_Len", "Bwd_Header_Len",
        "Fwd_Packets_per_s", "Bwd_Packets_per_s", "Min_Packet_Len",
        "Max_Packet_Len", "Packet_Len_Mean", "Packet_Len_Std",
        "Packet_Len_Variance",
    ]

    data_rows, labels = [], []

    for attack_name, (proportion, _) in ATTACK_TYPES.items():
        n = int(n_samples * proportion)
        if attack_name == "BENIGN":
            chunk = rng.normal(loc=0.3, scale=0.15, size=(n, n_features))
        elif "Flood" in attack_name:
            chunk = rng.exponential(scale=0.8, size=(n, n_features))
            chunk[:, 1] += rng.uniform(50, 200, n)   # high packet count
        elif "Amplify" in attack_name:
            chunk = rng.normal(loc=0.7, scale=0.2, size=(n, n_features))
            chunk[:, 4] += rng.uniform(100, 500, n)  # high bwd length
        else:
            chunk = rng.uniform(0.1, 0.9, size=(n, n_features))

        chunk = np.clip(chunk, 0, None)
        data_rows.append(chunk)
        labels.extend([attack_name] * n)

    X = np.vstack(data_rows)
    df = pd.DataFrame(X, columns=feature_names)
    df["Label"] = labels

    # Shuffle
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
# 3. DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────
class DataPreprocessor:
    """Handles all data ingestion, cleaning, and feature engineering."""

    def __init__(self, label_column: str = "Label"):
        self.label_column = label_column
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.selector = None
        self.selected_features = None

    def load_csv(self, filepath: str) -> pd.DataFrame:
        df = pd.read_csv(filepath)
        print(f"[DATA] Loaded {len(df):,} rows × {len(df.columns)} columns from {filepath}")
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        initial = len(df)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        df = df.drop_duplicates()
        print(f"[CLEAN] {initial:,} → {len(df):,} rows after cleaning "
              f"({initial - len(df):,} removed)")
        return df

    def encode_labels(self, y: pd.Series) -> np.ndarray:
        encoded = self.label_encoder.fit_transform(y)
        print(f"[LABELS] Classes: {list(self.label_encoder.classes_)}")
        return encoded

    def select_features(self, X: pd.DataFrame, y: np.ndarray,
                        k: int = 20) -> pd.DataFrame:
        self.selector = SelectKBest(score_func=f_classif, k=k)
        X_sel = self.selector.fit_transform(X, y)
        mask = self.selector.get_support()
        self.selected_features = X.columns[mask].tolist()
        print(f"[FEATURES] Selected top-{k}: {self.selected_features}")
        return pd.DataFrame(X_sel, columns=self.selected_features)

    def scale(self, X_train, X_test):
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s  = self.scaler.transform(X_test)
        return X_train_s, X_test_s

    def preprocess(self, df: pd.DataFrame, n_features: int = 20):
        df = self.clean(df)
        feature_cols = [c for c in df.columns if c != self.label_column]
        X = df[feature_cols].select_dtypes(include=[np.number])
        y = self.encode_labels(df[self.label_column])

        X_sel = self.select_features(X, y, k=n_features)

        X_train, X_test, y_train, y_test = train_test_split(
            X_sel, y,
            test_size=CONFIG["test_size"],
            random_state=CONFIG["random_state"],
            stratify=y,
        )
        X_train_s, X_test_s = self.scale(X_train, X_test)
        print(f"[SPLIT] Train={len(X_train_s):,}  Test={len(X_test_s):,}")
        return X_train_s, X_test_s, y_train, y_test


# ─────────────────────────────────────────────
# 4. MODEL DEFINITIONS
# ─────────────────────────────────────────────
def build_models() -> dict:
    """Return dict of classifier name → sklearn estimator."""
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_split=5,
            n_jobs=-1, random_state=CONFIG["random_state"]
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.1, max_depth=5,
            random_state=CONFIG["random_state"]
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=15, random_state=CONFIG["random_state"]
        ),
        "SVM (RBF)": SVC(
            kernel="rbf", C=10, gamma="scale",
            probability=True, random_state=CONFIG["random_state"]
        ),
        "MLP Neural Net": MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu", solver="adam",
            max_iter=300, early_stopping=True,
            random_state=CONFIG["random_state"]
        ),
        "Naive Bayes": GaussianNB(),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=100, random_state=CONFIG["random_state"]
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=500, n_jobs=-1, random_state=CONFIG["random_state"]
        ),
    }


def build_ensemble(models: dict) -> VotingClassifier:
    """Soft-voting ensemble of top classifiers."""
    estimators = [
        ("rf",  models["Random Forest"]),
        ("gb",  models["Gradient Boosting"]),
        ("mlp", models["MLP Neural Net"]),
    ]
    return VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)


# ─────────────────────────────────────────────
# 5. TRAINER & EVALUATOR
# ─────────────────────────────────────────────
class DDoSClassifier:
    """Train, evaluate, and persist DDoS detection models."""

    def __init__(self, label_encoder: LabelEncoder):
        self.le = label_encoder
        self.results: dict = {}
        self.trained_models: dict = {}

    # ── Training ──────────────────────────────
    def train_all(self, models: dict, X_train, y_train):
        print("\n" + "═" * 60)
        print("  TRAINING PHASE")
        print("═" * 60)
        for name, clf in models.items():
            print(f"  ▶ Training {name} ...", end=" ", flush=True)
            clf.fit(X_train, y_train)
            self.trained_models[name] = clf
            print("done")

        print("  ▶ Training Ensemble Classifier ...", end=" ", flush=True)
        ensemble = build_ensemble(self.trained_models)
        ensemble.fit(X_train, y_train)
        self.trained_models["Ensemble"] = ensemble
        print("done")

    # ── Evaluation ────────────────────────────
    def evaluate_all(self, X_test, y_test):
        print("\n" + "═" * 60)
        print("  EVALUATION PHASE")
        print("═" * 60)
        class_names = self.le.classes_

        summary_rows = []
        for name, clf in self.trained_models.items():
            y_pred = clf.predict(X_test)
            acc  = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            self.results[name] = {
                "accuracy":  round(acc  * 100, 2),
                "precision": round(prec * 100, 2),
                "recall":    round(rec  * 100, 2),
                "f1_score":  round(f1   * 100, 2),
                "y_pred":    y_pred,
                "clf":       clf,
            }
            summary_rows.append({
                "Model": name, "Accuracy (%)": acc * 100,
                "Precision (%)": prec * 100, "Recall (%)": rec * 100,
                "F1 Score (%)": f1 * 100,
            })
            print(f"  {name:<25}  Acc={acc*100:.2f}%  F1={f1*100:.2f}%")

        self.summary_df = pd.DataFrame(summary_rows).sort_values(
            "F1 Score (%)", ascending=False
        )
        return self.summary_df

    # ── Cross-Validation ──────────────────────
    def cross_validate(self, X_train, y_train, cv: int = 5):
        print(f"\n  Cross-Validation ({cv}-fold) on top models ...")
        top_models = ["Random Forest", "Gradient Boosting", "MLP Neural Net"]
        for name in top_models:
            clf = self.trained_models.get(name)
            if clf is None:
                continue
            scores = cross_val_score(clf, X_train, y_train,
                                     cv=cv, scoring="f1_weighted", n_jobs=-1)
            print(f"  {name:<25}  CV-F1: {scores.mean()*100:.2f}% ± {scores.std()*100:.2f}%")

    # ── Save / Load ───────────────────────────
    def save_models(self, directory: str = CONFIG["model_dir"]):
        for name, clf in self.trained_models.items():
            safe = name.replace(" ", "_").replace("(", "").replace(")", "")
            path = os.path.join(directory, f"{safe}.joblib")
            joblib.dump(clf, path)
        print(f"\n  Models saved to '{directory}/'")

    def load_model(self, name: str, directory: str = CONFIG["model_dir"]):
        safe = name.replace(" ", "_").replace("(", "").replace(")", "")
        path = os.path.join(directory, f"{safe}.joblib")
        return joblib.load(path)

    # ── Best Model ────────────────────────────
    @property
    def best_model_name(self) -> str:
        return self.summary_df.iloc[0]["Model"]

    @property
    def best_model(self):
        return self.trained_models[self.best_model_name]


# ─────────────────────────────────────────────
# 6. REAL-TIME PREDICTION PIPELINE
# ─────────────────────────────────────────────
class DDoSPredictor:
    """
    Wraps a trained model + preprocessor for real-time inference.
    Call predict_packet(feature_dict) for single-flow prediction.
    """

    def __init__(self, model, preprocessor: DataPreprocessor):
        self.model = model
        self.preprocessor = preprocessor

    def predict_packet(self, feature_dict: dict) -> dict:
        """
        feature_dict: {feature_name: value, ...}
        Returns: {"label": str, "confidence": float, "is_attack": bool}
        """
        row = pd.DataFrame([feature_dict])
        # Align to selected features
        for col in self.preprocessor.selected_features:
            if col not in row.columns:
                row[col] = 0.0
        row = row[self.preprocessor.selected_features]
        row_s = self.preprocessor.scaler.transform(row)

        pred_idx = self.model.predict(row_s)[0]
        label = self.preprocessor.label_encoder.inverse_transform([pred_idx])[0]

        confidence = 0.0
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(row_s)[0]
            confidence = float(proba.max())

        return {
            "label": label,
            "confidence": round(confidence * 100, 2),
            "is_attack": label.upper() != "BENIGN",
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.preprocessor.selected_features:
            if col not in df.columns:
                df[col] = 0.0
        X = df[self.preprocessor.selected_features]
        X_s = self.preprocessor.scaler.transform(X)
        preds = self.model.predict(X_s)
        labels = self.preprocessor.label_encoder.inverse_transform(preds)
        df = df.copy()
        df["Predicted_Label"] = labels
        df["Is_Attack"] = [lbl.upper() != "BENIGN" for lbl in labels]
        return df


# ─────────────────────────────────────────────
# 7. VISUALIZATION
# ─────────────────────────────────────────────
class Visualizer:

    @staticmethod
    def plot_comparison(summary_df: pd.DataFrame, save_path: str = None):
        fig, ax = plt.subplots(figsize=(12, 6))
        metrics = ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1 Score (%)"]
        x = np.arange(len(summary_df))
        w = 0.2
        colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            ax.bar(x + i * w, summary_df[metric], w, label=metric, color=color, alpha=0.85)
        ax.set_xticks(x + 1.5 * w)
        ax.set_xticklabels(summary_df["Model"], rotation=25, ha="right", fontsize=9)
        ax.set_ylim(0, 110)
        ax.set_ylabel("Score (%)")
        ax.set_title("DDoS Classifier Performance Comparison", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    @staticmethod
    def plot_confusion_matrix(y_test, y_pred, class_names, model_name: str,
                              save_path: str = None):
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    @staticmethod
    def plot_feature_importance(model, feature_names: list, top_n: int = 15,
                                save_path: str = None):
        if not hasattr(model, "feature_importances_"):
            print("  Feature importance not available for this model.")
            return
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        plt.figure(figsize=(10, 5))
        plt.bar(range(top_n),
                importances[indices],
                color=plt.cm.viridis(np.linspace(0, 1, top_n)))
        plt.xticks(range(top_n),
                   [feature_names[i] for i in indices],
                   rotation=45, ha="right", fontsize=8)
        plt.title("Top Feature Importances (Random Forest)", fontsize=13, fontweight="bold")
        plt.ylabel("Importance")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    @staticmethod
    def plot_attack_distribution(y, label_encoder, save_path: str = None):
        labels, counts = np.unique(y, return_counts=True)
        names = label_encoder.inverse_transform(labels)
        plt.figure(figsize=(10, 5))
        colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
        plt.bar(names, counts, color=colors, edgecolor="white", linewidth=1.2)
        plt.title("Attack Type Distribution in Dataset", fontsize=13, fontweight="bold")
        plt.ylabel("Sample Count")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()


# ─────────────────────────────────────────────
# 8. MAIN PIPELINE
# ─────────────────────────────────────────────
def main():
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║   ML-Based DDoS Attack Classification & Prediction       ║")
    print("╚" + "═" * 58 + "╝\n")

    # ── Step 1: Data ──────────────────────────
    print("[STEP 1] Generating / Loading Dataset ...")
    df = generate_synthetic_dataset(n_samples=20_000, n_features=40)
    print(f"  Dataset shape: {df.shape}")
    print(f"  Label distribution:\n{df['Label'].value_counts()}\n")

    # ── Step 2: Preprocess ────────────────────
    print("[STEP 2] Preprocessing ...")
    preprocessor = DataPreprocessor(label_column=CONFIG["label_column"])
    X_train, X_test, y_train, y_test = preprocessor.preprocess(
        df, n_features=CONFIG["n_features"]
    )

    # ── Step 3: Train ─────────────────────────
    models = build_models()
    classifier = DDoSClassifier(label_encoder=preprocessor.label_encoder)
    classifier.train_all(models, X_train, y_train)

    # ── Step 4: Evaluate ──────────────────────
    print("\n[STEP 4] Evaluating Models ...")
    summary = classifier.evaluate_all(X_test, y_test)
    print("\n  ── Summary Table ──")
    print(summary.to_string(index=False))

    # ── Step 5: Cross-Validation ──────────────
    print(f"\n[STEP 5] Cross-Validation ...")
    classifier.cross_validate(X_train, y_train, cv=CONFIG["cv_folds"])

    # ── Step 6: Save ──────────────────────────
    classifier.save_models(CONFIG["model_dir"])
    joblib.dump(preprocessor.scaler,
            os.path.join(CONFIG["model_dir"], "scaler.joblib"))

    joblib.dump(preprocessor.label_encoder,
            os.path.join(CONFIG["model_dir"], "label_encoder.joblib"))

    joblib.dump(preprocessor.selected_features,
            os.path.join(CONFIG["model_dir"], "selected_features.joblib"))
    summary.to_csv(
        os.path.join(CONFIG["results_dir"], "model_comparison.csv"), index=False
    )

    # ── Step 7: Visualize ─────────────────────
    print("\n[STEP 7] Generating Visualizations ...")
    viz = Visualizer()
    viz.plot_comparison(
        summary,
        save_path=os.path.join(CONFIG["results_dir"], "comparison.png")
    )
    best_name = classifier.best_model_name
    viz.plot_confusion_matrix(
        y_test,
        classifier.results[best_name]["y_pred"],
        preprocessor.label_encoder.classes_,
        model_name=best_name,
        save_path=os.path.join(CONFIG["results_dir"], "confusion_matrix.png")
    )
    viz.plot_feature_importance(
        classifier.trained_models["Random Forest"],
        preprocessor.selected_features,
        save_path=os.path.join(CONFIG["results_dir"], "feature_importance.png")
    )
    viz.plot_attack_distribution(
        y_test, preprocessor.label_encoder,
        save_path=os.path.join(CONFIG["results_dir"], "attack_distribution.png")
    )

    # ── Step 8: Real-Time Demo ────────────────
    print("\n[STEP 8] Real-Time Prediction Demo ...")
    predictor = DDoSPredictor(
        model=classifier.best_model,
        preprocessor=preprocessor,
    )
    sample_flow = {f: np.random.uniform(0, 1) for f in preprocessor.selected_features}
    result = predictor.predict_packet(sample_flow)
    print(f"  Sample flow prediction: {result}")

    print("\n  ✅ Pipeline complete. Best model: "
          f"{best_name} (F1={classifier.results[best_name]['f1_score']}%)")
    return classifier, preprocessor, predictor


# ─────────────────────────────────────────────
# 9. ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    clf, prep, pred = main()
