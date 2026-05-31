
# =========================================
# Predictive Maintenance - Model Training
# =========================================

# =========================================
# 1. Import Required Libraries
# =========================================

import os
import joblib
import pandas as pd

import mlflow
import mlflow.sklearn

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from huggingface_hub import (
    HfApi,
    create_repo
)


# =========================================
# 2. Hugging Face Authentication
# =========================================

print("=" * 60)
print("STEP 1: Hugging Face Authentication")
print("=" * 60)

HF_TOKEN = os.getenv("HF_TOKEN")

api = HfApi(token=HF_TOKEN)

print("Hugging Face authentication successful.\n")


# =========================================
# 3. Create HF Model Repository
# =========================================

print("=" * 60)
print("STEP 2: Creating Hugging Face Model Repository")
print("=" * 60)

MODEL_REPO_ID = (
    "geniusut/engine-predictive-maintenance-model"
)

create_repo(
    repo_id=MODEL_REPO_ID,
    repo_type="model",
    exist_ok=True
)

print("Model repository verified successfully.\n")


# =========================================
# 4. Load Train and Test Datasets
# =========================================

print("=" * 60)
print("STEP 3: Loading Train and Test Datasets")
print("=" * 60)

TRAIN_PATH = (
    "hf://datasets/geniusut/engine-predictive-maintenance/train.csv"
)

TEST_PATH = (
    "hf://datasets/geniusut/engine-predictive-maintenance/test.csv"
)

train_df = pd.read_csv(TRAIN_PATH)

test_df = pd.read_csv(TEST_PATH)

print("Train and Test datasets loaded successfully.\n")

print("Train Dataset Shape:", train_df.shape)

print("Test Dataset Shape :", test_df.shape)


# =========================================
# 5. Define Features and Target
# =========================================

print("\n" + "=" * 60)
print("STEP 4: Feature and Target Split")
print("=" * 60)

target_col = "Engine_Condition"

X_train = train_df.drop(columns=[target_col])

y_train = train_df[target_col]

X_test = test_df.drop(columns=[target_col])

y_test = test_df[target_col]

print("X_train Shape:", X_train.shape)

print("X_test Shape :", X_test.shape)

print("y_train Shape:", y_train.shape)

print("y_test Shape :", y_test.shape)


# =========================================
# 6. MLflow Experiment Setup
# =========================================

print("\n" + "=" * 60)
print("STEP 5: MLflow Experiment Setup")
print("=" * 60)

mlflow.set_experiment(
    "Engine_Predictive_Maintenance"
)

print("MLflow experiment created successfully.")


# =========================================
# 7. Create Artifact Folders
# =========================================

os.makedirs(
    "engine-predictive-maintenance/artifacts",
    exist_ok=True
)

os.makedirs(
    "engine-predictive-maintenance/models",
    exist_ok=True
)


# =========================================
# 8. Define Models and Hyperparameters
# =========================================

print("\n" + "=" * 60)
print("STEP 6: Defining Models and Hyperparameters")
print("=" * 60)

models = {

    "DecisionTree": {

        "model": DecisionTreeClassifier(
            random_state=42,
            class_weight='balanced'
        ),

        "params": {

            "max_depth": [3, 5, 10],

            "min_samples_split": [2, 5],

            "criterion": ["gini", "entropy"]
        }
    },


    "RandomForest": {

        "model": RandomForestClassifier(
            random_state=42,
            class_weight='balanced'
        ),

        "params": {

            "n_estimators": [100, 200],

            "max_depth": [5, 10],

            "min_samples_split": [2, 5]
        }
    },


    "XGBoost": {

        "model": XGBClassifier(
            random_state=42,
            eval_metric='logloss'
        ),

        "params": {

            "n_estimators": [100, 200],

            "max_depth": [3, 5],

            "learning_rate": [0.01, 0.1]
        }
    }
}

print("Models and hyperparameter grids defined successfully.")


# =========================================
# 9. Model Training Loop
# =========================================

print("\n" + "=" * 60)
print("STEP 7: Model Training Started")
print("=" * 60)

best_model = None
best_model_name = None
best_recall = 0

model_results = []


for model_name, config in models.items():

    print("\n" + "=" * 60)
    print(f"Training Model: {model_name}")
    print("=" * 60)

    model = config["model"]

    param_grid = config["params"]


    # =====================================
    # Start MLflow Run
    # =====================================

    with mlflow.start_run(run_name=model_name):

        # =================================
        # Grid Search
        # =================================

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=3,
            n_jobs=-1,
            scoring='recall'
        )

        grid_search.fit(X_train, y_train)

        print("GridSearchCV completed successfully.\n")


        # =================================
        # Best Model
        # =================================

        current_best_model = (
            grid_search.best_estimator_
        )

        print("Best Parameters Found:")

        print(grid_search.best_params_)


        # =================================
        # Train Predictions
        # =================================

        y_train_pred = current_best_model.predict(
            X_train
        )

        # =================================
        # Test Predictions
        # =================================

        y_test_pred = current_best_model.predict(
            X_test
        )


        # =================================
        # Train Metrics
        # =================================

        train_accuracy = accuracy_score(
            y_train,
            y_train_pred
        )

        train_precision = precision_score(
            y_train,
            y_train_pred
        )

        train_recall = recall_score(
            y_train,
            y_train_pred
        )

        train_f1 = f1_score(
            y_train,
            y_train_pred
        )



        # =================================
        # Test Metrics
        # =================================

        test_accuracy = accuracy_score(
            y_test,
            y_test_pred
        )

        test_precision = precision_score(
            y_test,
            y_test_pred
        )

        test_recall = recall_score(
            y_test,
            y_test_pred
        )

        test_f1 = f1_score(
            y_test,
            y_test_pred
        )



        # =================================
        # Print Train Metrics
        # =================================

        print("\nTRAIN EVALUATION METRICS")
        print("-" * 40)

        print(f"Train Accuracy : {train_accuracy:.4f}")

        print(f"Train Precision: {train_precision:.4f}")

        print(f"Train Recall   : {train_recall:.4f}")

        print(f"Train F1 Score : {train_f1:.4f}")



        # =================================
        # Print Test Metrics
        # =================================

        print("\nTEST EVALUATION METRICS")
        print("-" * 40)

        print(f"Test Accuracy : {test_accuracy:.4f}")

        print(f"Test Precision: {test_precision:.4f}")

        print(f"Test Recall   : {test_recall:.4f}")

        print(f"Test F1 Score : {test_f1:.4f}")





        # =================================
        # Confusion Matrix
        # =================================

        cm = confusion_matrix(
            y_test,
            y_test_pred
        )

        print("\nConfusion Matrix")
        print("-" * 60)

        tn, fp, fn, tp = cm.ravel()

        cm_df = pd.DataFrame(
            [
                [f"{tn} (TN)", f"{fp} (FP)"],
                [f"{fn} (FN)", f"{tp} (TP)"]
            ],
            index=[
                "Actual Healthy",
                "Actual Maintenance Required"
            ],
            columns=[
                "Predicted Healthy",
                "Predicted Maintenance Required"
            ]
        )

        print(cm_df)


        # =================================
        # Confusion Matrix Plot
        # =================================

        plt.figure(figsize=(6, 5))

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues'
        )

        plt.title(
            f"{model_name} - Confusion Matrix"
        )

        plt.xlabel("Predicted Label")

        plt.ylabel("Actual Label")

        confusion_matrix_path = (
            f"engine-predictive-maintenance/artifacts/"
            f"{model_name}_confusion_matrix.png"
        )

        plt.savefig(confusion_matrix_path)

        plt.close()

        print("Confusion matrix plot saved.")


        # =================================
        # Feature Importance
        # =================================

        feature_importance = pd.DataFrame({

            "Feature": X_train.columns,

            "Importance": (
                current_best_model.feature_importances_
            )
        })

        feature_importance = (
            feature_importance.sort_values(
                by="Importance",
                ascending=False
            )
        )

        print("\nFeature Importance")
        print("-" * 40)

        print(feature_importance)


        # =================================
        # Feature Importance Plot
        # =================================

        plt.figure(figsize=(10, 6))

        sns.barplot(
            data=feature_importance,
            x="Importance",
            y="Feature"
        )

        plt.title(
            f"{model_name} - Feature Importance"
        )

        feature_plot_path = (
            f"engine-predictive-maintenance/artifacts/"
            f"{model_name}_feature_importance.png"
        )

        plt.savefig(feature_plot_path)

        plt.close()

        print("Feature importance plot saved.")


        # =================================
        # Save Feature Importance CSV
        # =================================

        feature_importance_path = (
            f"engine-predictive-maintenance/artifacts/"
            f"{model_name}_feature_importance.csv"
        )

        feature_importance.to_csv(
            feature_importance_path,
            index=False
        )


        # =================================
        # Log Parameters
        # =================================

        mlflow.log_param(
            "model_name",
            model_name
        )

        mlflow.log_params(
            grid_search.best_params_
        )


        # =================================
        # Log Test Metrics
        # =================================

        mlflow.log_metric(
            "test_accuracy",
            test_accuracy
        )

        mlflow.log_metric(
            "test_precision",
            test_precision
        )

        mlflow.log_metric(
            "test_recall",
            test_recall
        )

        mlflow.log_metric(
            "test_f1_score",
            test_f1
        )




        # =================================
        # Log Artifacts
        # =================================

        mlflow.log_artifact(
            confusion_matrix_path
        )

        mlflow.log_artifact(
            feature_plot_path
        )

        mlflow.log_artifact(
            feature_importance_path
        )


        # =================================
        # Log Model
        # =================================

        mlflow.sklearn.log_model(
            current_best_model,
            f"{model_name}_model"
        )

        print("Model logged to MLflow successfully.")


        # =================================
        # Save Results
        # =================================

        model_results.append({

            "Model": model_name,

            "Train Accuracy": round(train_accuracy, 4),

            "Test Accuracy": round(test_accuracy, 4),

            "Train Precision": round(train_precision, 4),

            "Test Precision": round(test_precision, 4),

            "Train Recall": round(train_recall, 4),

            "Test Recall": round(test_recall, 4),

            "Train F1 Score": round(train_f1, 4),

            "Test F1 Score": round(test_f1, 4)
        })


        # =================================
        # Select Best Model
        # =================================

        if test_recall > best_recall:

            best_recall = test_recall

            best_model = current_best_model

            best_model_name = model_name


# =========================================
# 10. Model Comparison Summary
# =========================================

print("\n" + "=" * 60)
print("MODEL COMPARISON SUMMARY")
print("=" * 60)

results_df = pd.DataFrame(model_results)
pd.set_option('display.max_columns', None)

pd.set_option('display.width', None)

pd.set_option('display.max_colwidth', None)

print(results_df)


# =========================================
# 11. Save Model Comparison CSV
# =========================================

comparison_path = (
    "engine-predictive-maintenance/artifacts/"
    "model_comparison.csv"
)

results_df.to_csv(
    comparison_path,
    index=False
)


# =========================================
# 12. Save Best Model
# =========================================

print("\n" + "=" * 60)
print("BEST MODEL SELECTION")
print("=" * 60)

print(f"Best Model : {best_model_name}")

print(f"Best Test Recall: {best_recall:.4f}")

best_model_path = (
    "engine-predictive-maintenance/models/"
    "best_model.pkl"
)

joblib.dump(
    best_model,
    best_model_path
)

print(f"\nBest model saved at: {best_model_path}")


# =========================================
# 13. Upload Best Model to HF
# =========================================

print("\nUploading best model to Hugging Face...")

api.upload_file(
    path_or_fileobj=best_model_path,
    path_in_repo="best_model.pkl",
    repo_id=MODEL_REPO_ID,
    repo_type="model"
)

print("Best model uploaded successfully.")


# =========================================
# 14. Completion Message
# =========================================

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("""
Tasks Completed:
1. Train/Test datasets loaded
2. MLflow experiment created
3. Multiple models trained
4. Hyperparameter tuning completed
5. Train/Test evaluation metrics generated
6. Confusion matrix plotted
7. Feature importance analysis completed
8. Parameters and metrics logged
9. Artifacts logged to MLflow
10. Model comparison completed
11. Best model selected
12. Best model uploaded to Hugging Face Model Hub
""")
