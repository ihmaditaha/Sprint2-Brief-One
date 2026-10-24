import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from src.config import CLASS_NAMES, MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI, MODELS_DIR

def evaluate_model(model, test_dataset, model_name="Model"):
    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=0)
    print(f"[{model_name}] Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.4f}")
    return test_loss, test_accuracy

def get_classification_report(model, test_dataset, prefix=""):
    y_true = np.concatenate([labels.numpy() for _, labels in test_dataset])
    y_true_indices = np.argmax(y_true, axis=1)

    y_probabilities = model.predict(test_dataset, verbose=0)
    y_pred = np.argmax(y_probabilities, axis=1)

    report = classification_report(
        y_true_indices,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,
    )

    matrix = confusion_matrix(y_true_indices, y_pred)
    mlflow.log_text(report, f"evaluation/{prefix}_classification_report.txt")

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {prefix.replace('_', ' ').title()}")

    output_path = Path(f"confusion_matrix_{prefix}.png")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    mlflow.log_artifact(str(output_path))
    output_path.unlink(missing_ok=True)

    print(f"\nClassification Report ({prefix}):\n", report)
    return report, matrix

def evaluate_both_and_register_champion(test_dataset):
    """Loads both transfer learning and fine-tuned models, evaluates them side-by-side,
    compares performance, and registers the superior one as the Production champion."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    
    tl_path = MODELS_DIR / "transfer_learning_best.keras"
    ft_path = MODELS_DIR / "fine_tuning_best.keras"
    
    results = {}
    
    if tl_path.exists():
        print("\n--- Evaluating Transfer Learning Model ---")
        tl_model = tf.keras.models.load_model(tl_path)
        with mlflow.start_run(run_name="Evaluation-Transfer-Learning"):
            loss, acc = evaluate_model(tl_model, test_dataset, "Transfer Learning")
            mlflow.log_metrics({"test_loss": float(loss), "test_accuracy": float(acc)})
            get_classification_report(tl_model, test_dataset, prefix="transfer_learning")
            mlflow.tensorflow.log_model(tl_model, artifact_path="model")
            results["transfer_learning"] = {"loss": loss, "accuracy": acc, "path": str(tl_path)}
    
    if ft_path.exists():
        print("\n--- Evaluating Fine-Tuning Model ---")
        ft_model = tf.keras.models.load_model(ft_path)
        with mlflow.start_run(run_name="Evaluation-Fine-Tuning"):
            loss, acc = evaluate_model(ft_model, test_dataset, "Fine-Tuning")
            mlflow.log_metrics({"test_loss": float(loss), "test_accuracy": float(acc)})
            get_classification_report(ft_model, test_dataset, prefix="fine_tuning")
            mlflow.tensorflow.log_model(ft_model, artifact_path="model")
            results["fine_tuning"] = {"loss": loss, "accuracy": acc, "path": str(ft_path)}
                
    if not results:
        print("No saved models found to evaluate.")
        return None, None

    champion_key = max(results, key=lambda k: (results[k]["accuracy"], -results[k]["loss"]))
    champion_data = results[champion_key]
    
    print(f"\n🏆 Comparative Champion Selected: {champion_key.upper()} with Accuracy: {champion_data['accuracy']:.4f}")
    
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment:
        run_name_query = f"Evaluation-Transfer-Learning" if champion_key == "transfer_learning" else "Evaluation-Fine-Tuning"
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.runName = '{run_name_query}'"
        )
        if runs:
            champion_run_id = runs[0].info.run_id
            model_uri = f"runs:/{champion_run_id}/model"
            model_name = "NeuroVision_MobileNetV2_Champion"
            try:
                model_version = mlflow.register_model(model_uri=model_uri, name=model_name)
                client.transition_model_version_stage(
                    name=model_name,
                    version=model_version.version,
                    stage="Production",
                    archive_existing_versions=True
                )
                print(f"Successfully registered and promoted version {model_version.version} to Production.")
            except Exception as e:
                print(f"Model registration note: {e}")
    return champion_key, results