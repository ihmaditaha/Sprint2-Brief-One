import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import mlflow
import mlflow.tensorflow

from src.config import (
    IMAGE_SIZE,
    IMAGE_CHANNELS,
    NUM_CLASSES,
    MODELS_DIR,
    TRANSFER_LEARNING_EPOCHS,
    TRANSFER_LEARNING_LR,
    FINE_TUNING_EPOCHS,
    FINE_TUNING_LR,
    FINE_TUNE_LAST_LAYERS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    BATCH_SIZE,
)

def build_model():
    base_model = MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(*IMAGE_SIZE, IMAGE_CHANNELS)
    )
    base_model.trainable = False

    model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(name="flatten1"),
    tf.keras.layers.Dense(128, activation='relu',name="FC1"),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(NUM_CLASSES,activation='softmax',name="FC2")
    ],name="cnn_model")
    return model, base_model

def get_callbacks(model_path):
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(filepath=str(model_path), monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6),
        tf.keras.callbacks.BackupAndRestore(str(model_path.parents[-1]/"backup"), save_freq="epoch")
    ]

def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

def train_transfer_learning(model, train_dataset, val_dataset):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    setup_mlflow()
    mlflow.tensorflow.autolog(log_models=False)

    with mlflow.start_run(run_name="Transfer-Learning"):
        mlflow.log_params({
            "model_name": "MobileNetV2",
            "training_stage": "transfer_learning",
            "epochs": TRANSFER_LEARNING_EPOCHS,
            "learning_rate": TRANSFER_LEARNING_LR,
            "batch_size": BATCH_SIZE,
        })

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=TRANSFER_LEARNING_LR),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=TRANSFER_LEARNING_EPOCHS,
            callbacks=get_callbacks(MODELS_DIR / "transfer_learning_best.keras"),
        )

        fig = plt.figure()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(['Train', 'Val'], loc='upper left')
        mlflow.log_figure(fig, "plots/TL_acc.png")

        fig = plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Val'], loc='upper left')
        mlflow.log_figure(fig, "plots/TL_loss.png")

        mlflow.log_metrics({
            "best_val_accuracy": float(max(history.history["val_accuracy"])),
            "best_val_loss": float(min(history.history["val_loss"])),
        })
        mlflow.log_artifact(str(MODELS_DIR / "transfer_learning_best.keras"))

    return model, history

def train_fine_tuning(model, base_model, train_dataset, val_dataset):
    base_model.trainable = True
    for layer in base_model.layers[:-FINE_TUNE_LAST_LAYERS]:
        layer.trainable = False

    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNING_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    setup_mlflow()
    mlflow.tensorflow.autolog(log_models=False)

    with mlflow.start_run(run_name="Fine-Tuning"):
        mlflow.log_params({
            "model_name": "MobileNetV2",
            "training_stage": "fine_tuning",
            "epochs": FINE_TUNING_EPOCHS,
            "learning_rate": FINE_TUNING_LR,
            "fine_tune_last_layers": FINE_TUNE_LAST_LAYERS,
            "batch_size": BATCH_SIZE,
        })

        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=FINE_TUNING_EPOCHS,
            callbacks=get_callbacks(MODELS_DIR / "fine_tuning_best.keras"),
        )
        fig = plt.figure()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(['Train', 'Val'], loc='upper left')
        mlflow.log_figure(fig, "plots/FT_acc.png")

        fig = plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Val'], loc='upper left')
        mlflow.log_figure(fig, "plots/FT_loss.png")

        mlflow.log_metrics({
            "best_val_accuracy": float(max(history.history["val_accuracy"])),
            "best_val_loss": float(min(history.history["val_loss"])),
        })
        mlflow.log_artifact(str(MODELS_DIR / "fine_tuning_best.keras"))

    return model, history