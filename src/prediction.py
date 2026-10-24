import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
import mlflow
from mlflow.tracking import MlflowClient

from src.config import IMAGE_SIZE, CLASS_NAMES, MLFLOW_TRACKING_URI

def load_champion_model(model_name="NeuroVision_MobileNetV2_Champion"):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    
    try:
        latest_versions = client.get_latest_versions(model_name, stages=["Production"])
        if not latest_versions:
            raise ValueError(f"No model found in 'Production' stage for model name: '{model_name}'.")
            
        production_version = latest_versions[0].version
        model_uri = f"models:/{model_name}/Production"
        
        print(f"Loading Production Champion Model (Version {production_version}) from MLflow Registry...")
        model = mlflow.tensorflow.load_model(model_uri)
        return model
    except Exception as e:
        print(f"Could not load from MLflow registry ({e}). Falling back to local saved model...")
        fallback_path = Path("models/best_model.keras")
        if fallback_path.exists():
            return tf.keras.models.load_model(str(fallback_path))
        raise FileNotFoundError("No model found in MLflow registry or local filesystem.")

def preprocess_image(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    preprocessed_img = img_array / 127.5 - 1
    preprocessed_img = np.expand_dims(preprocessed_img, axis=0)
    return preprocessed_img

def predict(image_path):
    model = load_champion_model()
    
    print(f"Processing image: {image_path}")
    processed_image = preprocess_image(image_path)
    
    predictions = model.predict(processed_image)
    
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    confidence = predictions[predicted_class_index]
        
    return predicted_class_name, confidence

if __name__ == "__main__":
    pass