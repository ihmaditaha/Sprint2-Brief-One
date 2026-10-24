import os
import zipfile
from pathlib import Path
import gdown
import tensorflow as tf
import numpy as np

from src.config import (
    FILE_ID,
    DATASET_ZIP_NAME,
    EXTRACT_DIR,
    CLASS_NAMES,
    IMAGE_SIZE,
    BATCH_SIZE,
    RANDOM_STATE,
)

def download_and_extract_dataset():
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    zip_output = PROJECT_ROOT_ZIP = Path(DATASET_ZIP_NAME)
    
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    
    if not zip_output.exists():
        print("Downloading dataset from Google Drive...")
        gdown.download(url, str(zip_output), quiet=False)
    
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_output, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
        
    data_path = EXTRACT_DIR
    if (data_path / "Data").exists():
        data_path = data_path / "Data"
    elif (data_path / "tumors" / "Data").exists():
        data_path = data_path / "tumors" / "Data"
        
    print(f"Targeting data directory at: {data_path}")
    return data_path

def get_datasets(data_path):

    list_ds = tf.data.Dataset.list_files(str(data_path / "*/*"), shuffle=False)
    
    image_count = sum(1 for _ in data_path.glob("*/*.jpg"))
    print(f"Total images found: {image_count}")
    
    list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)
    
    test_size = int(image_count * 0.1)
    val_size = int(image_count * 0.1)
    
    test_ds = list_ds.take(test_size)
    temp_ds = list_ds.skip(test_size)
    val_ds = temp_ds.take(val_size)
    train_ds = temp_ds.skip(val_size)
    
    return train_ds, val_ds, test_ds

def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    one_hot = parts[-2] == np.array(CLASS_NAMES)
    return one_hot

def decode_img(img):
    img = tf.io.decode_jpeg(img, channels=3)
    return tf.image.resize(img, IMAGE_SIZE)

def process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


def prepare_dataset(ds, training=False):
    ds = ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(1000, seed=RANDOM_STATE)
    rescale_layer = tf.keras.layers.Rescaling(1./127.5, offset=-1)
    ds = ds.map(lambda x, y: (rescale_layer(x), y))
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds