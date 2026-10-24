import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

from src.prediction import predict
from src.config import CLASS_NAMES

st.set_page_config(
    page_title="NeuroVision AI",
    layout="wide",
)

st.title("NeuroVision AI")
st.subheader("Brain MRI Classification")
st.write(
    "Upload a brain MRI image to classify it into one of the four categories using the MLflow Production Champion model."
)


uploaded_file = st.file_uploader(
    "Upload a brain MRI image",
    type=["jpg", "jpeg", "png", "bmp"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded MRI Image",
        width=350,
    )

    if st.button("Predict"):
        with st.spinner("Analyzing image via MLflow Champion model..."):
            
            temp_path = "temp_uploaded_mri.jpg"
            image.save(temp_path)
            predicted_class_name, confidence = predict(temp_path)            


        st.success(f"Predicted Class: **{predicted_class_name.upper()}**")
        st.metric(
            label="Confidence Score",
            value=f"{confidence:.2f}%",
        )
