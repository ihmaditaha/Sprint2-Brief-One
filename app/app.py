import streamlit as st
from PIL import Image
from pathlib import Path
import tempfile
import sys

root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0 , str(root_dir))
from src.predict import predict_image


st.set_page_config(page_title="NeuroVision AI", layout="centered")

st.title("NeuroVision AI")
st.write("Brain MRI Image Classification")
st.write("Upload an MRI image to predict its class.")

uploaded_file = st.file_uploader(
    "Choose an MRI image", type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Image", width=350)

    if st.button("Predict"):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
            image_path = Path(temp.name)
            image.save(image_path)
        try:
            result = predict_image(image_path)

            st.success(f"Predicted class: {result['class']}")
            st.metric("Confidence", f"{result['confidence']:.2%}")

        except Exception as error:
            st.error(f"Prediction failed: {error}")

        finally:
            image_path.unlink(missing_ok=True)
