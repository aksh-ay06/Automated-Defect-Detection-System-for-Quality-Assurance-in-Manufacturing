import streamlit as st
import requests
from PIL import Image
import io

API_URL = "http://api:8000"

st.set_page_config(page_title="Manufacturing QA Inspector", layout="centered")
st.title("Manufacturing Defect Inspector")
st.markdown("Upload a part image to get an instant **PASS** or **FAIL** verdict.")

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded Part", use_container_width=True)

    with st.spinner("Analyzing..."):
        files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
        try:
            resp = requests.post(f"{API_URL}/predict", files=files, timeout=30)
            resp.raise_for_status()
            result = resp.json()
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to the API server. Make sure the API is running.")
            st.stop()
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    label = result["label"]
    prob = result["defect_probability"]
    recon_err = result["reconstruction_error"]
    threshold = result["anomaly_threshold"]
    novel = result["novel_defect_suspected"]

    if label == "PASS":
        st.success(f"**PASS** - Defect probability: {prob:.1%}")
    else:
        st.error(f"**FAIL** - Defect probability: {prob:.1%}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Defect Prob", f"{prob:.1%}")
    col2.metric("Recon Error", f"{recon_err:.5f}")
    col3.metric("Threshold", f"{threshold:.5f}")

    if novel:
        st.warning("Novel / unseen defect type suspected (high reconstruction error).")
