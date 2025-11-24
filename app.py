import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

st.set_page_config(page_title="Apnea Classifier", layout="wide")

# ---------- helpers ----------
def load_raw_224(f):
    img = tf.keras.preprocessing.image.load_img(f, target_size=(224, 224))  # RGB
    x = tf.keras.preprocessing.image.img_to_array(img).astype("float32")    # 0..255
    return x

@st.cache_resource
def load_model_H1():
    return keras.models.load_model("models/H1_best.keras", compile=False)

@st.cache_resource
def load_model_H2():
    return keras.models.load_model("models/H2_best.keras", compile=False)  # <-- FIXED


st.header("Feng Chai University Research on Apnea - Professor David Shen - 2025")

# ---------- UI: choose which model ----------
choice = st.radio("Sambhav Poudel | poudelsb@mail.uc.edu", ["Model Info","H1: Apnea / Hypopnea", "H2: Hypopnea / CSA / MSA / OSA"], horizontal=True)

if choice == "Model Info":

    col1, col2, col3 = st.columns(3)
    import matplotlib.pyplot as plt
    import streamlit as st

    col1, col2, col3 = st.columns(3)
    with col1:
            y = [4305,1376,468,679,2454]
            x = ["Apnea","Hypopnea","MSA","CSA","OSA"]
            title = "Dataset Distribution Used for Training 70%"
            fig, ax = plt.subplots()
            ax.bar(x, y, color="orange")
            ax.set_title(title)
            ax.set_xlabel("Classes")
            ax.set_ylabel("Number of Samples")
            st.pyplot(fig, use_container_width=True)

            st.markdown("""
            <h6 style=' color: #FFA500;'> Functional Model: ResNet50</h3>
            """, unsafe_allow_html=True)
            st.write("Epochs: 10")
            st.write("Batch Size: 32")
            st.write("Optimizer: Adam")
            st.write("Activation Function: Softmax or Sigmoid")

    with col2:    
            x= ["Apnea","Hypopnea","MSA","CSA","OSA"]
            y = [1846,590,201,292,1053]
            title = "Dataset Distribution Used for Validation 30%"
            fig, ax = plt.subplots()
            ax.bar(x, y, color="red")
            ax.set_title(title)
            ax.set_xlabel("Classes")
            ax.set_ylabel("Number of Samples")
            st.pyplot(fig, use_container_width=True)

            st.markdown("""
            <h6 style=' color: #FF4B4B;'>H1 Model Performance on Validation Set</h3>
            """, unsafe_allow_html=True)
            st.write("**Accuracy:**  0.9862")
            st.write("**F1-Score:**  0.9862")
            st.write("**Loss:**  0.2464")

            st.markdown("""
            <h6 style=' color: #FF4B4B;'>H1 Model Performance on Validation Set</h3>
            """, unsafe_allow_html=True)

            st.write("**Accuracy:**  0.9629")
            st.write("**F1-Score:**  0.9631")
            st.write("**Loss:**  0.2738")


    with col3:    
            x= ["Apnea","Hypopnea","MSA","CSA","OSA"]
            y = [2590,1597,167,319,1707]
            title = "Dataset Distribution Used for Testing 30 people"
            fig, ax = plt.subplots()
            ax.bar(x, y, color="green")
            ax.set_title(title)
            ax.set_xlabel("Classes")
            ax.set_ylabel("Number of Samples")
            st.pyplot(fig, use_container_width=True)

            st.markdown("""
            <h6 style=' color: #4CAF50;'>H2 Model Performance on Testing Set</h3>
            """, unsafe_allow_html=True)

            st.write("Accuracy:  0.9419")
            st.write("F1-Score:  0.9448")
            st.write("Loss: 0.5379")

            st.markdown("""
            <h6 style='color: #4CAF50;'>H2 Model Performance on Testing Set</h3>
            """, unsafe_allow_html=True)
            st.write("Accuracy:  0.9489")
            st.write("F1-Score:  0.9432")
            st.write("Loss: 0.4888")

elif choice.startswith("H1"):
    model = load_model_H1()
    CLASS_NAMES = ["Apnea", "Hypopnea"]  # must match training folder order used in training
    st.header("Apnea / Hypopnea")

    files = st.file_uploader("Upload image(s)",
                             type=["png", "jpg", "jpeg"],
                             accept_multiple_files=True,
                             key="uploader_h1")  # unique key

    if files and st.button("Predict (H1)", key="predict_h1"):
        batch = np.stack([load_raw_224(f) for f in files], axis=0)
        preds = model.predict(batch)
        top = np.argmax(preds, axis=1)
        for f, i, p in zip(files, top, preds):
            st.write(f"**{f.name}** → {CLASS_NAMES[i]}  | probs {np.round(p, 3)}")

else:
    model = load_model_H2()
    CLASS_NAMES = ["CSA", "Hypopnea", "MSA", "OSA"]  # must match H2 training order
    st.header("Hypopnea / CSA / MSA / OSA")

    files = st.file_uploader(
        "Upload image(s)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files='directory',         # bool, not 'directory'
        key="uploader_h2"
    )

    if files and st.button("Predict (H2)", key="predict_h2"):
        batch = np.stack([load_raw_224(f) for f in files], axis=0)
        preds = model.predict(batch)
        top = np.argmax(preds, axis=1)                       # predicted class indices
        pred_names = [CLASS_NAMES[i] for i in top]           # class names

        # ---- counts per class ----
        counts = Counter(pred_names)
        total = len(pred_names)

        st.subheader("Prediction counts")
        for cls in CLASS_NAMES:
            st.write(f"{cls}: {counts.get(cls, 0)}")
        st.write(f"**Total images:** {total}")

        # nice table + bar chart
        df_counts = pd.DataFrame({
            "class": CLASS_NAMES,
            "count": [counts.get(c, 0) for c in CLASS_NAMES]
        })
        st.dataframe(df_counts, use_container_width=True)
       

