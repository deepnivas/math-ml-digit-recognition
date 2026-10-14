import streamlit as st
import numpy as np
import cv2
 
from streamlit_drawable_canvas import st_canvas
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from src.pca_from_scratch import PCAFromScratch
from src.lda_from_scratch import LDAFromScratch


st.set_page_config(page_title="Math ML Digit Recognition", page_icon="🔢")


@st.cache_resource
def train_model():
    digits = load_digits()
    X = digits.data
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pca = PCAFromScratch(20)
    X_train_pca = pca.fit_transform(X_train)

    lda = LDAFromScratch(9)
    X_train_lda = lda.fit_transform(X_train_pca, y_train)

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_lda, y_train)

    return pca, lda, model


pca, lda, model = train_model()


st.title("🔢 Mathematics-Driven Digit Recognition")
st.write("Draw a handwritten digit from 0 to 9.")
st.markdown("**64 pixels → PCA 20D → LDA 9D → KNN**")
st.divider()

st.subheader("Draw your digit")

canvas = st_canvas(
    fill_color="rgba(0,0,0,0)",
    stroke_width=18,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    
   return_image_data=True, 
   key="canvas"
)


if st.button("Predict Digit", type="primary"):

    if canvas.image_data is None:
        st.warning("Please draw a digit.")

    else:
        rgba = canvas.image_data.astype(np.uint8)
        gray = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)

        if np.max(gray) == 0:
            st.warning("Please draw a digit.")

        else:
            coords = cv2.findNonZero(gray)
            x, y, w, h = cv2.boundingRect(coords)

            digit = gray[y:y + h, x:x + w]

            size = max(w, h)
            square = np.zeros((size, size), dtype=np.uint8)

            x_offset = (size - w) // 2
            y_offset = (size - h) // 2

            square[
                y_offset:y_offset + h,
                x_offset:x_offset + w
            ] = digit

            digit_8x8 = cv2.resize(
                square,
                (8, 8),
                interpolation=cv2.INTER_AREA
            )

            digit_8x8 = digit_8x8.astype(np.float32) / 255.0 * 16.0

            image = digit_8x8.reshape(1, 64)

            image_pca = pca.transform(image)
            image_lda = lda.transform(image_pca)

            prediction = model.predict(image_lda)[0]
            probabilities = model.predict_proba(image_lda)[0]
            confidence = probabilities[prediction]

            st.divider()
            st.success(f"Predicted Digit: {prediction}")
            st.metric("KNN Confidence", f"{confidence * 100:.2f}%")

            st.subheader("Processed 8 × 8 Image")
            st.image(digit_8x8 /16.0, width=200)
