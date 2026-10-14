import streamlit as st
import numpy as np
import cv2
import joblib

from pathlib import Path
from streamlit_drawable_canvas import st_canvas


st.set_page_config(
    page_title="MNIST Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "mnist_model.joblib"
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error("MNIST model not found. Run the training script first.")
    st.stop()


bundle = load_model()

pca = bundle["pca"]
model = bundle["model"]
model_name = bundle["model_name"]
test_accuracy = bundle["accuracy"]


def preprocess_digit(rgba):

    gray = cv2.cvtColor(
        rgba.astype(np.uint8),
        cv2.COLOR_RGBA2GRAY
    )

    _, mask = cv2.threshold(
        gray, 20, 255, cv2.THRESH_BINARY
    )

    coords = cv2.findNonZero(mask)

    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    crop = gray[y:y + h, x:x + w]

    # Preserve the digit's proportions.
    scale = 20.0 / max(w, h)

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    if max(w, h) > 20:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_CUBIC

    resized = cv2.resize(
        crop,
        (new_w, new_h),
        interpolation=interpolation
    )

    # Place the digit on a 28x28 black canvas.
    image = np.zeros((28, 28), dtype=np.uint8)

    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    image[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = resized

    return image


# ======================================================
# INTERFACE
# ======================================================

st.title("🔢 MNIST Handwritten Digit Recognition")

st.write(
    "Draw a digit from 0 to 9. The application will "
    "process it using your PCA implementation and "
    "the trained SVM classifier."
)

st.markdown(
    f"**Model:** {model_name}  \n"
    f"**Held-out MNIST test accuracy:** "
    f"{test_accuracy * 100:.2f}%"
)

st.caption(
    "Test accuracy is measured on MNIST test images, "
    "not on user-drawn images."
)

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
    key="mnist_canvas"
)


if st.button("Predict Digit", type="primary"):

    if canvas.image_data is None:
        st.warning("Draw a digit first.")

    else:
        processed = preprocess_digit(canvas.image_data)

        if processed is None:
            st.warning("Draw a digit first.")

        else:
            # The model was trained on raw pixel values
            # in the range 0 to 255.
            features = processed.astype(
                np.float32
            ).reshape(1, 784)

            # Apply the PCA learned from training data.
            reduced = pca.transform(features)
            # Predict the digit
            prediction = int(model.predict(reduced)[0])

            # Calculate SVM confidence
            probabilities = model.predict_proba(reduced)[0]
            confidence = float(probabilities[prediction])

            # Display prediction and confidence
            st.divider()
            st.subheader("Prediction")
            st.success(f"Predicted digit: {prediction}")

            st.metric(
                "Model Confidence",
                f"{confidence * 100:.2f}%"
            )

            st.subheader("Processed 28 × 28 Image")

            # Classify with the saved SVM.
           
            probabilities = model.predict_proba(reduced)[0]
            confidence = float(probabilities[int(prediction)])

            st.divider()
            st.subheader("Prediction")

            st.success(
                f"Predicted digit: {prediction}"
            )
            st.metric(
                    "Model confidence",
                    f"{confidence * 100:.2f}%"
                )
            st.subheader("Processed 28 × 28 input")

            st.image(
                processed,
                width=200,
                clamp=True
            )

            st.caption(
                "This is the image representation supplied "
                "to the trained PCA + SVM model."
            )