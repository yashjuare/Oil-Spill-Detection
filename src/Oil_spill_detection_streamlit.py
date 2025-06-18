import streamlit as st
import numpy as np
from tensorflow.keras.models import model_from_json
from PIL import Image
from ultralytics import YOLO

# Load CNN model
with open("src/model210425.json", "r") as json_file:
    loaded_model = model_from_json(json_file.read())
loaded_model.load_weights("src/model210425.weights.h5")

# Load YOLOv8 model
yolov8_model = YOLO("src/best.pt")

# Prediction function
def predict(data):
    resized_image = Image.fromarray(data).resize((128, 128))
    test_pred = np.array(resized_image).astype('float32') / 255
    test_pred = np.expand_dims(test_pred, axis=0)
    return loaded_model.predict(test_pred)

# Mask function (YOLOv8 detection)
def mask(data):
    new_results = yolov8_model.predict(data, conf=0.3)
    return new_results[0].plot()  # Returns NumPy image with detection drawn

# App UI
def main():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://www.shutterstock.com/shutterstock/photos/2472934623/display_1500/stock-photo-cargo-container-ship-cargo-vessel-ship-carrying-container-and-running-for-import-export-concept-2472934623.jpg");
            background-size: cover;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Oil Spill Detection System")
    st.sidebar.title("Upload or Choose a Sample Image")

    # Upload image
    file_upload = st.sidebar.file_uploader("Upload SAR image", type=['jpg', 'png'])

    # Sample image selection
    sample_option = st.sidebar.selectbox(
        "Or select a sample image",
        ("", "Class1_sample1.jpg", "Class1_sample2.jpg", "Class1_sample3.jpg",
         "Class1_sample4.jfif", "Class1_sample5.jfif", "Class1_sample6.jfif",
         "Class0_sample1.jpg", "Class0_sample2.jpg", "Class0_sample3.jpg",
         "Class0_sample4.jfif", "Class0_sample5.jfif", "Class0_sample6.jfif"),
        index=0
    )

    # Load image from either file or sample
    data = None
    if file_upload:
        data = Image.open(file_upload)
        st.session_state["image_data"] = np.array(data)
        st.image(data, caption="Uploaded Image", use_column_width=True)
    elif sample_option:
        data = Image.open(f"sample_images/{sample_option}")
        st.session_state["image_data"] = np.array(data)
        st.image(data, caption=f"Sample: {sample_option}", use_column_width=True)
    else:
        st.sidebar.info("Upload an image or choose a sample above.")

    # Predict button
    if st.button("Predict"):
        if "image_data" in st.session_state:
            result = predict(st.session_state["image_data"])
            st.session_state["prediction"] = result
            st.write(f"Raw Model Output: {result}")
            if result[0][0] > 0.5:
                st.success("✅ Oil Spill Detected!")
            else:
                st.info("🟢 No Oil Spill Detected.")

    # View mask button
    if st.button("View Mask Image"):
        if "image_data" in st.session_state:
            mask_img = mask(st.session_state["image_data"])
            st.image(mask_img, caption="Detected Mask (YOLOv8)", use_column_width=True)

# Run app
if __name__ == "__main__":
    main()
