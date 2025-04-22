import streamlit as st
import pandas as pd 
import numpy as np
import cv2
from tensorflow.keras.models import model_from_json
from PIL import Image
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import keras

json_file = open("src/model210425.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("src/model210425.weights.h5")

def predict(data):
    resized_image = cv2.resize(data, (128, 128))
    test_pred = resized_image.astype('float32') / 255
    test_pred = np.expand_dims(test_pred, axis=0)
    predictions = loaded_model.predict(test_pred)    
    return predictions

def main():
    st.title("Oil Spilled Detection in Sea Water ")
    st.sidebar.title("Oil Spilled Detection in Sea Water")
    html_temp = """
        <div style="background-color:tomato;padding:10px">
        <h2 style="color:white;text-align:center;">Convolutional_neural_network </h2>
        </div>
        
        """
    
    st.markdown(html_temp, unsafe_allow_html = True)
    st.text("")
    

    file_upload = st.sidebar.file_uploader("Choose SAR image file of sea water " ,type=['.jpg','.png'],accept_multiple_files=False,key="fileUploader")
    if file_upload is not None :
            data = Image.open(file_upload)
            
            data = np.array(data)
        
            st.image(data, caption="Uploaded Image.", use_column_width= True)  
        
    else:
        st.sidebar.warning("you need to upload image file.")

            
    result = ""
    if st.button("Predict"):
        result = predict(data)
        if result is not None:
            st.write(f"Prediction result: {result}")
            
            if result[0][0] > 0.5:
                st.write("Prediction: Oil Spill Detected!")
            else:
                st.write("Prediction: No Oil Spill Detected!")
        else:
            st.write("Prediction failed. Please try again.")


if __name__=='__main__':
    main()
