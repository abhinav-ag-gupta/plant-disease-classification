import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import json
from tensorflow.keras.models import load_model

model = load_model('model.keras')

with open('class_names.json','r') as f:
    class_names = json.load(f)


st.title('Plant Disease Classification App')

st.write("Upload an Image and Model will classify")

uploaded_fle = st.file_uploader("Choose an image",type=['png','jpg','jpeg'])

def preprocess_image(image):
    image=image.resize([128,128])
    img_array = np.array(image)/255.0

    if img_array.shape[-1]==4:
        img_array=img_array[:,:,:3]

    img_array = np.expand_dims(img_array,axis=0)
    return img_array

if uploaded_fle is not None:
    image = Image.open(uploaded_fle)

    st.image(image=image, caption="Uploaded Image", width="stretch")
    img = preprocess_image(image)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    confidence = round(np.max(predictions),3)

    st.subheader("Predictions")
    st.write(f"Class: {class_names[predicted_class]}")
    st.write(f"Confidence:{confidence}")