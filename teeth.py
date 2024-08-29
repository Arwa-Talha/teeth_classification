import streamlit as st
import numpy as np 
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
model=load_model(r"C:\Users\arwat\teeth.h5")
st.title("Teeth APP Classification ")
st.write("welcome to my Teeth APP classification")
uploaded_image = st.file_uploader("Upload an Image", type = ['jpg', 'png', 'jpeg'])
if uploaded_image is not None:
    img = Image.open(uploaded_image)  # Open the image
    st.image(img, caption='Uploaded Image..')  # Show image on Streamlit
    
    # 1- Convert image to array
    new_image = np.array(img)
    
    # 2- Resize image to be 128 * 128
    new_image = cv2.resize(new_image, (128, 128))
    
    # 3- Convert image to grayscale (if needed) and then to 3-channel RGB
    if new_image.ndim == 3:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    
    # Convert grayscale to RGB by stacking the grayscale image into 3 channels
    new_image = np.stack((new_image,) * 3, axis=-1)
    
    # 4- Normalize image
    new_image = new_image.astype('float32') / 255.0
    
    # 5- Reshape image to match the model input shape
    new_image = new_image.reshape(1, 128, 128, 3)  # Ensure it has 3 channels
    
 # Make prediction
    prediction = model.predict(new_image)
    predicted_class = np.argmax(prediction) # Get the index of the predicted class  , axis=1)[0] 
    #predicted_class_label = class_labels[predicted_class_index]  # Map index to class label
    
    st.write(f'Predicted Class: {predicted_class}')
