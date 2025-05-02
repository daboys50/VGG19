import streamlit as st
from keras.applications import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input, decode_predictions
import numpy as np

# Load the pre-trained VGG19 model
model = VGG19(weights='imagenet')

# Streamlit UI
st.title('VGG19 Image Classifier')
st.write('Upload an image to classify it using the pre-trained VGG19 model.')

# Upload image through Streamlit interface
img_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if img_file is not None:
    # Load and preprocess the image
    img = image.load_img(img_file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Make predictions
    preds = model.predict(x)

    # Decode predictions
    decoded_preds = decode_predictions(preds, top=3)[0]
    
    # Display image and predictions
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write('Predictions:')
    for i, (imagenet_id, label, prob) in enumerate(decoded_preds):
        st.write(f"{i+1}. {label}: {prob*100:.2f}%")
