import streamlit as st
from keras.applications import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D
import numpy as np
import os

# --- Transfer Learning Setup ---
def build_transfer_model(num_classes=5):  # change num_classes to fit your use case
    base_model = VGG19(weights='imagenet', include_top=False)

    # Freeze the convolutional base
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom top layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)  # for multi-class classification

    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# Load or build the model
MODEL_PATH = 'transfer_vgg19_model.h5'

if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    model = build_transfer_model(num_classes=5)  # customize this
    # Note: Model should be trained before use if not already saved

# --- Streamlit UI ---
st.title('Transfer Learning with VGG19')
st.write('Upload an image to classify it using a custom-trained VGG19 model.')

img_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if img_file is not None:
    # Load and preprocess the image
    img = image.load_img(img_file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Make prediction
    preds = model.predict(x)
    pred_class = np.argmax(preds, axis=1)[0]

    # Display results
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write('Predicted Class:', pred_class)
