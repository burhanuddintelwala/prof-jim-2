import streamlit as st
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.densenet import DenseNet201
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
import pickle

# Load the pre-trained model and tokenizer
caption_model = load_model(r"saved_model.keras")

with open(r"tokenizer_2.pkl", 'rb') as f:
    tokenizer = pickle.load(f)

max_length = 35  # Define your max_length here

# Function to extract image features using DenseNet
def extract_image_features(image_path, model):
    img = load_img(image_path, target_size=(224, 224))  # Resize image to match model input
    img = img_to_array(img) / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Expand dimensions to fit model input
    feature = model.predict(img, verbose=0)  # Get features from the model
    return feature

# Function to convert integer back to word using the tokenizer
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Function to predict caption for a given image
def predict_caption(image):
    # Load the DenseNet model for feature extraction
    densenet_model = DenseNet201()
    feature_extraction_model = Model(inputs=densenet_model.input, outputs=densenet_model.layers[-2].output)
    
    # Extract features from the image
    feature = extract_image_features(image, feature_extraction_model)

    # Initialize the caption generation
    in_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)

        # Predict the next word
        y_pred = caption_model.predict([feature, sequence])
        y_pred = np.argmax(y_pred)

        # Convert the predicted integer to a word
        word = idx_to_word(y_pred, tokenizer)

        if word is None:
            break
            
        in_text += " " + word
        
        if word == 'endseq':
            break
            
    return in_text.replace("startseq", "").replace("endseq", "").strip()  # Clean the output caption

# Streamlit App
st.title("Image Caption Generator")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Add a button to generate the caption
    if st.button("Generate Caption"):
        # Save the uploaded image temporarily
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Predict the caption
        predicted_caption = predict_caption("temp_image.jpg")
        
        # Display the predicted caption
        st.write("**Predicted Caption:**", predicted_caption)
