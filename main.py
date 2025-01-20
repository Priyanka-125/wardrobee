import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Function to extract features from an image
def feature_extraction(img_path, model):
    img = Image.open(img_path).resize((224, 224))
    img_array = np.array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# Function to generate embeddings and filenames
def generate_embeddings(image_dir, model):
    embeddings = []
    filenames = []
    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            filenames.append(img_path)
            embeddings.append(feature_extraction(img_path, model))
    with open('embeddings.pkl', 'wb') as f:
        pickle.dump(embeddings, f)
    with open('filenames.pkl', 'wb') as f:
        pickle.dump(filenames, f)
    return np.array(embeddings), filenames

# Check if embeddings.pkl and filenames.pkl exist
if not (os.path.exists('embeddings.pkl') and os.path.exists('filenames.pkl')):
    st.info("Generating embeddings for images...")
    # Ensure the images directory exists
    image_dir = 'images'
    if not os.path.exists(image_dir) or not os.listdir(image_dir):
        st.error("The 'images/' directory is missing or empty. Add images to the folder and restart the app.")
        st.stop()
    
    # Load pre-trained ResNet50 model for feature extraction
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    model = tf.keras.Sequential([base_model, GlobalMaxPooling2D()])
    
    # Generate embeddings
    feature_list, filenames = generate_embeddings(image_dir, model)
    st.success("Embeddings generated successfully!")
else:
    # Load precomputed features and filenames
    feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
    filenames = pickle.load(open('filenames.pkl', 'rb'))

# Title of the app
st.title('Fashion Recommender System')

# File upload section
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    # Create uploads directory if it doesn't exist
    uploads_dir = "uploads"
    os.makedirs(uploads_dir, exist_ok=True)

    # Save uploaded file
    uploaded_path = os.path.join(uploads_dir, uploaded_file.name)
    with open(uploaded_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    # Display uploaded image
    display_image = Image.open(uploaded_path)
    st.image(display_image)

    # Extract features from uploaded image
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    model = tf.keras.Sequential([base_model, GlobalMaxPooling2D()])
    features = feature_extraction(uploaded_path, model)

    # Function to recommend similar images based on features
    def recommend(features, feature_list):
        neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
        neighbors.fit(feature_list)
        distances, indices = neighbors.kneighbors([features])
        return indices

    # Get recommendations
    indices = recommend(features, feature_list)

    # Display recommended images
    cols = st.columns(5)
    for i in range(5):
        cols[i].image(filenames[indices[0][i]])
