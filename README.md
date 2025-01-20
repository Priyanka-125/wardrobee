# Wardrobe Suggestion

This is a **Fashion Recommender System** designed to suggest visually similar clothing items based on an uploaded image. It uses deep learning for feature extraction and k-nearest neighbors for finding similar items.

---

## Features

- **Image Upload**: Upload an image of clothing to find similar items.
- **Feature Extraction**: Uses a pre-trained ResNet50 model for extracting image features.
- **Recommendations**: Recommends five similar items from the dataset.
- **Interactive UI**: Built using Streamlit for an intuitive user experience.

---

## How It Works

1. **Preprocessing the Dataset**:
   - Images in the dataset are processed using a pre-trained ResNet50 model.
   - Features are extracted and stored in `embeddings.pkl` for fast lookup.

2. **Image Upload**:
   - Users upload an image of a clothing item.
   - The uploaded image is processed to extract its features.

3. **Similarity Search**:
   - k-Nearest Neighbors (k-NN) is used to find the most similar items based on extracted features.

4. **Recommendation Display**:
   - The system displays the top 5 most similar items from the dataset.

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/SahilTwitZ/wardrobe-suggestion.git
   cd wardrobe-suggestion
   ```

2. **Install Dependencies**:
   Make sure you have Python 3.8 or higher installed. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Dataset**:
   - Place your clothing images in the `images/` folder.
   - Run the `main.py` script to preprocess and generate `embeddings.pkl`.

   ```bash
   python main.py
   ```

4. **Run the Application**:
   Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```

---

## Project Structure

```
Wardrobe Suggestion/
├── app.py               # Main Streamlit application
├── main.py              # Script for feature extraction
├── test.py              # Test scripts for debugging
├── images/              # Folder containing dataset images
├── uploads/             # Folder to store uploaded images temporarily
├── datasets/            # (Optional) Additional dataset-related files
├── embeddings.pkl       # Precomputed features for the dataset
├── filenames.pkl        # List of image filenames
├── requirements.txt     # Python dependencies
├── sample/              # Sample files for testing
└── README.md            # Project documentation
```

---

## Technologies Used

- **Frameworks**:
  - Streamlit (for UI)
  - Scikit-learn (for k-NN)
- **Deep Learning**:
  - TensorFlow & Keras (ResNet50)
- **Languages**:
  - Python

---

## Future Enhancements

- Add support for filtering recommendations by attributes (e.g., color, style).
- Incorporate user feedback to improve recommendation quality.
- Extend the system to recommend matching accessories, shoes, etc.

---

## Contributing

Contributions are welcome! If you'd like to improve this project, please:
1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments

- [ResNet50](https://keras.io/api/applications/resnet/) for feature extraction.
- [Streamlit](https://streamlit.io/) for creating a user-friendly interface.
