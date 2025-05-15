# Handwritten Word/Phrase Recognition App Documentation

## Project Overview
The Handwritten Word/Phrase Recognition App is a Streamlit-based web application designed to recognize handwritten words or phrases containing digits (0-9), uppercase letters (A-Z), and lowercase letters (a-z). Users can input handwriting via a drawing canvas or by uploading an image. It offers a user-friendly interface that makes the recognition process intuitive and accessible. The app uses OpenCV for character segmentation and a pre-trained TensorFlow/Keras model for character recognition, providing detailed results including the recognized phrase, confidence scores, and a visualization of segmented characters.

### Purpose
The project aims to provide an interactive tool for recognizing handwritten text, suitable for educational, prototyping, or research purposes. It demonstrates the integration of computer vision (OpenCV) and deep learning (TensorFlow/Keras) in a user-friendly web interface.

### Key Features
- **Drawing Canvas**: Draw words/phrases on a white canvas with black ink for real-time recognition.
- **Image Upload**: Upload images (PNG, JPG, JPEG, BMP) of handwritten text for processing.
- **Character Segmentation**: Uses OpenCV to segment individual characters with bounding boxes.
- **Recognition**: Recognizes 62 classes (0-9, A-Z, a-z) using a pre-trained model.
- **Visualization**: Displays the processed image with red bounding boxes and predicted characters.
- **Confidence Scores**: Provides per-character and average confidence metrics.
- **Error Handling**: Gracefully handles cases like missing models or undetectable characters.

## Technical Details
### Architecture
The app follows a modular pipeline:
1. **Input Handling**: Accepts input from a Streamlit canvas or uploaded image.
2. **Preprocessing**: Converts images to grayscale and inverts colors (black text on white background).
3. **Segmentation**: Uses OpenCV to detect and sort character bounding boxes.
4. **Character Extraction**: Crops individual characters with padding.
5. **Prediction**: Preprocesses characters (resize, normalize, rotate) and feeds them to the model.
6. **Output**: Displays the recognized phrase, confidence scores, and visualization.

### Components
- **Streamlit**: Provides the web interface with tabs for drawing and image upload.
- **OpenCV**: Handles image preprocessing and character segmentation.
- **TensorFlow/Keras**: Loads and uses the pre-trained model for character recognition.
- **Pillow (PIL)**: Assists with image resizing and format conversion.
- **NumPy**: Manages array operations for image processing.
- **streamlit-drawable-canvas**: Enables the interactive drawing canvas.

### Model
- **File**: `handwritten_recognition_model.h5` (must be in the same directory).
- **Classes**: 62 (0-9: classes 0-9; A-Z: classes 10-35; a-z: classes 36-61).
- **Input**: 28x28 grayscale images, normalized to [0, 1], with specific rotation/flip preprocessing.
- **Output**: Probability distribution over 62 classes.

### Segmentation Process
- **Preprocessing**: Converts input to grayscale, inverts colors (`cv2.bitwise_not`).
- **Thresholding**: Applies binary thresholding (`cv2.threshold`) for contour detection.
- **Contour Detection**: Uses `cv2.findContours` to identify character boundaries.
- **Filtering**: Excludes contours based on area, width, and height thresholds.
- **Sorting**: Orders bounding boxes left-to-right for correct character sequence.
- **Extraction**: Crops character images with padding for model input.

### User Interface
- **Tabs**:
  - **Draw**: Canvas (800x150 pixels, white background, black stroke) for handwriting input.
  - **Upload Image**: File uploader for PNG, JPG, JPEG, BMP images.
- **Outputs**:
  - Recognized phrase and average confidence (displayed as metrics).
  - Visualization image with red bounding boxes and predicted characters.
  - Per-character details (character, confidence, position).
- **Feedback**: Spinner during processing, error messages for issues (e.g., no characters detected).

## Setup and Installation
### Prerequisites
- Python 3.7 or higher
- Pre-trained model file (`handwritten_recognition_model.h5`)
- Internet connection for initial package installation

### Dependencies
- `streamlit`
- `tensorflow`
- `opencv-python-headless`
- `numpy`
- `pillow`
- `streamlit-drawable-canvas`

### Installation Steps
1. **Clone or Download the Project**:
   - Save the script as `app10.py`.
   - Place `handwritten_recognition_model.h5` in the same directory.

2. **Set Up a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install streamlit tensorflow opencv-python-headless numpy pillow streamlit-drawable-canvas
   ```

4. **Run the App**:
   ```bash
   streamlit run app10.py
   ```
   - The app will open in your default browser (e.g., `http://localhost:8501`).

## Usage Instructions
1. **Launch the App**:
   - Run the command above to start the Streamlit server.
2. **Draw Tab**:
   - Draw a word/phrase on the canvas using the mouse.
   - Wait for the app to process (spinner appears).
   - View results: recognized phrase, confidence, segmented characters, and details.
3. **Upload Image Tab**:
   - Upload an image with handwritten text (white background, dark text recommended).
   - Results are displayed similarly to the Draw tab.
4. **Interpret Results**:
   - **Recognized Phrase**: The predicted text.
   - **Average Confidence**: Mean confidence across all characters.
   - **Character Segmentation**: Visual with bounding boxes and predicted characters.
   - **Character Details**: Lists each character, its confidence, and position.

## Limitations
- **Model Dependency**: Requires `handwritten_recognition_model.h5` to function.
- **Segmentation**: May fail with cursive, overlapping, or very small/large characters.
- **Input Constraints**: Canvas size limits phrase length; uploaded images must be clear.
- **Performance**: Processing speed depends on input complexity and system resources.
- **Supported Formats**: Only PNG, JPG, JPEG, BMP for uploads.

## Troubleshooting
- **Model Not Found**: Ensure `handwritten_recognition_model.h5` is in the script directory.
- **No Characters Detected**: Verify input has clear, dark text on a white background.
- **Canvas Issues**: Refresh the browser or clear the cache (`streamlit cache clear`).
- **Dependency Errors**: Reinstall packages or use a clean virtual environment.
- **Low Accuracy**: Check input quality or model training data compatibility.

## Future Enhancements
- Add real-time recognition during drawing.
- Improve segmentation for cursive or complex handwriting.
- Support model retraining or fine-tuning via the UI.
- Allow colored text or additional image formats.
- Implement canvas undo/redo functionality.

## License
This project is for educational and non-commercial use. Ensure compliance with licenses for TensorFlow, OpenCV, and other dependencies. The pre-trained model’s license must also be verified.

---

# README.md

## Handwritten Word/Phrase Recognition App

A Streamlit web app for recognizing handwritten words or phrases (digits 0-9, letters A-Z, a-z) using OpenCV for character segmentation and a TensorFlow/Keras model for recognition. Users can draw on a canvas or upload images to see the recognized text, confidence scores, and segmented characters.

### Features
- Draw words/phrases on a canvas or upload images (PNG, JPG, JPEG, BMP).
- Recognizes 62 characters (0-9, A-Z, a-z) with a pre-trained model.
- Segments characters using OpenCV and displays bounding boxes.
- Shows recognized phrase, confidence scores, and visualization.

### Prerequisites
- Python 3.7+
- Pre-trained model file: `handwritten_recognition_model.h5`

### Installation
1. **Clone/Download**:
   - Save the script as `app10.py`.
   - Place `handwritten_recognition_model.h5` in the same directory.
2. **Install Dependencies**:
   ```bash
   pip install streamlit tensorflow opencv-python-headless numpy pillow streamlit-drawable-canvas
   ```
3. **Run the App**:
   ```bash
   streamlit run app10.py
   ```

### Usage
- **Draw**: Use the canvas to write a word/phrase.
- **Upload**: Upload an image with handwritten text.
- **View Results**: See the recognized phrase, confidence, segmented characters, and details.

### File Structure
```
project_directory/
├── app10.py                      # Main Streamlit script
├── handwritten_recognition_model.h5  # Pre-trained model
```

### Troubleshooting
- **Model Error**: Ensure `handwritten_recognition_model.h5` is present.
- **No Recognition**: Use clear, dark text on a white background.
- **Canvas Issues**: Refresh browser or clear cache (`streamlit cache clear`).

### License
For educational use. Check licenses for TensorFlow, OpenCV, and the model.
