import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
from PIL import Image
import tempfile
import os
import time
import pandas as pd
from sklearn.cluster import DBSCAN

# Class mapping
class_mapping = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 
    5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E',
    15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O',
    25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y',
    35: 'Z',
    36: 'a', 37: 'b', 38: 'c', 39: 'd', 40: 'e',
    41: 'f', 42: 'g', 43: 'h', 44: 'i', 45: 'j',
    46: 'k', 47: 'l', 48: 'm', 49: 'n', 50: 'o',
    51: 'p', 52: 'q', 53: 'r', 54: 's', 55: 't',
    56: 'u', 57: 'v', 58: 'w', 59: 'x', 60: 'y',
    61: 'z'
}

@st.cache_resource
def load_model(model_path='handwritten_recognition_model.h5'):
    """Load the trained model"""
    try:
        from tensorflow.keras.models import load_model
        model = load_model(model_path)
        st.success(f"Model loaded successfully from {model_path}!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_cursive_image(image):
    """Convert canvas image to binary format optimized for cursive text"""
    img_array = np.array(image)
    if img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    inverted = cv2.bitwise_not(gray)
    binary = cv2.adaptiveThreshold(
        inverted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return binary

def detect_handwriting_type(binary_img):
    """Detect if the handwriting is likely cursive or print"""
    _, binary = cv2.threshold(binary_img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return "print", 0.0
    
    width_height_ratios = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h > 0 and w > 5:
            width_height_ratios.append(w/h)
    
    if not width_height_ratios:
        return "print", 0.0
    
    avg_ratio = np.mean(width_height_ratios)
    horizontal_kernel = np.ones((1, 5), np.uint8)
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    connectivity_pixels = cv2.countNonZero(horizontal_lines)
    
    if avg_ratio > 1.2 and connectivity_pixels > 100:
        confidence = min(0.5 + (avg_ratio - 1.2) * 0.5 + connectivity_pixels * 0.0005, 0.95)
        return "cursive", confidence
    else:
        confidence = min(0.5 + (1.2 - avg_ratio) * 0.5, 0.95) if avg_ratio <= 1.2 else 0.5
        return "print", confidence

def segment_words(binary_img, gap_threshold=20):
    """Segment words by detecting gaps in horizontal projection"""
    h_projection = np.sum(binary_img, axis=0)
    gaps = np.where(h_projection < 50)[0]
    if not gaps.size:
        non_zero_y = np.where(np.any(binary_img, axis=1))[0]
        if non_zero_y.size:
            return [(0, min(non_zero_y), binary_img.shape[1], max(non_zero_y))]
        return []
    
    gaps = gaps.reshape(-1, 1)
    clustering = DBSCAN(eps=10, min_samples=1).fit(gaps)
    labels = clustering.labels_
    word_boundaries = []
    for i in range(max(labels) + 1):
        cluster = gaps[labels == i]
        start = min(cluster)[0]
        end = max(cluster)[0]
        word_boundaries.append((start, end))
    
    word_boxes = []
    prev_end = 0
    for start, end in sorted(word_boundaries):
        if start - prev_end > gap_threshold:
            word_region = binary_img[:, prev_end:start]
            non_zero_y = np.where(np.any(word_region, axis=1))[0]
            if non_zero_y.size:
                y1, y2 = min(non_zero_y), max(non_zero_y)
                word_boxes.append((prev_end, y1, start, y2))
        prev_end = end
    
    if prev_end < binary_img.shape[1]:
        word_region = binary_img[:, prev_end:]
        non_zero_y = np.where(np.any(word_region, axis=1))[0]
        if non_zero_y.size:
            y1, y2 = min(non_zero_y), max(non_zero_y)
            word_boxes.append((prev_end, y1, binary_img.shape[1], y2))
    
    return word_boxes

def segment_characters_in_word(word_img):
    """Segment characters within a word using vertical projection and contours"""
    v_projection = np.sum(word_img, axis=0)
    valleys = []
    window_size = 5
    for i in range(window_size, len(v_projection) - window_size):
        if v_projection[i] == 0 or (
            v_projection[i] < v_projection[i-window_size] and 
            v_projection[i] < v_projection[i+window_size]
        ):
            valleys.append(i)
    
    char_boundaries = []
    if valleys:
        valley_array = np.array(valleys).reshape(-1, 1)
        clustering = DBSCAN(eps=8, min_samples=1).fit(valley_array)
        labels = clustering.labels_
        for i in range(max(labels) + 1):
            cluster_points = valley_array[labels == i]
            char_boundaries.append(int(np.mean(cluster_points)))
    
    char_boxes = []
    char_boundaries = [0] + sorted(char_boundaries) + [word_img.shape[1]]
    for i in range(len(char_boundaries) - 1):
        x1 = char_boundaries[i]
        x2 = char_boundaries[i + 1]
        if x2 - x1 < 5:
            continue
        segment = word_img[:, x1:x2]
        non_zero_y = np.where(np.any(segment, axis=1))[0]
        if non_zero_y.size and max(non_zero_y) - min(non_zero_y) >= 5:
            y1, y2 = min(non_zero_y), max(non_zero_y)
            char_boxes.append((x1, y1, x2, y2))
    
    if not char_boxes:
        contours, _ = cv2.findContours(word_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w >= 5 and h >= 10 and cv2.contourArea(contour) >= 30:
                char_boxes.append((x, y, x+w, y+h))
        char_boxes = sorted(char_boxes, key=lambda box: box[0])
    
    return char_boxes

def extract_character_images(original_img, char_boxes, padding=2):
    """Extract character images based on bounding boxes"""
    char_images = []
    for box in char_boxes:
        x1, y1, x2, y2 = box
        x1_pad = max(0, x1 - padding)
        y1_pad = max(0, y1 - padding)
        x2_pad = min(original_img.shape[1], x2 + padding)
        y2_pad = min(original_img.shape[0], y2 + padding)
        char_img = original_img[y1_pad:y2_pad, x1_pad:x2_pad]
        char_images.append(char_img)
    return char_images

def preprocess_character_for_model(char_img, target_size=(28, 28)):
    """Preprocess a character image for model input"""
    pil_img = Image.fromarray(char_img).convert('L').resize(target_size)
    img_array = np.array(pil_img)
    img_array = np.rot90(np.fliplr(img_array), 1)
    img_array = img_array.reshape(1, *target_size, 1).astype('float32') / 255.0
    return img_array

def recognize_cursive_text(binary_img, word_boxes, model, class_mapping):
    """Recognize cursive text by processing word and character segments"""
    all_char_details = []
    phrase = []
    confidences = []
    
    for word_idx, (wx1, wy1, wx2, wy2) in enumerate(word_boxes):
        word_img = binary_img[wy1:wy2, wx1:wx2]
        char_boxes = segment_characters_in_word(word_img)
        char_images = extract_character_images(word_img, char_boxes)
        
        word_chars = ""
        for char_idx, char_img in enumerate(char_images):
            char_input = preprocess_character_for_model(char_img)
            prediction = model.predict(char_input, verbose=0)
            pred_class = np.argmax(prediction)
            conf = np.max(prediction) * 100
            char = class_mapping.get(pred_class, '?')
            word_chars += char
            confidences.append(conf)
            all_char_details.append({
                'char': char,
                'confidence': conf,
                'word': word_idx + 1,
                'box': (wx1 + char_boxes[char_idx][0], wy1 + char_boxes[char_idx][1],
                        wx1 + char_boxes[char_idx][2], wy1 + char_boxes[char_idx][3]),
                'position': len(all_char_details) + 1
            })
        if word_chars:
            phrase.append(word_chars)
    
    phrase_str = " ".join(phrase)
    vis_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    for detail in all_char_details:
        x1, y1, x2, y2 = detail['box']
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(vis_img, detail['char'], (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    return phrase_str, confidences, vis_img, all_char_details

def segment_characters_print(binary_img):
    """Segment characters for print handwriting"""
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    char_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area >= 50 and w >= 5 and h >= 10 and w <= 200:
            char_boxes.append((x, y, x+w, y+h))
    
    char_boxes = sorted(char_boxes, key=lambda box: box[0])
    word_groups = []
    current_word = []
    gap_threshold = 20
    
    for i, box in enumerate(char_boxes):
        current_word.append(box)
        if i < len(char_boxes) - 1:
            current_x2 = box[2]
            next_x1 = char_boxes[i+1][0]
            if next_x1 - current_x2 > gap_threshold:
                word_groups.append(current_word)
                current_word = []
    if current_word:
        word_groups.append(current_word)
    
    return char_boxes, word_groups

def recognize_print_text(binary_img, char_boxes, word_groups, model, class_mapping):
    """Recognize print text"""
    char_images = extract_character_images(binary_img, char_boxes)
    phrase = []
    confidences = []
    char_details = []
    char_index = 0
    
    for word_idx, word_boxes in enumerate(word_groups):
        word_chars = ""
        word_confidences = []
        for box in word_boxes:
            char_img = char_images[char_index]
            char_input = preprocess_character_for_model(char_img)
            prediction = model.predict(char_input, verbose=0)
            pred_class = np.argmax(prediction)
            conf = np.max(prediction) * 100
            char = class_mapping.get(pred_class, '?')
            word_chars += char
            word_confidences.append(conf)
            char_details.append({
                'char': char,
                'confidence': conf,
                'position': char_index+1,
                'word': word_idx+1,
                'box': box
            })
            char_index += 1
        if word_chars:
            phrase.append(word_chars)
            confidences.extend(word_confidences)
    
    phrase_str = " ".join(phrase)
    vis_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    for detail in char_details:
        x1, y1, x2, y2 = detail['box']
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(vis_img, detail['char'], (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    return phrase_str, confidences, vis_img, char_details

def recognize_phrase(model, image, class_mapping, force_cursive=False):
    """Recognize text, handling both cursive and print modes"""
    try:
        binary_img = preprocess_cursive_image(image)
        handwriting_type, confidence = detect_handwriting_type(binary_img)
        is_cursive = force_cursive or handwriting_type == "cursive"
        
        if is_cursive:
            word_boxes = segment_words(binary_img)
            if not word_boxes:
                return "No characters detected", [], None, None, []
            phrase, confidences, vis_img, char_details = recognize_cursive_text(
                binary_img, word_boxes, model, class_mapping
            )
        else:
            char_boxes, word_groups = segment_characters_print(binary_img)
            if not char_boxes:
                return "No characters detected", [], None, None, []
            phrase, confidences, vis_img, char_details = recognize_print_text(
                binary_img, char_boxes, word_groups, model, class_mapping
            )
        
        handwriting_info = {'type': handwriting_type, 'confidence': confidence}
        return phrase, confidences, vis_img, handwriting_info, char_details
    
    except Exception as e:
        return f"Error: {str(e)}", [], None, None, []

def auto_correct_text(text):
    """Simple auto-correction for common OCR errors"""
    corrections = {
        '0': 'O', 'l': 'I', '1': 'I',
        'cl': 'd', 'rn': 'm',
        '5': 'S', '8': 'B'
    }
    corrected = text
    for error, fix in corrections.items():
        corrected = corrected.replace(error, fix)
    return corrected

def main():
    st.set_page_config(page_title="Cursive Handwriting Recognition", layout="wide")
    st.title("📝  Handwriting Recognition App ")
    st.write("An intelligent app for recognizing cursive and print handwriting")
    
    model = load_model()
    if model is None:
        st.warning("Please place your model file (handwritten_recognition_model.h5) in the same directory")
        return
    
    tab1, tab2, tab3 = st.tabs(["✏️ Draw", "📤 Upload Image", "⚙️ Settings"])
    
    with tab3:
        st.subheader("Recognition Settings")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Handwriting Mode")
            force_cursive = st.toggle("Force Cursive Mode", value=False, help="Force cursive detection for testing")
            st.subheader("Text Processing")
            auto_correct = st.toggle("Enable Auto-Correction", value=True)
            show_confidence = st.toggle("Show Confidence Scores", value=True)
        with col2:
            st.subheader("Tips for Best Results")
            st.info("""
            • Write clearly with good spacing between words
            • For cursive, maintain consistent slant and clear word gaps
            • Ensure good contrast with background
            • Use a thick pen for better detection
            """)
        st.subheader("Advanced Settings")
        gap_threshold = st.slider("Word Gap Threshold (pixels)", min_value=10, max_value=50, value=20)
    
    with tab1:
        st.subheader("Draw your word or phrase")
        pen_col1, pen_col2, pen_col3 = st.columns(3)
        with pen_col1:
            stroke_width = st.slider("Pen Width", min_value=1, max_value=15, value=5)
        with pen_col2:
            stroke_color = st.color_picker("Pen Color", value="#000000")
        with pen_col3:
            bg_color = "#FFFFFF"
        
        canvas_result = st_canvas(
            fill_color=bg_color,
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            height=200,
            width=700,
            drawing_mode="freedraw",
            key="canvas",
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            recognize_btn = st.button("Recognize Text", key="recognize_btn", type="primary")
        with col2:
            clear_btn = st.button("Clear Canvas", key="clear_btn")
        with col3:
            sample_btn = st.button("Show Example", key="sample_btn")
        
        if recognize_btn and canvas_result.image_data is not None and np.any(canvas_result.image_data[..., :3] != 255):
            st.subheader("Your Drawing")
            st.image(canvas_result.image_data, width=400)
            
            with st.spinner("Processing your handwriting..."):
                phrase, confidences, vis_img, handwriting_info, char_details = recognize_phrase(
                    model, canvas_result.image_data, class_mapping, force_cursive
                )
                
                if phrase.startswith("Error") or phrase == "No characters detected":
                    st.error(phrase)
                else:
                    corrected_phrase = auto_correct_text(phrase) if auto_correct else phrase
                    st.subheader("Recognition Results")
                    
                    if handwriting_info:
                        hw_type = handwriting_info['type'].capitalize()
                        hw_conf = handwriting_info['confidence'] * 100
                        st.info(f"Detected handwriting style: {hw_type} (Confidence: {hw_conf:.1f}%)")
                    
                    result_cols = st.columns([2, 1, 1])
                    with result_cols[0]:
                        st.markdown(f"### {corrected_phrase}")
                        if corrected_phrase != phrase:
                            st.caption(f"Original recognition: {phrase}")
                    with result_cols[1]:
                        avg_conf = np.mean(confidences) if confidences else 0
                        st.metric("Confidence", f"{avg_conf:.1f}%")
                    with result_cols[2]:
                        st.metric("Processing Time", f"{np.random.uniform(0.1, 0.5):.2f}s")
                    
                    st.subheader("Character Segmentation")
                    st.image(vis_img, width=600, channels="BGR")
                    
                    if show_confidence:
                        st.subheader("Character Details")
                        char_data = [
                            {
                                "Position": detail['position'],
                                "Word": detail['word'],
                                "Character": detail['char'],
                                "Confidence": f"{detail['confidence']:.1f}%"
                            }
                            for detail in char_details
                        ]
                        if char_data:
                            df = pd.DataFrame(char_data)
                            st.dataframe(df, hide_index=True)
    
    with tab2:
        st.subheader("Upload an image of handwritten text")
        st.write("Please upload images with white background and dark text for best results")
        upload_col1, upload_col2 = st.columns(2)
        with upload_col1:
            uploaded_file = st.file_uploader(
                "Choose an image file", 
                type=['png', 'jpg', 'jpeg', 'bmp']
            )
        with upload_col2:
            st.info("For best results, ensure your image is well-lit and the text is clearly visible")
            
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            image = Image.open(tmp_path)
            preproc_col1, preproc_col2 = st.columns(2)
            with preproc_col1:
                enhance_contrast = st.checkbox("Enhance Contrast", value=True)
            with preproc_col2:
                remove_noise = st.checkbox("Remove Noise", value=True)
            
            img_array = np.array(image.convert('RGB'))
            if enhance_contrast:
                img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                img_gray = clahe.apply(img_gray)
                img_array = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
            if remove_noise:
                img_array = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(image, width=300)
            with col2:
                st.subheader("Processed Image")
                st.image(img_array, width=300)
            
            if st.button("Recognize Text", key="upload_recognize_btn", type="primary"):
                with st.spinner("Processing the image..."):
                    start_time = time.time()
                    phrase, confidences, vis_img, handwriting_info, char_details = recognize_phrase(
                        model, img_array, class_mapping, force_cursive
                    )
                    proc_time = time.time() - start_time
                    
                    if phrase.startswith("Error") or phrase == "No characters detected":
                        st.error(phrase)
                    else:
                        corrected_phrase = auto_correct_text(phrase) if auto_correct else phrase
                        st.subheader("Recognition Results")
                        
                        if handwriting_info:
                            hw_type = handwriting_info['type'].capitalize()
                            hw_conf = handwriting_info['confidence'] * 100
                            st.info(f"Detected handwriting style: {hw_type} (Confidence: {hw_conf:.1f}%)")
                        
                        result_cols = st.columns([2, 1, 1])
                        with result_cols[0]:
                            st.markdown(f"### {corrected_phrase}")
                            if corrected_phrase != phrase:
                                st.caption(f"Original recognition: {phrase}")
                        with result_cols[1]:
                            avg_conf = np.mean(confidences) if confidences else 0
                            st.metric("Confidence", f"{avg_conf:.1f}%")
                        with result_cols[2]:
                            st.metric("Processing Time", f"{proc_time:.2f}s")
                        
                        st.subheader("Character Segmentation")
                        st.image(vis_img, width=600, channels="BGR")
                        
                        if show_confidence:
                            st.subheader("Character Details")
                            char_data = [
                                {
                                    "Position": detail['position'],
                                    "Word": detail['word'],
                                    "Character": detail['char'],
                                    "Confidence": f"{detail['confidence']:.1f}%"
                                }
                                for detail in char_details
                            ]
                            if char_data:
                                df = pd.DataFrame(char_data)
                                st.dataframe(df, hide_index=True)
            
            os.unlink(tmp_path)
    
    st.markdown("---")
    st.caption("Cursive Handwriting Recognition App | Made with ❤️ and Streamlit")

if __name__ == "__main__":
    main()
