import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
from PIL import Image

# --- App Configuration ---
st.set_page_config(page_title="AI Vision Pro", layout="wide")

# Custom CSS for a better UI
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; }
    </style>
    """, unsafe_allow_index=True)

def main():
    st.title("🤖 AI Vision Pro")
    st.subheader("High-speed Computer Vision (No API required)")
    
    # Sidebar for Task Selection
    option = st.sidebar.selectbox(
        'Choose a Vision Task:',
        ['Face Detection', 'Selfie Segmentation (Background Blur)']
    )

    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert uploaded file to OpenCV format
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)

        with col2:
            if option == 'Face Detection':
                process_face_detection(img_array)
            elif option == 'Selfie Segmentation (Background Blur)':
                process_segmentation(img_array)

def process_face_detection(img):
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        annotated_img = img.copy()
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                ih, iw, _ = annotated_img.shape
                x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)
                cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 255, 0), 5)
            
            st.image(annotated_img, caption="AI Detection Result", use_container_width=True)
            st.success(f"Detected {len(results.detections)} face(s)!")
        else:
            st.warning("No faces detected.")

def process_segmentation(img):
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_seg:
        # Process the image
        results = selfie_seg.process(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        mask = results.segmentation_mask
        
        # Apply blur to background
        condition = np.stack((mask,) * 3, axis=-1) > 0.5
        blurred_img = cv2.GaussianBlur(img, (55, 55), 0)
        output_image = np.where(condition, img, blurred_img)
        
        st.image(output_image, caption="AI Background Blur", use_container_width=True)
        st.info("The AI isolated the subject and blurred the background.")

if __name__ == "__main__":
    main()
