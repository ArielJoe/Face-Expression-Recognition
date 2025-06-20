import streamlit as st
import cv2
import numpy as np
from skimage.transform import resize
import joblib
import matplotlib.pyplot as plt
import time
from PIL import Image
import os

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('best_facial_expression_model.pkl')
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

# Preprocess image (adapted for NumPy array input)
def preprocess_image(img):
    # Convert to grayscale if not already
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Resize to 64x64
    img = resize(img, (64, 64), anti_aliasing=True)
    # Scale to 0-255 and convert to uint8
    img = (img * 255).astype(np.uint8)
    # Flatten to (1, 4096)
    img_flat = img.reshape(1, -1)
    return img, img_flat

# Predict expression and generate visualization
def predict_expression(model, img):
    img_original, img_processed = preprocess_image(img)
    proba = model.predict_proba(img_processed)[0]
    pred = np.argmax(proba)
    label_map = {0: 'neutral', 1: 'happy', 2: 'sad', 3: 'angry'}
    expression = label_map[pred]
    confidence = proba[pred] * 100

    # Create probability bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(label_map.values(), proba * 100, color='skyblue')
    bars[pred].set_color('salmon')
    ax.set_ylabel('Probability (%)')
    ax.set_title(f'Prediction: {expression} ({confidence:.1f}%)')
    ax.set_ylim(0, 100)
    for i, prob in enumerate(proba):
        ax.text(i, prob * 100 + 2, f'{prob*100:.1f}%', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()

    return expression, confidence, fig

# Streamlit app
def main():
    st.title("Real-Time Facial Expression Detection")
    st.write("This app uses your webcam (locally) or uploaded images to detect facial expressions in real-time.")

    # Load model
    model = load_model()
    if model is None:
        return

    # Initialize face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        st.error("Error loading face detector. Please ensure OpenCV is installed correctly.")
        return

    # Check if running on Streamlit Cloud (no webcam available)
    is_cloud = os.getenv("STREAMLIT_CLOUD") == "true" or "streamlitcloud" in os.getenv("SERVER_SOFTWARE", "").lower()

    # Camera selection dropdown (only for local use)
    cap = None
    if not is_cloud:
        camera_options = [f"Camera {i}" for i in range(3)]  # Options for indices 0, 1, 2
        selected_camera = st.selectbox("Select your camera", camera_options, index=0)
        camera_index = camera_options.index(selected_camera)

        # Initialize video capture with selected index
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            st.error(f"Cannot access {selected_camera}. This may be due to missing browser or system permissions.")
            st.write("**Steps to Grant Permission:**")
            st.write("1. Ensure no other app is using the webcam.")
            st.write("2. In your browser (e.g., Chrome):")
            st.write("   - Go to Settings > Privacy and Security > Site Settings > Camera.")
            st.write("   - Allow camera access for this site (http://localhost:8501).")
            st.write("3. Check system permissions (e.g., Windows: Settings > Privacy > Camera).")
            st.write("4. Refresh the page and try a different camera index if needed.")
        else:
            st.success(f"Webcam {selected_camera} is accessible.")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    else:
        st.warning("Webcam access is not available on this hosted version. Please use the image upload option below.")

    # Create placeholders for video, chart, and status
    video_placeholder = st.empty()
    chart_placeholder = st.empty()
    status_placeholder = st.empty()

    # File uploader for manual image upload (available locally and on cloud)
    uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        try:
            # Convert uploaded file to image array
            img = np.array(Image.open(uploaded_file))
            expression, confidence, fig = predict_expression(model, img)
            st.image(img, caption=f"Prediction: {expression} ({confidence:.1f}%)", use_container_width=True)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error processing uploaded image: {str(e)}")

    # Toggle for running/stopping webcam (only for local use)
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False

    if not is_cloud and st.button("Start/Stop Camera"):
        st.session_state.is_running = not st.session_state.is_running
        if not st.session_state.is_running and cap is not None:
            cap.release()
            status_placeholder.write("Camera stopped. Click 'Start/Stop Camera' to begin.")
            chart_placeholder.empty()

    while not is_cloud and st.session_state.is_running and cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            status_placeholder.error("Failed to capture video frame. Try restarting the camera or checking permissions.")
            st.session_state.is_running = False
            cap.release()
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            (x, y, w, h) = faces[0]  # Process first face
            face_img = gray[y:y+h, x:x+w]
            try:
                expression, confidence, fig = predict_expression(model, face_img)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f'{expression} ({confidence:.1f}%)', (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                chart_placeholder.pyplot(fig)
                status_placeholder.write(f"Detected: {expression} with {confidence:.1f}% confidence")
            except Exception as e:
                status_placeholder.error(f"Prediction error: {str(e)}")
                chart_placeholder.empty()
        else:
            status_placeholder.write("No face detected.")
            chart_placeholder.empty()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB")

        time.sleep(0.03)

    # Ensure camera is released when stopped
    if not is_cloud and not st.session_state.is_running and cap is not None:
        cap.release()
        status_placeholder.write("Camera stopped. Click 'Start/Stop Camera' to begin.")
        chart_placeholder.empty()

if __name__ == "__main__":
    main()