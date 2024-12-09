import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time

st.title("Gestura")
# Set page config for a cleaner look
st.set_page_config(layout="wide", page_title="Gestura - Real-time Hand Detection")

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .metrics-container {
        background-color: #1E1E1E;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .result-container {
        background-color: #262730;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7)

# Create two columns for camera and metrics
col1, col2 = st.columns([2, 1])

with col1:
    # Camera view
    st.subheader("Real-time Detection")
    camera_placeholder = st.empty()
    
    # Detection result
    result_placeholder = st.empty()

with col2:
    # Metrics section
    st.subheader("Detection Metrics")
    
    # Create metrics containers
    metrics_placeholder = st.empty()
    
    # Initialize metrics
    if 'total_detections' not in st.session_state:
        st.session_state.total_detections = 0
    if 'successful_detections' not in st.session_state:
        st.session_state.successful_detections = 0
    if 'confidence_scores' not in st.session_state:
        st.session_state.confidence_scores = []

# Start/Stop button
start_button = st.button("Start Detection")

if start_button:
    cap = cv2.VideoCapture(0)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access camera")
                break
                
            # Process frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            # Draw hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame_rgb,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Update metrics
                    st.session_state.total_detections += 1
                    st.session_state.successful_detections += 1
                    confidence = results.multi_handedness[0].classification[0].score
                    st.session_state.confidence_scores.append(confidence)
            
            # Display frame
            camera_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
            
            # Update metrics display
            with metrics_placeholder.container():
                st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
                
                # Calculate accuracy
                accuracy = (st.session_state.successful_detections / max(1, st.session_state.total_detections)) * 100
                avg_confidence = np.mean(st.session_state.confidence_scores) * 100 if st.session_state.confidence_scores else 0
                
                # Display metrics
                col1, col2 = st.columns(2)
                col1.metric("Detection Accuracy", f"{accuracy:.1f}%")
                col2.metric("Avg Confidence", f"{avg_confidence:.1f}%")
                
                st.metric("Total Detections", st.session_state.total_detections)
                
                # Show confidence history chart
                if st.session_state.confidence_scores:
                    st.line_chart(st.session_state.confidence_scores[-50:])  # Show last 50 scores
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Add a small delay
            time.sleep(0.1)
            
            # Check for stop button
            if st.button("Stop", key="stop_button"):
                break
                
    finally:
        cap.release()

