import cv2
import mediapipe as mp
import pickle
import numpy as np
import streamlit as st
import time
from collections import deque

model_dict21 = pickle.load(open('./model21.p', 'rb'))
model21 = model_dict21['model']

model_dict42 = pickle.load(open('./model42.p', 'rb'))
model42 = model_dict42['model']

def initialize_tracking():
    return {
        'detection_history': deque(maxlen=5),
        'last_detection': None,
        'current_letter': None,
        'frame_predictions': [],
        'last_process_time': time.time(),
        'process_interval': 1.5,
        'is_collecting': False,
        'collection_start_time': None,
        'countdown_active': False,
        'countdown_start': None,
        'is_detection_paused': False,
        'total_detections': 0,
        'successful_detections': 0,
        'fps_history': deque(maxlen=30),
        'confidence_history': deque(maxlen=30)
    }

def process_predictions(predictions):
    if not predictions:
        return None, 0.0

    pred_stats = {}
    for pred, conf in predictions:
        if pred not in pred_stats:
            pred_stats[pred] = {'count': 0, 'confidence': 0}
        pred_stats[pred]['count'] += 1
        pred_stats[pred]['confidence'] = max(pred_stats[pred]['confidence'], conf)

    best_pred = max(pred_stats.items(), key=lambda x: x[1]['confidence'])
    return best_pred[0], best_pred[1]['confidence']

def main():
    st.set_page_config(layout="wide", page_title="Gestura")
    
    # CSS Styling
    st.markdown("""
        <style>
        .main {
            background-color: #1E1E1E;
            color: white;
        }
        .big-text {
            font-size: 120px !important;
            font-weight: bold !important;
            text-align: center !important;
            padding: 20px !important;
            background-color: #2E2E2E;
            border-radius: 10px;
            margin: 20px 0;
        }
        .word-text {
            font-size: 48px !important;
            font-weight: bold !important;
            text-align: center !important;
            padding: 15px !important;
            background-color: #3E3E3E;
            border-radius: 10px;
            margin: 10px 0;
        }
        .stButton>button {
            width: 100%;
            height: 60px;
            font-size: 24px;
            background-color: #2E2E2E;
            color: white;
        }
        .history-text {
            font-size: 24px !important;
            font-weight: normal !important;
            text-align: left !important;
            padding: 10px !important;
            background-color: #2E2E2E;
            border-radius: 10px;
            margin: 10px 0;
        }
        .history-item {
            display: inline-block;
            padding: 5px 10px;
            margin: 3px;
            background-color: #3E3E3E;
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.image("gesture.png", width=400)

    if 'tracking' not in st.session_state:
        st.session_state.tracking = initialize_tracking()

    # Create main layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Live Detection")
        video_placeholder = st.empty()
        
        # Buttons row
        button_col1, button_col2, button_col3 = st.columns(3)
        with button_col1:
            stop_button = st.button("Stop Detection")
        with button_col2:
            space_button = st.button("Add Space")
        with button_col3:
            clear_button = st.button("Clear Word")

    with col2:
        # Latest Detection
        latest_detection = st.empty()
        
        # Word Formation
        st.markdown("### Formed Word")
        word_placeholder = st.empty()
        
        # Detection Metrics
        st.markdown("### Detection Metrics")
        metrics_placeholder = st.empty()
        
        # Detection History
        st.markdown("### Detection History")
        history_placeholder = st.empty()

    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    # Tambahkan placeholder untuk countdown
    countdown_placeholder = st.empty()

    # Handle button actions
    if clear_button:
        # Reset all
        st.session_state.tracking['detection_history'].clear()  
        st.session_state.tracking['frame_predictions'] = []      
        st.session_state.tracking['last_detection'] = None      
        st.session_state.tracking['current_letter'] = None     

    if space_button:
        if len(st.session_state.tracking['detection_history']) < 5:  
            st.session_state.tracking['detection_history'].append(" ")

    while not stop_button:
        current_time = time.time()
        
        # Handle countdown logic
        if st.session_state.tracking['countdown_active']:
            elapsed = current_time - st.session_state.tracking['countdown_start']
            remaining = 3 - int(elapsed)
            
            if remaining > 0:
                # Tampilkan countdown dengan instruksi
                countdown_html = """
                    <div style="
                        position: absolute;
                        top: 50%;
                        left: 50%;
                        transform: translate(-50%, -50%);
                        text-align: center;
                        z-index: 9999;
                    ">
                        <div style="
                            font-size: 120px;
                            font-weight: bold;
                            color: white;
                            background-color: rgba(0,0,0,0.7);
                            padding: 40px;
                            border-radius: 20px;
                            margin-bottom: 20px;
                            animation: fadeInOut 1s infinite;
                        ">
                            {number}
                        </div>
                        <div style="
                            font-size: 24px;
                            color: white;
                            background-color: rgba(0,0,0,0.7);
                            padding: 20px;
                            border-radius: 10px;
                        ">
                            Siapkan isyarat berikutnya...
                        </div>
                    </div>
                    <style>
                        @keyframes fadeInOut {{
                            0% {{ opacity: 0.5; }}
                            50% {{ opacity: 1; }}
                            100% {{ opacity: 0.5; }}
                        }}
                    </style>
                """.format(number=remaining)
                
                countdown_placeholder.markdown(countdown_html, unsafe_allow_html=True)
                st.session_state.tracking['is_detection_paused'] = True  # Pause deteksi
            else:
                # Reset countdown dan mulai deteksi
                st.session_state.tracking['countdown_active'] = False
                st.session_state.tracking['is_detection_paused'] = False
                countdown_placeholder.empty()
                st.session_state.tracking['frame_predictions'] = []  # Reset predictions
                continue

        # Skip detection if paused during countdown
        if st.session_state.tracking['is_detection_paused']:
            # Tetap tampilkan frame kamera tapi skip deteksi
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                video_placeholder.image(frame, channels="BGR")
            continue

        # Normal detection flow
        frame_start_time = time.time()
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()
        H, W, _ = frame.shape()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        prediction_text = "..."
        confidence_score = 0.0

        if results.multi_hand_landmarks:
            st.session_state.tracking['total_detections'] += 1
            
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[1].x
                    y = hand_landmarks.landmark[i].y
                    z = hand_landmarks.landmark[i].z
                    data_aux.append((x,y,z))
                    x_.append(x)
                    y_.append(y)

            x1 = int(min(x_) * W)
            y1 = int(min(y_) * H)

            x2 = int(max(x_) * W)
            y2 = int(max(y_) * H)

            if len(data_aux) == 42:
                prediction = model42.predict([np.asarray(data_aux).flatten()])
                prediction_text = prediction[0]
                confidence_score = 0.95
            elif len(data_aux) == 21:
                prediction = model21.predict([np.asarray(data_aux).flatten()])
                prediction_text = prediction[0]
                confidence_score = 0.90

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 4)
            
            # Simpan prediksi jika valid
            if prediction_text != "No Gesture Detected":
                st.session_state.tracking['frame_predictions'].append(
                    (prediction_text, confidence_score)
                )

        # Cek waktu untuk memproses prediksi
        elapsed_time = current_time - st.session_state.tracking['last_process_time']
        if elapsed_time >= st.session_state.tracking['process_interval']:
            # Proses prediksi yang terkumpul
            best_prediction, confidence = process_predictions(
                st.session_state.tracking['frame_predictions']
            )
            
            if best_prediction:
                # Update tracking
                st.session_state.tracking['detection_history'].append(best_prediction)
                st.session_state.tracking['current_letter'] = best_prediction
                st.session_state.tracking['last_detection'] = best_prediction
                st.session_state.tracking['successful_detections'] += 1
                
                # Aktifkan countdown setelah deteksi berhasil
                st.session_state.tracking['countdown_active'] = True
                st.session_state.tracking['countdown_start'] = current_time
            
            # Reset frame predictions
            st.session_state.tracking['frame_predictions'] = []
            st.session_state.tracking['last_process_time'] = current_time

        # Update UI elements
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB")
        
        # Update Latest Detection - hanya menampilkan huruf saat ini
        current_letter = st.session_state.tracking['current_letter'] or "No Gesture Detected"
        latest_detection.markdown(f"""
            <div class="detection-text">
                {current_letter}
            </div>
        """, unsafe_allow_html=True)
        
        # Update Formed Word
        formed_word = ''.join(list(st.session_state.tracking['detection_history']))
        word_placeholder.markdown(f"""
            <div class="word-text" style="font-size: 36px; font-weight: bold; 
                 text-align: center; padding: 15px; background-color: #2E2E2E; 
                 border-radius: 10px; margin: 10px 0;">
                {formed_word}
            </div>
        """, unsafe_allow_html=True)
        
        # Update Detection History
        history = list(st.session_state.tracking['detection_history'])
        history_string = ' → '.join(history) if history else "No detections yet"
        history_placeholder.markdown(f"""
            <div style="font-size: 18px; padding: 10px; 
                 background-color: #2E2E2E; border-radius: 5px; margin-top: 10px;">
                {history_string}
            </div>
        """, unsafe_allow_html=True)

        # Update metrics display
        metrics_html = """
            <div style="background-color: #2E2E2E; padding: 15px; border-radius: 10px; margin: 10px 0;">
                <div style="font-size: 18px; margin-bottom: 10px;">
                    <strong>Detection Rate:</strong> {:.1f}%
                </div>
                <div style="font-size: 18px; margin-bottom: 10px;">
                    <strong>Average Confidence:</strong> {:.1f}%
                </div>
                <div style="font-size: 18px;">
                    <strong>FPS:</strong> {:.1f}
                </div>
            </div>
        """.format(
            (st.session_state.tracking['successful_detections'] / max(1, st.session_state.tracking['total_detections'])) * 100,
            np.mean(list(st.session_state.tracking['confidence_history'])) * 100 if st.session_state.tracking['confidence_history'] else 0,
            np.mean(list(st.session_state.tracking['fps_history'])) if st.session_state.tracking['fps_history'] else 0
        )
        metrics_placeholder.markdown(metrics_html, unsafe_allow_html=True)

    cap.release()

if __name__ == '__main__':
    main()