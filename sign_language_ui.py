import streamlit as st
import cv2
import mediapipe as mp
import pickle
import numpy as np
import time

def detect_bisindo_letter(frame, hands, model21, model42):
    data_aux = []
    x_ = []
    y_ = []
    
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(frame_rgb)
    prediction = None
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

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
            prediction = model42.predict([np.asarray(data_aux).flatten()])[0]
        elif len(data_aux) == 21:
            prediction = model21.predict([np.asarray(data_aux).flatten()])[0]
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 4)
        if prediction:
            cv2.putText(frame, prediction, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,0,255), 3,
                        cv2.LINE_AA)
    
    return frame, prediction

def main():
    # Konfigurasi halaman
    st.set_page_config(
        page_title="Penerjemah BISINDO",
        layout="centered",  # Menggunakan layout centered
        initial_sidebar_state="expanded"
    )

    # Custom CSS untuk styling
    st.markdown("""
        <style>
        .main > div {
            padding-top: 2rem;
        }
        .stButton button {
            width: 100%;
            margin-top: 1rem;
        }
        .translation-result {
            margin-top: 1rem;
            padding: 1rem;
            background-color: #1E1E1E;
            border-radius: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Judul utama di tengah
    st.markdown("<h1 style='text-align: center;'>Penerjemah BISINDO</h1>", unsafe_allow_html=True)
    
    mode = "Video"

    if mode == "Video":
        # Container untuk kamera
        st.subheader("Kamera")
        camera_placeholder = st.empty()
        
        # Container untuk hasil
        result_container = st.container()
        
        # Tombol kontrol
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            start_button = st.button("Mulai Terjemahan", use_container_width=True)

        if start_button:
            cap = cv2.VideoCapture(0)
            translated_text = ""
            
            try:
                frame_count = 0  # Untuk key unik
                stop_button = False
                
                while not stop_button:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Tidak dapat mengakses kamera.")
                        break
                    
                    # Deteksi dan proses frame
                    processed_frame, letter = detect_bisindo_letter(frame, hands, model21, model42)
                    
                    # Tampilkan frame yang sudah diproses
                    frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    camera_placeholder.image(
                        frame_rgb, 
                        channels="RGB",
                        use_container_width=True
                    )
                    
                    # Update terjemahan jika ada deteksi baru
                    if letter:
                        translated_text += letter
                    
                    with result_container:
                        st.markdown(f"""
                            <div class='translation-result'>
                                <h4>Huruf Terdeteksi: {letter if letter else 'Tidak ada deteksi'}</h4>
                                <h4>Terjemahan:</h4>
                                <p style='font-size: 1.2em;'>{translated_text}</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Tombol stop dengan key unik
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if st.button("Stop", key=f"stop_button_{frame_count}", use_container_width=True):
                            stop_button = True
                            break
                    
                    frame_count += 1
                    time.sleep(0.1)
                            
            finally:
                cap.release()

    else:  # Mode Teks
        st.subheader("Terjemahkan Teks ke BISINDO")
        input_text = st.text_input(
            "Masukkan teks untuk diterjemahkan:",
            "",
            key="text_input"
        ).upper()
        
        if input_text:
            st.markdown("""
                <div class='translation-result'>
                    <h4>Visualisasi BISINDO:</h4>
                    <p style='font-size: 1.2em;'>{}</p>
                </div>
            """.format(input_text), unsafe_allow_html=True)

if __name__ == "__main__":
    main()