import streamlit as st
import torch
import numpy as np
import librosa
import os
import torch.nn as nn

class DeepFakeAudioCNN(nn.Module):
    def __init__(self):
        super(DeepFakeAudioCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2, 2), 
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2), 
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2), 
            nn.Dropout(0.3)      
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(21504, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 1), nn.Sigmoid() 
        )
    def forward(self, x):
        return self.classifier(self.features(x))

def process_audio_to_spectrogram(file_path, target_sr=22050, duration=4):
    # librosa.load seamlessly uses soundfile behind the scenes to decode mp3/m4a/wav into standard arrays
    y, sr = librosa.load(file_path, sr=target_sr, duration=duration)
    target_length = target_sr * duration
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)), mode='constant')
    else:
        y = y[:target_length]
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    return (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)

st.set_page_config(page_title="AI Audio Detector", page_icon="🛡️")
st.title("🛡️ Regional Language Audio Deepfake Detector")
st.write("Upload a regional voice note below to verify if it is an authentic human or an AI voice clone.")

#  MULTI-FORMAT SUPPORT TRIGGERED HERE:
uploaded_file = st.file_uploader(
    "Upload Audio Note (Voice Message, Call Recording, etc.)", 
    type=["wav", "mp3", "m4a", "aac", "ogg"]
)

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    with st.spinner("Analyzing audio micro-textures..."):
        spec = process_audio_to_spectrogram(uploaded_file)
        spec_tensor = torch.tensor(spec).unsqueeze(0).unsqueeze(0).float()

        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "regional_detector.pth")

        model = DeepFakeAudioCNN()
        model.load_state_dict(torch.load(model_path))
        model.eval()

        with torch.no_grad():
            prediction = model(spec_tensor).item()

        confidence = prediction if prediction > 0.5 else (1 - prediction)
        percentage = f"{confidence * 100:.2f}%"

        st.markdown("---")
        if prediction > 0.5:
            st.error(" **Deepfake Variant Detected!**")
            st.metric(label="AI Probability Clone Score", value=percentage)
        else:
            st.success(" **Authentic Human Voice Verified**")
            st.metric(label="Human Consistency Score", value=percentage)
