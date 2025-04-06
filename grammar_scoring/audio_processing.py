import librosa
import librosa.display
import matplotlib.pyplot as plt
import IPython.display as ipd
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
import os

def load_audio(file_path):
    """Load an audio file and return the waveform and sample rate."""
    try:
        waveform, sample_rate = librosa.load(file_path, sr=None)
        return waveform, sample_rate
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return None, None

def display_audio(file_path):
    """Display audio waveform and create an interactive player."""
    waveform, sample_rate = load_audio(file_path)
    if waveform is None:
        return
        
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(waveform, sr=sample_rate)
    plt.title('Audio Waveform')
    plt.tight_layout()
    plt.show()
    
    try:
        ipd.display(ipd.Audio(file_path))
    except Exception as e:
        print(f"Error playing audio: {e}")

def extract_audio_features(file_path):
    """Extract acoustic features from an audio file."""
    waveform, sample_rate = load_audio(file_path)
    if waveform is None:
        return {}
        
    features = {}
    
    # Basic features
    features['duration'] = librosa.get_duration(y=waveform, sr=sample_rate)
    
    # Spectral features
    features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=waveform, sr=sample_rate)[0])
    features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=waveform, sr=sample_rate)[0])
    features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=waveform, sr=sample_rate)[0])
    
    # Rhythm features
    tempo, _ = librosa.beat.beat_track(y=waveform, sr=sample_rate)
    features['tempo'] = tempo
    
    # MFCC features
    mfccs = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=13)
    for i, mfcc in enumerate(mfccs):
        features[f'mfcc_{i}'] = np.mean(mfcc)
    
    return features

def process_audio_dataset(audio_files):
    """Process multiple audio files and extract features."""
    results = []
    
    for file_path in tqdm(audio_files, desc="Processing audio files"):
        features = extract_audio_features(file_path)
        if features:
            features['audio_path'] = file_path
            results.append(features)
    
    return pd.DataFrame(results)

def find_audio_files(directory):
    """Find all audio files in a directory."""
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    audio_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                audio_files.append(os.path.join(root, file))
    
    return audio_files