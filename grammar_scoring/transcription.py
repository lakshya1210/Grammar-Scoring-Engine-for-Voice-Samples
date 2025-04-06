import speech_recognition as sr
import pandas as pd
from tqdm.notebook import tqdm
import os

def transcribe_audio(audio_path):
    """Transcribe audio using Google Speech Recognition."""
    recognizer = sr.Recognizer()
    
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return text
    except Exception as e:
        print(f"Error transcribing audio {os.path.basename(audio_path)}: {e}")
        return None

def transcribe_audio_whisper(audio_path):
    """Transcribe audio using OpenAI's Whisper (if available)."""
    try:
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        print(f"Error transcribing with whisper: {e}")
        return None

def process_audio_files(df, audio_column='audio_path', transcribe=True, use_whisper=False):
    """Process multiple audio files from a DataFrame."""
    results = df.copy()
    
    if 'transcription' not in results.columns:
        results['transcription'] = None
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Transcribing audio"):
        audio_path = row[audio_column]
        
        if pd.notna(results.at[idx, 'transcription']) and not transcribe:
            continue
            
        if use_whisper:
            transcription = transcribe_audio_whisper(audio_path)
        else:
            transcription = transcribe_audio(audio_path)
            
        if transcription:
            results.at[idx, 'transcription'] = transcription
    
    transcribed_count = results['transcription'].notna().sum()
    print(f"Successfully transcribed {transcribed_count} of {len(results)} audio files")
    
    return results