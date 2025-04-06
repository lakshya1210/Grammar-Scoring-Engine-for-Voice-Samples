
import os
import subprocess
import sys
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import soundfile as sf

def check_kaggle():
    """Check if running in Kaggle environment."""
    return os.path.exists('/kaggle/input')

def install_required_packages():
    """Install required packages if not already installed."""
    try:
        import whisper
    except ImportError:
        print("Installing whisper for improved transcription...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openai-whisper"])

def convert_audio_format(input_file, output_file=None):
    """Convert audio to a compatible format for transcription."""
    if output_file is None:
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = f"converted_{base_name}.wav"
    
    try:
        # Load the audio file with librosa
        y, sr = librosa.load(input_file, sr=None)
        
        # Save as WAV file
        sf.write(output_file, y, sr)
        print(f"Successfully converted {input_file} to {output_file}")
        return output_file
    except Exception as e:
        print(f"Error converting audio: {e}")
        return None

def display_analysis_report(result):
    """Display a formatted analysis report."""
    transcription = result.get('transcription', 'N/A')
    grammar_score = result.get('grammar_score', 0)
    error_count = result.get('error_count', 0)
    
    print("\n===== Grammar Analysis Report =====")
    print(f"Transcription: {transcription[:100]}{'...' if len(str(transcription)) > 100 else ''}")
    print(f"Grammar Score: {grammar_score:.2f}/100")
    print(f"Error Count: {error_count}")
    print("\n===================================")