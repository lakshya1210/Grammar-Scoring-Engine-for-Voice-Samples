import os
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import IPython.display as ipd

# Import modules
from .audio_processing import load_audio, display_audio, extract_audio_features, process_audio_dataset, find_audio_files
from .transcription import transcribe_audio, transcribe_audio_whisper, process_audio_files
from .grammar_analysis import analyze_grammar, get_grammar_features, analyze_transcriptions
from .scoring import calculate_grammar_score, score_samples
from .visualization import plot_score_distribution, plot_error_categories, visualize_results
from .utils import check_kaggle, install_required_packages, convert_audio_format, display_analysis_report

def process_single_audio(audio_file, use_whisper=False):
    """Process a single audio file and return analysis results."""
    # Check if audio file exists
    if not os.path.exists(audio_file):
        print(f"Audio file not found: {audio_file}")
        return None
    
    # Display audio if in notebook
    try:
        display_audio(audio_file)
    except:
        pass
    
    # Extract features
    features = extract_audio_features(audio_file)
    if not features:
        print("Failed to extract audio features")
        return None
    
    # Create DataFrame with single row
    df = pd.DataFrame([features])
    df['audio_path'] = audio_file
    
    # Transcribe
    if use_whisper:
        try:
            import whisper
        except ImportError:
            install_required_packages()
            
    df = process_audio_files(df, audio_column='audio_path', use_whisper=use_whisper)
    
    # Analyze grammar
    df = analyze_transcriptions(df)
    
    # Calculate scores
    df = score_samples(df)
    
    # Create result dictionary
    result = {
        'audio_path': audio_file,
        'transcription': df.at[0, 'transcription'] if not df.empty else None,
        'grammar_score': df.at[0, 'grammar_score'] if not df.empty else 0,
        'error_count': df.at[0, 'error_count'] if not df.empty else 0,
        'error_rate': df.at[0, 'error_rate'] if not df.empty else 0,
        'grammar_features': df.at[0, 'grammar_features'] if not df.empty else {}
    }
    
    # Display report
    display_analysis_report(result)
    
    return result

def complete_grammar_scoring_workflow(dataset_name=None, audio_file=None, use_whisper=False):
    """
    Complete workflow for grammar scoring.
    
    Args:
        dataset_name: Path to dataset folder
        audio_file: Path to single audio file
        use_whisper: Whether to use Whisper for transcription
        
    Returns:
        DataFrame with results or single result dictionary
    """
    print("Running Grammar Scoring Engine...")
    
    # Choose mode based on inputs
    if audio_file:
        print(f"Processing single audio file: {os.path.basename(audio_file)}")
        return process_single_audio(audio_file, use_whisper)
    
    elif dataset_name:
        # Check if running in Kaggle
        if check_kaggle() and not os.path.exists(dataset_name):
            # Try to find dataset in Kaggle inputs
            kaggle_path = f"/kaggle/input/{dataset_name}"
            if os.path.exists(kaggle_path):
                dataset_name = kaggle_path
                print(f"Running on Kaggle, using path: {dataset_name}")
            else:
                print(f"Dataset not found at {kaggle_path}")
                return None
        
        print(f"Processing dataset: {dataset_name}")
        
        # Find audio files
        audio_files = find_audio_files(dataset_name)
        print(f"Found {len(audio_files)} audio files")
        
        if len(audio_files) == 0:
            print("No audio files found")
            return pd.DataFrame()
        
        # Process audio files
        df = process_audio_dataset(audio_files)
        
        # Transcribe audio
        df = process_audio_files(df, use_whisper=use_whisper)
        
        # Analyze grammar
        df = analyze_transcriptions(df)
        
        # Calculate scores
        df = score_samples(df)
        
        # Visualize results
        visualize_results(df)
        
        return df
    
    else:
        print("No dataset or audio file specified")
        return None

def record_and_analyze_audio(duration=5, use_whisper=True):
    """
    Record audio directly and analyze it for grammar.
    
    Note: This function requires additional packages and may not work in all environments.
    
    Args:
        duration: Recording duration in seconds
        use_whisper: Whether to use Whisper for transcription
        
    Returns:
        Analysis results
    """
    try:
        import sounddevice as sd
        import soundfile as sf
        import numpy as np
        from datetime import datetime
        
        # Install sounddevice if needed
        try:
            import sounddevice
        except ImportError:
            import subprocess, sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "sounddevice"])
            import sounddevice as sd
        
        print(f"Recording {duration} seconds of audio... Speak now!")
        
        # Record audio
        sample_rate = 44100
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"recording_{timestamp}.wav"
        sf.write(file_path, recording, sample_rate)
        
        print(f"Recording saved to {file_path}")
        
        # Analyze the audio
        result = complete_grammar_scoring_workflow(audio_file=file_path, use_whisper=use_whisper)
        return result
        
    except Exception as e:
        print(f"Error recording audio: {e}")
        return None