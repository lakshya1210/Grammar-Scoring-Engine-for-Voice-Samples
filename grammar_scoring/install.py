import subprocess
import sys
import os

def check_kaggle():
    return os.path.exists('/kaggle/input')

def install_dependencies():
    print("Installing dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    except Exception as e:
        print(f"Error installing main requirements: {e}")
    
    import nltk
    print("Downloading NLTK resources...")
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    
    print("Downloading spaCy model...")
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    except Exception as e:
        print(f"Error downloading spaCy model: {e}")
        if check_kaggle():
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0-py3-none-any.whl"])
            except Exception as e2:
                print(f"Error with alternative spaCy install: {e2}")

def setup_environment():
    """Set up the complete environment"""
    print("Setting up Grammar Scoring Engine environment...")
    
    install_dependencies()
    
    if check_kaggle():
        print("Running on Kaggle - environment setup complete")
    else:
        print("Not running on Kaggle - environment setup complete")
    
    print("\nAll set! You can now run the Grammar Scoring Engine.")

if __name__ == "__main__":
    setup_environment()