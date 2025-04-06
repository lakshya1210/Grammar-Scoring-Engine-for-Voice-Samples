from grammar_scoring.main import complete_grammar_scoring_workflow

# Example 1: Process a single audio file
result = complete_grammar_scoring_workflow(
    audio_file="path/to/your/audio.wav",
    use_whisper=True  # Set to True for better transcription quality
)

# Example 2: Process a directory of audio files
results_df = complete_grammar_scoring_workflow(
    dataset_name="path/to/your/audio/directory",
    use_whisper=False  # Set to True for better but slower transcription
)

# Print some results
if results_df is not None:
    print(f"Average Grammar Score: {results_df['grammar_score'].mean():.2f}/100")

