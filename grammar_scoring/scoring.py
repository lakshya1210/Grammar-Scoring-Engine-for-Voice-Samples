import numpy as np
import pandas as pd

def calculate_grammar_score(error_rate, text_length=0, min_score=0):
    """Calculate a grammar score from 0-100 based on error rate."""
    # Base calculation (inverse relationship with error rate)
    base_score = 100 * (1 - min(error_rate, 1))
    
    # Text length adjustment factor
    length_factor = min(1.0, text_length / 50)  # Normalize for very short texts
    
    # Apply length penalization for very short texts
    final_score = base_score * (0.5 + 0.5 * length_factor)
    
    # Ensure score is within bounds
    return max(min_score, min(100, final_score))

def score_samples(df):
    """Calculate grammar scores for samples."""
    results = df.copy()
    
    if 'error_rate' not in results.columns or 'transcription' not in results.columns:
        print("Required columns missing")
        return results
    
    # Calculate text lengths
    results['text_length'] = results['transcription'].fillna('').apply(len)
    
    # Calculate grammar scores
    results['grammar_score'] = results.apply(
        lambda row: calculate_grammar_score(
            row['error_rate'], 
            row['text_length']
        ) if pd.notna(row['transcription']) else 0,
        axis=1
    )
    
    # Round scores for display
    results['grammar_score'] = results['grammar_score'].round(2)
    
    return results