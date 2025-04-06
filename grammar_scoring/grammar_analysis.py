import language_tool_python
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy
import pandas as pd
from tqdm.notebook import tqdm
import os

# Initialize tools
try:
    tool = language_tool_python.LanguageTool('en-US')
except:
    tool = None
    print("Warning: LanguageTool could not be initialized. Grammar checking may be limited.")

try:
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = None
    print("Warning: spaCy model could not be loaded.")

def analyze_grammar(text):
    """Analyze grammar using LanguageTool."""
    if not text or not tool:
        return {
            'text': text or '',
            'error_count': 0,
            'errors': [],
            'error_categories': {},
            'error_rate': 0
        }
    
    matches = tool.check(text)
    
    word_count = len(word_tokenize(text))
    
    error_categories = {}
    for match in matches:
        category = match.category
        if category in error_categories:
            error_categories[category] += 1
        else:
            error_categories[category] = 1
    
    return {
        'text': text,
        'error_count': len(matches),
        'errors': matches,
        'error_categories': error_categories,
        'error_rate': len(matches) / max(word_count, 1)  
    }

def get_grammar_features(text):
    """Extract linguistic features from text."""
    if not text:
        return {}
    
    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    word_count = len(words)
    sentence_count = len(sentences)
    avg_sentence_length = word_count / max(sentence_count, 1)
    
    # POS tagging
    pos_tags = nltk.pos_tag(words)
    pos_counts = {}
    for _, tag in pos_tags:
        if tag in pos_counts:
            pos_counts[tag] += 1
        else:
            pos_counts[tag] = 1
    
    pos_ratios = {f'{pos}_ratio': count / max(word_count, 1) for pos, count in pos_counts.items()}
    
    # Grammar analysis
    grammar_analysis = analyze_grammar(text)
    error_rate = grammar_analysis['error_rate']
    
    features = {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'avg_sentence_length': avg_sentence_length,
        'error_rate': error_rate,
        'error_count': grammar_analysis['error_count']
    }
    
    features.update(pos_ratios)
    
    # Optional spaCy analysis
    if nlp and text:
        try:
            doc = nlp(text)
            
            dep_counts = {}
            for token in doc:
                if token.dep_ in dep_counts:
                    dep_counts[token.dep_] += 1
                else:
                    dep_counts[token.dep_] = 1
            
            dep_ratios = {f'{dep}_ratio': count / max(word_count, 1) 
                          for dep, count in dep_counts.items()}
            
            features.update(dep_ratios)
        except Exception as e:
            print(f"Error in spaCy analysis: {e}")
    
    return features

def analyze_transcriptions(df, text_column='transcription'):
    """Analyze grammar for multiple transcriptions."""
    if 'transcription' not in df.columns:
        print("No transcription column found")
        return df
    
    results = df.copy()
    
    # Initialize columns
    results['grammar_features'] = None
    results['error_count'] = 0
    results['error_rate'] = 0.0
    
    for idx, row in tqdm(results.iterrows(), total=len(results), desc="Analyzing grammar"):
        text = row[text_column]
        
        if pd.isna(text) or text == '':
            continue
            
        # Get grammar features
        features = get_grammar_features(text)
        
        # Update DataFrame
        results.at[idx, 'grammar_features'] = features
        results.at[idx, 'error_count'] = features.get('error_count', 0)
        results.at[idx, 'error_rate'] = features.get('error_rate', 0.0)
    
    return results