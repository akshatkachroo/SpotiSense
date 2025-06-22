import numpy as np
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import pickle
import torch

def load_emotion_model():
    # Load the pre-trained model and tokenizer
    model_name = "j-hartmann/emotion-english-distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Create the emotion classifier pipeline
    emotion_classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True
    )
    
    return emotion_classifier

def predict_emotion(text):
    # Load the emotion classifier
    emotion_classifier = load_emotion_model()
    
    # Get predictions
    predictions = emotion_classifier(text)[0]
    
    # Convert predictions to PyTorch tensor for processing
    confidence_scores = torch.tensor([pred['score'] for pred in predictions])
    
    # Apply softmax to get normalized probabilities
    probabilities = torch.softmax(confidence_scores, dim=0)
    
    # Get the highest probability emotion
    best_idx = torch.argmax(probabilities).item()
    best_prediction = predictions[best_idx]
    emotion = best_prediction['label']
    confidence = float(probabilities[best_idx].item())
    
    # Map the model's emotions to our categories
    emotion_mapping = {
        'joy': 'happy',
        'sadness': 'sad',
        'anger': 'angry',
        'fear': 'fear',
        'love': 'love',
        'surprise': 'happy',  # Map surprise to happy as it's often positive
        'neutral': 'calm'
    }
    
    mapped_emotion = emotion_mapping.get(emotion, 'calm')
    
    return mapped_emotion, confidence

def get_emotion_based_recommendations(emotion, tracks_df, n_recommendations=5):
    # Define emotion to audio feature mappings
    emotion_features = {
        'happy': {'valence': (0.7, 1.0), 'energy': (0.7, 1.0)},
        'sad': {'valence': (0.0, 0.3), 'energy': (0.0, 0.3)},
        'angry': {'valence': (0.0, 0.3), 'energy': (0.7, 1.0)},
        'calm': {'valence': (0.5, 0.7), 'energy': (0.0, 0.3)},
        'fear': {'valence': (0.0, 0.3), 'energy': (0.3, 0.7)},
        'love': {'valence': (0.7, 1.0), 'energy': (0.3, 0.7)}
    }
    
    # Get feature ranges for the emotion
    feature_ranges = emotion_features.get(emotion, {'valence': (0.4, 0.6), 'energy': (0.4, 0.6)})
    
    # Filter tracks based on emotion features
    filtered_tracks = tracks_df[
        (tracks_df['valence'].between(feature_ranges['valence'][0], feature_ranges['valence'][1])) &
        (tracks_df['energy'].between(feature_ranges['energy'][0], feature_ranges['energy'][1]))
    ]
    
    # Randomly select n_recommendations tracks
    if len(filtered_tracks) >= n_recommendations:
        recommendations = filtered_tracks.sample(n=n_recommendations)
    else:
        recommendations = filtered_tracks.sample(n=min(len(filtered_tracks), n_recommendations))
    
    return recommendations[['name', 'artists']].values.tolist()

if __name__ == "__main__":
    # Test the emotion detection
    test_texts = [
        "I am feeling really happy today, everything is going well",
        "I'm a bit sad because I miss my friends",
        "I'm feeling calm and peaceful after meditation",
        "I'm so in love with this new song",
        "I'm a bit anxious about the upcoming exam"
    ]
    
    for text in test_texts:
        emotion, confidence = predict_emotion(text)
        print(f"Text: {text}")
        print(f"Detected emotion: {emotion} (confidence: {confidence:.2%})")
        print() 
