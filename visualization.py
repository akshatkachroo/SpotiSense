import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

def create_emotion_radar_chart(song_features, emotion):
    """
    Create a radar chart showing the audio features of a song.
    """
    # Features to display
    features = ['valence', 'energy', 'acousticness', 'danceability', 'instrumentalness']
    
    # Get feature values
    values = [song_features[feature] for feature in features]
    
    # Number of features
    N = len(features)
    
    # Create angles for the radar chart
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Add the first value again to close the loop
    values += values[:1]
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
    
    # Plot the data
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, alpha=0.25)
    
    # Set the labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features)
    
    # Set the title
    plt.title(f'Song Features for {emotion.capitalize()} Mood', pad=20)
    
    # Convert plot to base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close()
    
    return img_str

def get_song_features(tracks_df, song_name, artist):
    """
    Get the audio features for a specific song.
    """
    song = tracks_df[
        (tracks_df['name'] == song_name) & 
        (tracks_df['artists'].str.contains(artist, case=False, na=False))
    ]
    
    if len(song) == 0:
        return None
    
    return song.iloc[0]

def visualize_recommendations(recommendations, tracks_df, emotion):
    """
    Create visualizations for the recommended songs.
    """
    visualizations = []
    
    for song_name, artist in recommendations:
        features = get_song_features(tracks_df, song_name, artist)
        if features is not None:
            img_str = create_emotion_radar_chart(features, emotion)
            visualizations.append({
                'song_name': song_name,
                'artist': artist,
                'visualization': img_str
            })
    
    return visualizations 