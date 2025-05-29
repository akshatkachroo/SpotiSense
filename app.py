import pickle
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
from dotenv import load_dotenv
import pandas as pd
from emotion_model import predict_emotion, get_emotion_based_recommendations
from visualization import visualize_recommendations

load_dotenv()

CLIENT_ID = os.environ.get("CLIENT_ID")
CLIENT_SECRET = os.environ.get("CLIENT_SECRET")

# Initialize Spotify API client
client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def get_song_album_cover_url(song_name, artist_name):
    search_query = f"track: {song_name} artist : {artist_name}"
    results = sp.search(q=search_query, type="track")

    if results and results["tracks"]["items"]:
        track = results["tracks"]["items"][0]
        album_cover_url = track["album"]["images"][0]["url"]
        print(album_cover_url)
        return album_cover_url
    else:
        return "https://i.pcmag.com/imagery/articles/01uUtdvBJd0VNaZlw5H1Qid-1..v1723632784.jpg"

def recommend(song):
    index = music[music['song'] == song].index[0]
    distances = sorted(list(enumerate(similarity[index])), key=lambda x: x[1], reverse=True)
    recommended_music_names = []
    recommended_music_posters = []
    for i in distances[1:6]:
        # fetch the movie poster
        artist = music.iloc[i[0]].artist
        print(artist)
        print(music.iloc[i[0]].song)
        recommended_music_posters.append(get_song_album_cover_url(music.iloc[i[0]].song, artist))
        recommended_music_names.append(music.iloc[i[0]].song)
    return recommended_music_names, recommended_music_posters

# Load data
st.header("SpotiSense - Music Recommender")
music = pickle.load(open("df.pkl", "rb"))
similarity = pickle.load(open("similarity.pkl", "rb"))
tracks_df = pd.read_csv("tracks_features.csv")

# Create tabs for different recommendation methods
tab1, tab2 = st.tabs(["Song-based Recommendations", "Emotion-based Recommendations"])

with tab1:
    st.subheader("Get recommendations based on a song")
    music_list = music['song'].values
    selected_movie = st.selectbox(
        "Type or select a song from the dropdown",
        music_list
    )

    if st.button("Show Song-based Recommendations"):
        recommended_music_names, recommended_music_posters = recommend(selected_movie)
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.text(recommended_music_names[0])
            st.image(recommended_music_posters[0])
        with col2:
            st.text(recommended_music_names[1])
            st.image(recommended_music_posters[1])
        with col3:
            st.text(recommended_music_names[2])
            st.image(recommended_music_posters[2])
        with col4:
            st.text(recommended_music_names[3])
            st.image(recommended_music_posters[3])
        with col5:
            st.text(recommended_music_names[4])
            st.image(recommended_music_posters[4])

with tab2:
    st.subheader("Get recommendations based on your emotions")
    user_text = st.text_area("How are you feeling today? Describe your emotions:", height=100)
    
    if st.button("Get Emotion-based Recommendations"):
        if user_text:
            # Predict emotion
            emotion, confidence = predict_emotion(user_text)
            
            # Display emotion and confidence
            st.write(f"Detected Emotion: {emotion.capitalize()}")
            st.write(f"Confidence: {confidence:.2%}")
            
            # Get recommendations
            recommendations = get_emotion_based_recommendations(emotion, tracks_df)
            
            # Get visualizations
            visualizations = visualize_recommendations(recommendations, tracks_df, emotion)
            
            # Display recommendations with visualizations
            st.subheader("Recommended Songs for Your Mood:")
            
            for i, viz in enumerate(visualizations):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.text(f"{viz['song_name']}\nby {viz['artist']}")
                    st.image(get_song_album_cover_url(viz['song_name'], viz['artist']))
                with col2:
                    st.image(f"data:image/png;base64,{viz['visualization']}")
                st.markdown("---")
        else:
            st.warning("Please enter some text to describe your emotions.")