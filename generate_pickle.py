import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Read the CSV file and take a sample
df = pd.read_csv("spotify_millsongdata.csv")
df = df.sample(5000).reset_index(drop=True)  # Take a sample of 5000 songs

# Clean the text data
df['text'] = df['text'].str.lower().replace(r"^\W\s", " ").replace(r"\n", " ", regex=True)

# Create a CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(df['text']).toarray()

# Calculate similarity matrix
similarity = cosine_similarity(vectors)

# Save the files
pickle.dump(df, open('df.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb')) 