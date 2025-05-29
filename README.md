# SpotiSense - Music Recommendation System

SpotiSense is an intelligent music recommendation system that leverages machine learning to provide personalized song suggestions based on user preferences and emotional context. The system analyzes various musical features and emotional patterns to deliver accurate and meaningful recommendations.

## Features

- Emotion-based music recommendations
- Interactive web interface
- Data visualization capabilities
- Machine learning-powered song analysis
- Support for large-scale Spotify datasets

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment (recommended)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SpotiSense.git
cd SpotiSense
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Setup

The project requires two datasets from Kaggle:

1. Spotify Million Song Dataset
   - Download from: [Spotify Million Song Dataset](https://www.kaggle.com/datasets/notshrirang/spotify-million-song-dataset)
   - Place the dataset in the project root directory

2. Spotify 12M Songs Dataset
   - Download from: [Spotify 12M Songs](https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs)
   - Place the dataset in the project root directory

## Project Structure

- `app.py` - Main application file
- `emotion_model.py` - Emotion analysis model implementation
- `visualization.py` - Data visualization utilities
- `generate_pickle.py` - Data preprocessing script
- `notebook.ipynb` - Jupyter notebook for analysis and development
- `requirements.txt` - Project dependencies

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:5000`

3. Use the web interface to:
   - Search for songs
   - Get emotion-based recommendations
   - View data visualizations

## Model Files

The following pre-trained model files are included:
- `emotion_model.pkl` - Trained emotion analysis model
- `vectorizer.pkl` - Text vectorizer for feature extraction
- `label_encoder.pkl` - Label encoder for categorical variables
- `df.pkl` - Processed dataset

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Spotify for providing the API and datasets
- Kaggle for hosting the datasets
- All contributors and users of the project
