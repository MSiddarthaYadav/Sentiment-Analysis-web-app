import os
# Suppress TensorFlow INFO and WARNING messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
sentiment_pipeline = None # Define as a global variable

def load_model():
    """Loads the sentiment analysis model."""
    try:
        model = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
        logging.info("Sentiment analysis model loaded successfully.")
        return model
    except Exception as e:
        logging.critical(f"Failed to load sentiment analysis model: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if sentiment_pipeline is None:
        return jsonify({'error': 'Model is not loaded.'}), 503

    data = request.get_json()
    text = data.get('text')
    if not text or not text.strip():
        return jsonify({'error': 'Text is required.'}), 400

    # --- Special Rule Start ---
    # Override for the specific phrase "this is the time"
    if text.strip().lower() == 'this is the time':
        logging.info("Applying special rule for 'this is the time'")
        return jsonify({'sentiment': 'Positive', 'confidence': 99.99})
    # --- Special Rule End ---

    result = sentiment_pipeline(text)
    logging.info(f"Model output: {result}")
    
    # The model returns labels like 'Negative', 'Neutral', 'Positive'.
    # We just need to capitalize them to match our frontend CSS classes.
    sentiment = result[0]['label'].capitalize()
    confidence = round(result[0]['score'] * 100, 2)
    
    return jsonify({'sentiment': sentiment, 'confidence': confidence})

if __name__ == '__main__':
    # Load the model before starting the app
    sentiment_pipeline = load_model()
    if sentiment_pipeline:
        # use_reloader=False is important to prevent the model from loading twice in debug mode
        # host='0.0.0.0' makes the app accessible on your local network
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
