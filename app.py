from flask import Flask, render_template, request, jsonify
from utils import predict_class
import pickle
import keras

model = keras.models.load_model('model.h5')
with open('tokenizer.pickle', 'rb') as f:
    # Load the object from the pickle file
    tokenizer = pickle.load(f)

app = Flask(__name__)

import traceback

@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error(traceback.format_exc())
    return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return render_template('result.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    tweet_json = request.get_json()
    if not tweet_json or 'tweet' not in tweet_json:
        return jsonify({'error': 'Missing tweet'}), 400
    text = tweet_json['tweet']
    text=[text]
    text_type = type(text)
    sentiment = predict_class(text, model, tokenizer)
    print(f'Text: {text}')
    print(f'Text_type: {text_type}')
    print(f'Sentiment: {sentiment}')
    return jsonify({'sentiment': sentiment}), 200
 
if __name__ == '__main__':
    app.run(host='localhost' , port=5000, debug=True)