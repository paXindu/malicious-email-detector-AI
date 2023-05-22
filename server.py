import re
import nltk
from nltk.corpus import stopwords
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)
CORS(app)

model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('spam_vectorizer.pkl')

nltk.download('stopwords')

def preprocess_message(message):
    
    message = re.sub(r'http\S+|www.\S+', '', message)
    
    
    message = re.sub(r'\S+@\S+', '', message)
    
    
    message = re.sub(r'[^\w\s]', '', message)
    
    
    message = message.lower()
    
    
    stop_words = set(stopwords.words('english'))
    message = ' '.join(word for word in message.split() if word not in stop_words)
    
    return message

@app.route('/predict/spam', methods=['POST'])
def predict_spam():
    message = request.data.decode('utf-8')
    
    message = preprocess_message(message)
    
    message_vectorized = vectorizer.transform([message])
    prediction_proba = model.predict_proba(message_vectorized)[0]
    prediction = model.predict(message_vectorized)[0]
    
    response = {
        'prediction': prediction,
        'prediction_percentage': max(prediction_proba) * 100
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
