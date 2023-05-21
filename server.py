from flask import Flask, request, jsonify
import joblib
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)


model = joblib.load('spam_model.pkl')


vectorizer = joblib.load('spam_vectorizer.pkl')

@app.route('/predict/spam', methods=['POST'])
def predict_spam():
   
    message = request.json['message']

    
    message_vectorized = vectorizer.transform([message])

    
    prediction = model.predict(message_vectorized)[0]

    
    response = {
        'prediction': prediction
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
