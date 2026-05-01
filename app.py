from flask import Flask, render_template, request
import pickle 
import re

app = Flask(__name__)

model = pickle.load(open('fake_news_model.pkl', 'rb'))
vectorizer = pickle.load(open('fake_news_vectorizer.pkl', 'rb'))

def wordopt(text):
    if ' - ' in text[:100]:
        text = text.split(' - ', 1)[1]
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news = request.form['news_text']
        clean_news = wordopt(news)
        vectorized_news = vectorizer.transform([clean_news])
        prediction = model.predict(vectorized_news)

        if prediction[0] == 0:
            result = "Fake News 🚨"
        else:
            result = "True News ✅"

        return render_template('index.html', prediction_text=f'Result: {result}', original_text=news)
    
if __name__ == '__main__':
    app.run(debug=True)