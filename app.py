from flask import Flask, render_template, request, redirect, url_for, flash, session
import numpy as np
from deep_translator import GoogleTranslator
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib

# Load models
model_lang = joblib.load('Language_model')
model_sequence = joblib.load('Sequence_model')
cv = joblib.load('vectorizer.pkl')  # Assuming you have a vectorizer
tokenizer = joblib.load('tokenizer.pkl')  # Assuming you have a tokenizer
label_encoder = joblib.load('label_encoder.pkl')  # Assuming you have a label encoder
max_length = 66  # Assuming max length of sequences

app = Flask(__name__)
app.secret_key = 'your_secret'

users = {
    "user1": "Jyoti@2002",
    "user2": "Bushra"
}

def translate_to_english(text, target_language):
    try:
        translated_text = GoogleTranslator(source=target_language, target='en').translate(text)
    except Exception as e:
        print("Translation failed:", e)
        translated_text = text
    return translated_text

@app.route('/')
def home():
    if 'logged_in' in session:
        return redirect(url_for('predict_sentiment'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username in users and users[username] == password:
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('predict_sentiment'))
        else:
            flash("Invalid username or password", 'error')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/predict_sentiment', methods=['GET', 'POST'])
def predict_sentiment():
    if 'logged_in' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        user_input = request.form['user_input']
        data = cv.transform([user_input]).toarray()
        pred_lang = model_lang.predict(data)
        print("Detected Language:", pred_lang)

        if user_input:
            detected_lang = pred_lang[0]

            if detected_lang is not None:
                if detected_lang != 'en':
                    modified_input = translate_to_english(user_input, detected_lang)
                    print("Modified English Text:", modified_input)
                else:
                    modified_input = user_input
            else:
                print("Language detection failed. Assuming English.")
                modified_input = user_input

            data = cv.transform([modified_input]).toarray()
            input_sequence = tokenizer.texts_to_sequences([modified_input])
            padded_input_sequence = pad_sequences(input_sequence, maxlen=max_length)
            prediction = model_sequence.predict(padded_input_sequence)
            predicted_label = label_encoder.inverse_transform([np.argmax(prediction[0])])
            print("Predicted Sentiment:", predicted_label)
        
        return render_template('predict_sentiment.html', user_input=user_input, detected_lang=detected_lang, 
                               modified_input=modified_input, predicted_label=predicted_label[0])
    return render_template('predict_sentiment.html')

if __name__ == '__main__':
    app.run(debug=True)