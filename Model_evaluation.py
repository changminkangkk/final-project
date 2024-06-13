# model_evaluation.py
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import re
import nltk

from nltk.tokenize import word_tokenize

# 처음 한 번만 다운로드하도록 합니다.
nltk.download('punkt')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    tokens = word_tokenize(text)  # nltk.tokenize.word_tokenize 사용
    return ' '.join(tokens)

def predict_code(model, tokenizer, max_sequence_len, new_code):
    new_code_processed = preprocess_text(new_code)
    new_code_seq = tokenizer.texts_to_sequences([new_code_processed])
    new_code_padded = pad_sequences(new_code_seq, maxlen=max_sequence_len, padding='post')

    prediction = model.predict(new_code_padded)
    return prediction[0][0]
