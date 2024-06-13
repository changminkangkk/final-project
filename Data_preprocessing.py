# data_preprocessing.py
import re
import nltk
from sklearn.model_selection import train_test_split

nltk.download('punkt')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    tokens = nltk.word_tokenize(text)
    return ' '.join(tokens)

def load_and_preprocess_data(chatgpt_code_file, human_code_file):
    with open(chatgpt_code_file, 'r', encoding='utf-8') as file:
        chatgpt_code = file.read().strip().split('\n\n')

    with open(human_code_file, 'r', encoding='utf-8') as file:
        human_code = file.read().strip().split('\n\n')

    data = chatgpt_code + human_code
    labels = [1] * len(chatgpt_code) + [0] * len(human_code)

    preprocessed_data = [preprocess_text(code) for code in data]

    # 데이터를 분할하여 반환
    X_train, X_test, y_train, y_test = train_test_split(preprocessed_data, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def save_preprocessed_data(X_train, X_test, y_train, y_test):
    with open('train_data.txt', 'w', encoding='utf-8') as file:
        for x, y in zip(X_train, y_train):
            file.write(f'__label__{y} {x}\n')

    with open('test_data.txt', 'w', encoding='utf-8') as file:
        for x, y in zip(X_test, y_test):
            file.write(f'__label__{y} {x}\n')
