# main.py
import data_collection
import data_preprocessing
import model_training
import model_evaluation

if __name__ == "__main__":
    # 데이터 수집
    data_collection.save_chatgpt_code()
    data_collection.save_human_code('Github_token', 'tensorflow/tensorflow')
    data_collection.merge_human_code()

    # 데이터 전처리
    X_train, X_test, y_train, y_test = data_preprocessing.load_and_preprocess_data('chatgpt_code.txt', 'human_code.txt')
    data_preprocessing.save_preprocessed_data(X_train, X_test, y_train, y_test)

    # 모델 학습
    model, tokenizer, max_sequence_len = model_training.train_model(X_train, y_train, X_test, y_test)

    # 모델 저장
    model.save('chatgpt_human_code_classifier.h5')

    # 사용자로부터 코드 입력 받기
    new_code = input("Enter new code: ")

    # 모델 평가 및 예측
    prediction = model_evaluation.predict_code(model, tokenizer, max_sequence_len, new_code)
    print(f"Prediction (1: ChatGPT, 0: Human): {prediction}")
