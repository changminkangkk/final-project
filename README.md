# final-project
 어떤 코드를 입력받았을 때, 이 코드가 chatgpt가 짠 코드인지 사람이 짠 코드인지 예측하는 코드를 만들었습니다.

 어떤 프로젝트를 할 지와 데이터를 어떤 식으로 받아올 지에 대해서만 저의 아이디어를 사용하고, 나머지는 거의 다 chatgpt의 도움을 받았습니다.

 그리고 실제로 코드를 실행하는 단계에서는 컴퓨터 사양으로 인하여 학습하는 파일의 숫자를 30개, epoch=1로 설정하여 진행하였고, 학습데이터로 사용된 chatgpt의 코드의 개수가 충분히 많지 않아서 예측능력이 굉장이 떨어졌습니다.

Data_collectiong.py

save_Human_code
Github에서 제공받은 개인의 Github token을 이용해서 'tensorflow/tensorflow' repository에 접근한 후에, max_files 개수만큼의 파일을 리스트에 저장합니다.
그냥 호출을 하는 경우에는 너무 많은 요청을 보내서 오류가 발생하는 상황이 생겨서 호출 후에 1초간 대기하여 요청 속도를 조절했습니다. 이러한 방법으로 사람이 만든 코드를 저장했습니다.

save_chatgpt_code
우선, chatgpt가 만든 코드를 얻기 위해서 Leetcode 문제의 Hard 난이도 문제들에 대한 코드를 생성했습니다.(최대한 긴 코드들을 학습시키기 위해서 hard문제에 대한 풀이 코드들을 선정했습니다.)
그리고 그 코드들을 리스트에 저장했습니다. 그리고 리스트에 있는 코드들을 txt 파일로 저장하는 코드를 통하여 txt 파일로 chaptgpt가 만든 코드를 txt 파일로 저장하였습니다.


Data_preprocessing.py

'preprocess_text' 함수에서 텍스트를 소문자로 변환하고, 공백을 정리하는 과정을 거칩니다. 이 과정은 텍스트 데이터를 정리하고 일관성있게 만들어 머신러닝 모델의 성능을 높이기 위해서 필요합니다.

'load_and_preprocess_data' 함수에서는 두 개의 코드 파일 'chatgpt_code_file', 'human_code_file'을 읽고, 데이터를 전처리한 후에 훈련 세트와 테스트 세트로 분할합니다.

'save_prerpocessed_data' 함수를 통하여 전처리된 데이터를 훈련 데이터와 테스트 데이터 파일에 저장했습니다.


Model_training.py

'tran_model' 함수에서는 텍스트 데이터를 머신 러닝 모델에 입렫될 수 있는 형태로 변형하기 위해서 Tokenizer()함수를 사용합니다. 그리고 데이터를 손실 함수와 최적화 알고리즘을 설정하여 모델을 구축한 후에, 데이터를 모델에 입력될 수 있도록 넘파이 배열로 변환하였습니다. model.fit()함수를 사용하여 훈련 데이터와 검증 데이터를 사용하며 지정한 epoch 수만큼 전체 데이터를 반복하여 학습합니다.


Model_evaluation.py

'predict_code' 함수에서는 훈련된 딥러닝 모델과 tokenizer, 최대 스퀀스 길이, 예측할 새로운 코드 문자열을 입력 받은 후에, GPT가 짜준 코드에 가깝다고 생각되면, 1에 가까운 값을 사람이 짠 코드에 가깝다고 생각되면 0에 가까운 float가 출력합니다.

결과

1) 사람이 만든 코드 입력 prediction:0.0031438772566616535
Enter new code: print("Hello World")
1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 660ms/step
Prediction (1: ChatGPT, 0: Human): 0.0031438772566616535


2) Chatgpt가 만든 코드 입력 prediction:0.019221099093556404
Enter new code: def isMatch(s: str, p: str) -> bool:
    m, n = len(s), len(p)
    
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    
    for j in range(1, n + 1):
        if p[j - 1] == '*':
            dp[0][j] = dp[0][j - 2]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == '*' and j > 1:
                dp[i][j] = dp[i][j - 2] or (dp[i - 1][j] and (s[i - 1] == p[j - 2] or p[j - 2] == '.'))
            elif p[j - 1] == '.' or s[i - 1] == p[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
    
    return dp[m][n]
1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 518ms/step
Prediction (1: ChatGPT, 0: Human): 0.019221099093556404


3) Chatgpt가 만든 코드 입력 prediction: 0.01140505913645029
Enter new code: def isMatch(s: str, p: str) -> bool:
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    
    dp[0][0] = True
    
    # Handle cases where s is empty
    for j in range(1, n + 1):
        if p[j - 1] == '*':
            dp[0][j] = dp[0][j - 1]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == '*':
                dp[i][j] = dp[i][j - 1] or dp[i - 1][j]
            elif p[j - 1] == '?' or s[i - 1] == p[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
    
    return dp[m][n]

1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 479ms/step
Prediction (1: ChatGPT, 0: Human): 0.01140505913645029


4) 사람이 만든 코드 입력 prediction : 0.009712227620184422
Enter new code: class Solution:
    def romanToInt(self, s: str) -> int:
        m = {
            'I': 1,
            'V': 5,
            'X': 10,
            'L': 50,
            'C': 100,
            'D': 500,
            'M': 1000
        }
        
        ans = 0
        
        for i in range(len(s)):
            if i < len(s) - 1 and m[s[i]] < m[s[i+1]]:
                ans -= m[s[i]]
            else:
                ans += m[s[i]]
        
1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 565ms/step
Prediction (1: ChatGPT, 0: Human): 0.009712227620184422
