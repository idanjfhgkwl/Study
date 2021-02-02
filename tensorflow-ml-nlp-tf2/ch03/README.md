# CH03. 자연어 처리 개요 

이번 장에서는 총 4개의 문제를 알아보겠다.  
1. 텍스트 분류
2. 텍스트 유사도 
3. 텍스트 생성 
4. 기계 이해 

위 4가지 문제를 알아보기 전에 먼저 알아야하는 것이 있다.  

**단어 표현**이란 모든 자연어 처리 문제의 기본 바탕이 되는 개념.  
자연어를 어떻게 표현할지 정하는 것이 각 문제를 해결하기 위한 출발점! 

## 단어 표현 
**원-핫 인코딩(one-hot-incoding)**
- 단어를 하나의 벡터로 표현하는 방법
- 각 값은 0 혹은 1 값만 가지는 벡터로 표현  

ex) 6개의 단어(남자, 여자, 아빠, 엄마, 삼촌, 이모) 
- 남자 : [1, 0, 0, 0, 0, 0]
- 여자 : [0, 1, 0, 0, 0, 0]
- 아빠 : [0, 0, 1, 0, 0, 0]
- 엄마 : [0, 0, 0, 1, 0, 0]
- 삼촌 : [0, 0, 0, 0, 1, 0]
- 이모 : [0, 0, 0, 0, 0, 1]

즉, 원-핫 인코딩 방식은 각 단어의 인텍스를 정한 후 각 단어의 벡터에서 그 단어에 해당하는 인덱스의 값을 1로 표현하는 방식  

but, **문제점 두 가지**  
1. 각 단어 벡터의 크기가 너무 커지기 때문에 공간은 너무 많이 사용하여 매우 비효율적
2. 단어의 의미나 특성 같은 것들을 표현 못함

그래서 제안된 다른 방법

1. **카운트 기반 방법**
2. **예측 기반 방법**  


**1. 카운트 기반 방법**  
 
 - 어떤 글의 문맥 안에 단어가 동시에 등장하는 횟수를 세는 방법
 - 단어가 동시에 등장하는 횟수를 '동시 출현' 혹은 '공기'라고 부르고 영어로는 Co-occurrence
 
다양한 카운트 기반 방법
- 특이값 분해(Singular Value Decomposition, SVD)
- 잠재의미분석(Latent Semantic Analysis, LSA)
- Hyperspace Analogue to Language(HAL)
- Hellinger PCA(Principal Component Analysis)

ex)  

*- 성진과 창욱은 야구장에 갔다.*  

*- 성진과 태균은 도서관에 갔다.*  

*- 성진과 창욱은 공부를 좋아한다.* 

![KakaoTalk_20201221_145323770.jpg](attachment:KakaoTalk_20201221_145323770.jpg)

![KakaoTalk_20201221_145332885.jpg](attachment:KakaoTalk_20201221_145332885.jpg)

이러한 카운트 기반 방법의 장점은 **빠르다는 점**이다.

**2. 예측 기반 방법**  
예측 기반 방법이란 신경망 구조 혹은 어떠한 모델을 사용해 특정 문맥에서 어떤 단어가 나올지를 예측하면서 단어를 벡터로 만드는 방식  
- Word2vec 
- NNLM(Neural Network Language Model) 
- RNNLM(Recurrent Neural Network Language Model)  

 Word2vec 을 가장 자주 사용하므로 이를 살펴보자.  
 Word2vec은 CBOW(Continuous Bag of Words)와 Skip-Gram이라는 두가지 모델로 나뉨   
 
**CBOW(Continuous Bag of Words)** : 어떤 단어를 문맥 안의 주변 단어들을 통해 예측하는 방법  
**Skip-Gram** : 어떤 단어를 가지고 특정 문맥 안의 주변 단어들을 예측하는 방법  
 
*ex) 창욱은 냉장고에서 음식을 꺼내서 먹었다.*  

- CBOW(Continuous Bag of Words) : 창욱은 냉장고에서 음식은 ___ 꺼내서 먹었다.
- Skip-Gram : ___ _____ 음식을 ____ _____ 

![KakaoTalk_20201221_152240407.jpg](attachment:KakaoTalk_20201221_152240407.jpg)

**CBOW**의 경우 다음과 같은 순서로 학습한다.  
1. 각 주변 단어들을 원-핫 벡터로 만들어 입력값으로 사용한다(입력층 벡터) 
2. 가중치 행렬을 각 원-핫 벡터에 곱해서 N-차원 벡터를 만든다(N-차원 은닉층) 
3. 만들어진 n- 차원 벡터를 모두 더한 후 개수로 나눠 평균 n-차원 벡터를 만든다(출력층 벡터)
4. n-차원 벡터에 다시 가중치 행렬을 곱해서 원-핫 벡터와 같은 차원의 벡터로 만든다. 
5. 만들어진 벡터를 실제 예측하려고 하는 단어의 원-핫 벡터와 비교해서 학습한다.  

**Skip-Gram**은 다음과 같은 순서로 학습한다.  
1. 하나의 단어를 원-핫 벡터로 만들어서 입력값으로 사용한다(입력층 벡터)
2. 가중치 행력을 원-핫 벡터에 곱해서 n-차원 벡터를 만든다(N-차원 은닉층) 
3. n-차원 벡터에 다시 가중치 행렬을 곱해서 원-핫 벡터와 같은 차원의 벡터로 만든다(출력층 벡터)
4. 만들어진 벡터를 실제 예측하려는 주변 단어들 각각의 원-핫 벡터와 비교해서 학습한다. 

## 1. 텍스트 분류 
- 텍스트 분류(Text Classification)는 자연어 처리 문제 중 가장 대표적이고 많이 접하는 문제
- 자연어 처리 기술을 활용해 특정 텍스트를 사람들이 정한 몇 가지 범주(Class) 중 어느 범주에 속하는지 분류하는 문제 
- 2가지 범주에 대해 구분하는 문제를 **이진 분류(Binara classification)문제** 3개 이상의 범주에 대해 분류하는 문제를 통틀어 **다중 범주 분류(Multi class classification)문제**라 한다. 

### 텍스트 분류의 예시  
- 스팸 분류 
- 감정 분류 
- 뉴스 기사 분류
- 지도 학습을 통한 텍스트 분류 
    + 나이브 베이즈 분류(Naive Bayes Classifier)
    + 서포트 벡터 머신(Support Vecter Machine)
    + 신경망(Neural Network)
    + 선형 분류(Linear Classifier)
    + 로지스틱 분류(Logistic classifier)
    + 랜덤 포레스트(Random Forest)
- 비지도 학습을 통한 텍스트 분류
    + K-평균 군집화(K-means Clustering)
    + 계층적 군집화(Hierarchical Clustering)  
    
지도 학습과 비지도 학습 중 어떤 방법을 사용할지 결정하는 데 기여하는 가장 큰 기준은 데이터에 정답 라벨이 있느냐 없느냐.    

## 2. 텍스트 유사도 
- 텍스트 유사도란 텍스트가 얼마나 유사한지를 표현하는 방식 중 하나  


    + 자카드 유사도
    + 코사인 유사도
    + 유클리디언 유사도
    + 맨하탄 유사도

ex)  

*- 휴일인 오늘도 서쪽을 중심으로 폭염이 이어졌는데요, 내일은 반가운 비 소식이 있습니다.*  

*- 폭염을 피해서 휴일에 놀러왔다가 갑작스런 비로 인해 망연자실하고 있습니다.*  

이 두 문장을 우선 TF-IDF로 벡터화


```python
from sklearn.feature_extraction.text import TfidfVectorizer
sent = ("휴일 인 오늘 도 서쪽 을 중심 으로 폭염 이 이어졌는데요, 내일 은 반가운 비 소식 이 있습니다.", "폭염 을 피해서 휴일 에 놀러왔다가 갑작스런 비 로 인해 망연자실 하고 있습니다.")
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(sent) # 문장 벡터화 진행 

idf = tfidf_vectorizer.idf_
print(dict(zip(tfidf_vectorizer.get_feature_names(), idf))) # 각 수치에 대한 값 시각화 
```

    {'갑작스런': 1.4054651081081644, '내일': 1.4054651081081644, '놀러왔다가': 1.4054651081081644, '망연자실': 1.4054651081081644, '반가운': 1.4054651081081644, '서쪽': 1.4054651081081644, '소식': 1.4054651081081644, '오늘': 1.4054651081081644, '으로': 1.4054651081081644, '이어졌는데요': 1.4054651081081644, '인해': 1.4054651081081644, '있습니다': 1.0, '중심': 1.4054651081081644, '폭염': 1.0, '피해서': 1.4054651081081644, '하고': 1.4054651081081644, '휴일': 1.0}
    

벡터화 한 값은 자카드 유사도를 제외한 유사도 측정에 사용할 것이다. 
자카드 유사도는 벡터화가 필요없다. 

### 자카드 유사도(Jaccard Similarity)
- 두 문장을 각각 단어의 집합으로 만든 뒤 두 집합을 통해 유사도를 측정하는 방식 중 하나.
- 유사도를 측정하는 방법은 두 집합의 교집합인 공통된 단어의 개수를 두 집합의 합집합, 즉 전체 단어의 수로 나누면 된다. 
- 결과값은 0~1 사이의 값, 1에 가까울수록 유사도가 높다는 의미

![eaef5aa86949f49e7dc6b9c8c3dd8b233332c9e7.svg](attachment:eaef5aa86949f49e7dc6b9c8c3dd8b233332c9e7.svg)

A = (휴일, 인, 오늘, 도, 서쪽, 을, 중심, 으로, 폭염, 이, 이어졌는데요, 내일, 은, 반가운, 비, 소식, 이, 있습니다.)  

B = (폭염, 을, 피해서, 휴일, 에, 놀러왔다가, 갑작스런, 비, 로, 인해, 망연자실, 하고, 있습니다.)

교집합 AB = (휴일, 폭염, 비, 을, 있습니다)

교집합의 개수는 6개, 합집합의 개수는 24개 이므로 자카드 유사도는 6/24 

### 코사인 유사도
- 두 개의 벡터값에서 코사인 각도를 구하는 방법
- -1 ~ 1사이의 값을 가지고 1에 가까울수록 유사
- 가장 널리 사용됨 
- 코사인 유사도는 두 벡터 간의 각도를 구하는 것으로 방향성의 개념이 더해져 더욱 정확
- 두 문장이 유사하다면 같은 방향, 유사하지 않을수록 직교

![2a8c50526e2cc7aa837477be87eff1ea703f9dec.svg](attachment:2a8c50526e2cc7aa837477be87eff1ea703f9dec.svg)

앞서 TF-IDF로 벡터화한 문장을 사용해 코사인 유사도를 구해보자. 


```python
from sklearn.metrics.pairwise import cosine_similarity

cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2]) # 첫 번쨰와 두 번째 문장 비교
```




    array([[0.17952266]])



A와 B의 코사인 유사도는 0.179로 산출된다.

### 유클리디언 유사도
- 가장 기본적인 거리를 측정하는 유사도 공식 
![%EB%8B%A4%EC%9A%B4%EB%A1%9C%EB%93%9C.jpg](attachment:%EB%8B%A4%EC%9A%B4%EB%A1%9C%EB%93%9C.jpg)  
- 유클리디언 거리(Euclidean Distance) 혹은 L2거리 라고 불리며, 두 점 사이의 최단 거리를 구하는 접근법




```python
from sklearn.metrics.pairwise import euclidean_distances

euclidean_distances(tfidf_matrix[0:1],tfidf_matrix[1:2])
```




    array([[1.28099753]])



앞서 나왔던 유사도 방식들은 모두 0과 1 사이의 값을 가졌는데, 유클리디언 유사도는 1보다 큰 값이 나왔다.  
일반적으로 유클리디언 유사도는 단순히 두 점 사이의 거리를 뜻하기 때문에 값에 제한이 없다.  
이러한 제한이 없는 방식은 사용하기 어렵기 때문에 값을 일반화(Normalize)한 후 다시 측정해야한다. 

L1 정규화 방법으로 정규화 해주자 


```python
import numpy as np 

def l1_normalize(v):
    norm = np.sum(v)
    return v / norm

tfidf_norm_l1 = l1_normalize(tfidf_matrix)
euclidean_distances(tfidf_norm_l1[0:1], tfidf_norm_l1[1:2])
```




    array([[0.20491229]])



### 맨하탄 유사도(Manhattan Similarity)

- 맨하탄 거리를 통해 유사도를 측정하는 방법 
- 맨하탄 거리란 사각형 격자로 이뤄진 지도에서 출발점에서 도착점까지를 가로지르지 않고 갈 수 있는 최단거리를 구하는 공식
- 유클리디언 거리를 L2 거리라고 하고, 맨하탄 거리를 L1 거리라고 부른다. 
![%EB%8B%A4%EC%9A%B4%EB%A1%9C%EB%93%9C%20%281%29.jpg](attachment:%EB%8B%A4%EC%9A%B4%EB%A1%9C%EB%93%9C%20%281%29.jpg)  
![KakaoTalk_20201221_231211256.png](attachment:KakaoTalk_20201221_231211256.png)

맨하탄 거리도 거리를 통해 유사도를 측정하는 것이기에 정규화를 해줘야한다. 


```python
from sklearn.metrics.pairwise import manhattan_distances

manhattan_distances(tfidf_norm_l1[0:1], tfidf_norm_l1[1:2])
```




    array([[0.77865927]])



맨하탄 유사도로 측정했을 때 유사도가 가장 높게 나왔다.  
의도하는 방향에 맞는 유사도 측정 방법을 고르는 것이 매우 중요하다. 

## 3. 기계 이해 
- 기계가 어떤 텍스트에 대한 정보를 학습하고 사용자가 질의를 던졌을 때 그에 대해 응답하는 문제 
- 기계가 텍스트를 이해하고 논리적 추론을 할 수 있는지 데이터 학습을 통해 보는 것 

### 데이터 이해하기 

문제를 해결하기 위한 모델에 문제가 없더라도 데이터마다 적합한 모델이 있는데 해당 모델과 데이터가 잘 맞지 않으면 좋은 결과를 얻을 수 없다. 즉, 아무리 좋은 모델이더라도 데이터와 궁합이 맞지 않는 모델이라면 여러 가지 문제에 직면하게 된다.   

이럴 때 하는 것이 **탐색적 데이터 분석(EDA : Exploratory Data Analysis)** 라 한다. 이 과정에서 생각하지 못한 데이터의 여러 패턴이나 잠재적인 문제점 등을 발견할 수 있다.  
![KakaoTalk_20201222_005301507.jpg](attachment:KakaoTalk_20201222_005301507.jpg)

영화 리뷰 데이터를 활용해 실습해보자. 


```python
import os 
import re 

import pandas as pd
import tensorflow as tf
from tensorflow.keras import utils

data_set = tf.keras.utils.get_file(
      fname="imdb.tar.gz", 
      origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", 
      extract=True)
```

- 텐서플로 케라스 모듈의 get_file 함수를 통해 IMDB 데이터를 가져온다. 
- origin에 데이터의 URL을 넣으면 해당 URL에서 데이터를 다운로드 
- tar.gz라는 것은 압축 파일임을 의미 
- extract를 통해 다운로드한 압축 파일의 압축 해제 여부 지정 
- fname은 다운로드한 파일의 이름을 재지정  

해당 데이터는 디렉터리 안에 txt 파일 형태로 있어서 판다스의 데이터프레임을 만들기 위해서는 변환 작업을 진행해야함  
변환 작업에 필요한 함수 두 개 
- 각 파일에서 리뷰 텍스트를 불러오는 함수 
- 각 리뷰에 해당하는 라벨값을 가져오는 함수


```python
# 각 파일에서 리뷰 텍스트를 불러오는 함수
def directory_data(directory):
    data = {}
    data["review"] = []
    for file_path in os.listdir(directory):
        with open(os.path.join(directory, file_path), "r", encoding='utf-8') as file:
            data["review"].append(file.read())
            
    return pd.DataFrame.from_dict(data)      
```


```python
def data(directory):
    pos_df = directory_data(os.path.join(directory, "pos")) # pos폴더에서 데이터프레임 반환
    neg_df = directory_data(os.path.join(directory, "neg")) # neg폴더에서 데이터프레임 반환 
    pos_df["sentiment"] = 1 # 긍정 positive는 1 
    neg_df["sentiment"] = 0 # 부정 negative는 0 

    return pd.concat([pos_df, neg_df])
```


```python
# 판다스 데이터프레임을 반환받는 구문
train_df = data(os.path.join(os.path.dirname(data_set), "aclImdb", "train"))
test_df = data(os.path.join(os.path.dirname(data_set), "aclImdb", "test"))
```

이렇게 만든 데이터프레임의 결과를 확인해보자


```python
train_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bromwell High is a cartoon comedy. It ran at t...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Homelessness (or Houselessness as George Carli...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Brilliant over-acting by Lesley Ann Warren. Be...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>This is easily the most underrated film inn th...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>This is not the typical Mel Brooks film. It wa...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



review 와 sentiment가 잘들어가 있는 것을 확인했다.   
판다스의 데이터프레임으로부터 리뷰 문장 리스트를 가져오는 함수를 만들어 보자. 


```python
reviews = list(train_df['review'])
```

review는 각 문장을 리스트로 담고 있다.  
그럼 단어를 토크나이징하고 문장마다 토크나이질된 단어의 수를 저장하고 그 단어들을 붙여 알파벳의 전체 개수를 저장하는 부분을 만들어보자. 


```python
# 문자열 문장 리스트를 토크나이징 
tokenized_reviews = [r.split() for r in reviews]

# 토크나이징된 리스트에 대한 각 길이를 저장 
review_len_by_token = [len(t) for t in tokenized_reviews]

# 토크나이징된 것을 붙여서 음절의 길이를 저장 
review_len_by_eumjeol = [len(s.replace(' ', ''))for s in reviews]
```

위와 같이 만드는 이유는 **문장에 포함된 단어와 알파벳의 개수에 대한 데이터 분석을 수월하게 만들기 위해서**  

이제 데이터 분석을 위한 사전 준비 작업이 완료됐으니 실제로 데이터 분석을 진행해 보자.  
먼저 히스토그램으로 문장을 구성하는 단어의 개수와 알파벳 개수를 알아보자.  


```python
import matplotlib.pyplot as plt 

# 그래프에 대한 이미지 크기 선언 
# figsize : (가로, 세로) 형태의 튜플로 입력
plt.figure(figsize=(12, 5))
# 히스토그램 선언 
# bins: 히스토그램 값들에 대한 버켓 범위
# range: x축 값의 범위
# alpha: 그래프 색상 투명도
# color: 그래프 색상
# label: 그래프에 대한 라벨
plt.hist(review_len_by_token, bins=50, alpha=0.5, color= 'r', label='word')
plt.hist(review_len_by_eumjeol, bins=50, alpha=0.5, color= 'b', label='alphabet')
plt.yscale('log', nonposy='clip')
# 그래프 제목 
plt.title('Review Length Histogram')
# 그래프 x축 라벨 
plt.xlabel('Review Length')
# 그래프 y축 라벨 
plt.ylabel('Number of Reviews')

```




    Text(0, 0.5, 'Number of Reviews')




    
![png](output_40_1.png)
    


- 빨간색 히스토그램은 단어 개수에 대한 히스토그램  
- 파란색은 알파벳 개수의 히스토그램  
- 데이터의 전체적인 분포를 시각적으로 확인 
- 이상치 값을 확인 

데이터 분포를 통계치로 수치화해보자


```python
import numpy as np 

print('문장 최대 길이 : {}'. format(np.max(review_len_by_token)))
print('문장 최소 길이 : {}'. format(np.min(review_len_by_token)))
print('문장 평균 길이 : {:2f}'. format(np.mean(review_len_by_token)))
print('문장 길이 표준편차 : {:2f}'. format(np.std(review_len_by_token)))
print('문장 중간 길이 : {}'. format(np.median(review_len_by_token)))
# 사분위에 대한 경우는 0~100 스케일로 돼 있음
print('제 1사분위 길이 : {}'. format(np.percentile(review_len_by_token, 25)))
print('제 3사분위 길이 : {}'. format(np.percentile(review_len_by_token, 75)))

```

    문장 최대 길이 : 2470
    문장 최소 길이 : 10
    문장 평균 길이 : 233.787200
    문장 길이 표준편차 : 173.729557
    문장 중간 길이 : 174.0
    제 1사분위 길이 : 127.0
    제 3사분위 길이 : 284.0
    

이번엔 박스플롯으로 데이터를 시각화해 보자.


```python
plt.figure(figsize=(12, 5))
# 박스 플롯 생성 
# 첫 번째 인자 : 여러 분포에 대한 데이터 리스트를 입력 
# labels : 입력한 데이터에 대한 라벨
# showmeans : 평균값을 마크함

plt.boxplot([review_len_by_token],
           labels =['token'],
           showmeans=True)

print(plt.boxplot)
```

    <function boxplot at 0x0000029A00682EE0>
    


    
![png](output_44_1.png)
    


위 박스플롯 결과값을 확대해서 살펴보자.

![KakaoTalk_20201222_161140925.jpg](attachment:KakaoTalk_20201222_161140925.jpg)

이번엔 문장의 알파벳 개수를 나타내는 박스플롯을 만들어보자. 


```python
plt.figure(figsize=(12, 5))
plt.boxplot([review_len_by_eumjeol],
           labels=['Eumjeol'],
           showmeans=True)
```




    {'whiskers': [<matplotlib.lines.Line2D at 0x29a013010a0>,
      <matplotlib.lines.Line2D at 0x29a01283d90>],
     'caps': [<matplotlib.lines.Line2D at 0x29a01283130>,
      <matplotlib.lines.Line2D at 0x29a01257df0>],
     'boxes': [<matplotlib.lines.Line2D at 0x29a013011c0>],
     'medians': [<matplotlib.lines.Line2D at 0x29a01257370>],
     'fliers': [<matplotlib.lines.Line2D at 0x29a01257280>],
     'means': [<matplotlib.lines.Line2D at 0x29a01257670>]}




    
![png](output_47_1.png)
    


다음으로 워드클라우드로 데이터를 시각화해보자. 


```python
!pip install wordcloud
```

    Collecting wordcloud
      Downloading wordcloud-1.8.1-cp38-cp38-win_amd64.whl (155 kB)
    Requirement already satisfied: numpy>=1.6.1 in c:\users\user\anaconda3\lib\site-packages (from wordcloud) (1.18.5)
    Requirement already satisfied: matplotlib in c:\users\user\anaconda3\lib\site-packages (from wordcloud) (3.2.2)
    Requirement already satisfied: pillow in c:\users\user\anaconda3\lib\site-packages (from wordcloud) (7.2.0)
    Requirement already satisfied: python-dateutil>=2.1 in c:\users\user\anaconda3\lib\site-packages (from matplotlib->wordcloud) (2.8.1)
    Requirement already satisfied: kiwisolver>=1.0.1 in c:\users\user\anaconda3\lib\site-packages (from matplotlib->wordcloud) (1.2.0)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in c:\users\user\anaconda3\lib\site-packages (from matplotlib->wordcloud) (2.4.7)
    Requirement already satisfied: cycler>=0.10 in c:\users\user\anaconda3\lib\site-packages (from matplotlib->wordcloud) (0.10.0)
    Requirement already satisfied: six>=1.5 in c:\users\user\anaconda3\lib\site-packages (from python-dateutil>=2.1->matplotlib->wordcloud) (1.15.0)
    Installing collected packages: wordcloud
    Successfully installed wordcloud-1.8.1
    


```python
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
%matplotlib inline

wordcloud = WordCloud(stopwords = STOPWORDS, background_color = 'black', width = 800, height = 600).generate(' '.join(train_df['review']))

plt.figure(figsize = (15, 10))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
```


    
![png](output_50_0.png)
    


결과값을 보면 'br'이 엄청 크게 보이는 것을 알 수 있다.  
이것은 <br>같은 HTML태그가 포함돼 있기 때문이다.  
이러한 부분은 학습에 도움이 되지 않으므로 제거해야 한다.   

마지막으로 긍정 부정의 분포를 확인해보자. 


```python
import seaborn as sns 
import matplotlib.pyplot as plt

sentiment = train_df['sentiment'].value_counts()
fig, axe = plt.subplots(ncols=1)
fig.set_size_inches(6, 3)
sns.countplot(train_df['sentiment'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x29a72240190>




    
![png](output_52_1.png)
    


긍정과 부정의 개수가 12000개로 동일한 것을 알 수 있다. 
