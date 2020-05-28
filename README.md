# kaggle_plant_pathology2020
- Google에서 제공해주는 Colab gpu 환경에서 다음과 같이 실험을 진행하였음.

- 이 대회는 Computer Vision and Pattern Recognition Conference(CVPR 2020)에서 Fine-Grained Visual Categorization FGVC7 워크숍 일부로써 kaggle Web Site를 통하여 데이터를 제공 받음.

### 문제
=> 농작물에 영향을 미치는 많은 질병의 오진은 화학 물질의 오용으로 이어질 수 있으며, 내성 병원균 균주의 출현, 투입 비용 증가, 심각한 경제적 손실 및 환경 영향으로 더 많은 발생이 이어질 수 있다. 따라서 컴퓨터 비전 기반 모델로 감염된 조직의 나이, 유전적 변이 및 나무 내의 빛 상태로 인한 증상의 큰 변화 등을 알고자 함.

### 목표 
사과 잎의 사진이 주어지면 건강 상태를 정확하게 평가할 수 있을까?

=> 이 대회는 건강에 좋은 잎, 사과 녹에 감염된 잎, 사과 녹병이 있는 잎 및 둘 이상의 질병을 가진 잎을 구별하는데 의미를 둠.

    1. 많은 질병을 정확하게 구별하고, 한 잎에서 하나 이상의 질병을 정확하게 구별.
    2. 희귀한 질병과 새로운 증상을 다룸.
    3. 깊이 인식 - 각도, 빛, 그늘, 잎의 생리적 연령에 대응.
    4. 학습 중 관련 기능 검색하기 위한 식별, 주석, 수량화 및 컴퓨터 비전 안내에 전문적인 지식을 통합.


### 데이터 정보

<p align="center">Plant Pathology Class</p>
<p align="center"><img src="https://user-images.githubusercontent.com/45933225/82079740-4fb07980-971e-11ea-9c9e-c5e74ac4709e.png" width="100%"></p>

    데이터 포맷 : jpg, jpeg
    총 3642장, Train : 1821장 Test : 1821장(Test data label 없음).
    해상도 : 2048 * 1365 * 3
    class : healthy(건강), multiple_diseases(복합 질병),rust(녹병), scab(붉은곰팡이병)
    
    
- Train data graph

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/81252299-8fad9780-9060-11ea-8c17-99b1abd089ef.png" width="75%"></p>

=> combinations 516개, healthy 91개, rust 622개, scab 592개 총 Train data 1821개 확인.

#### 데이터 분류

    1. Train data 각 클래스 별로 8:2 비율로 나누어서 validation data 생성.   -  모델의 성능을 일반화하기 위해서 validation data을 이용(이상적으로  8:2 비율이 적절하다고 함)
    2. 교차 검증(Cross Validation) KFold 적용.    -   모델이 데이터를 판단하기 위한 데이터 수가 적은 경우 검증 성능의 신뢰도가 떨어지는 것을 해결하고자 함(반대로 정확도 향상을 위한것으로도 말할 수 있음)
    
#### 데이터 보완
데이터의 수가 부족하여 데이터를 보완하느 방법으로 이미지의 레이블을 변경하지 않고 픽셀 변화시키는 CNN의 과적합 방지와 성능 향상을 위하여 Data Augmentation의 albumentations Moudle을 사용하였음.

다음과 같이 데이터에 적용함.

    Train data : Resize, RandomCrop, Resize, Flip, ShiftScaleRotate, HorizontalFlip
    Validation data : Resize
    Test data : Resize
    
### 실험 소개 및 수행 내역
문제 해결 방법으로 Computer Vision(CV)의 분야에서 사용하는 Convolution Neural Network(CNN)을 이용하여 컴퓨터가 영상(이미지 및 비디오) 디지털 이미지의 특징을 학습하여 특정 패턴을 사용한 이미지를 분류하고자 함.

다음과 아래의 순서대로 실험을 수행함.

    - 1. 데이터에 적합한 모델 고르기(VggNet, ResNet, DenseNet, EfficientNet).
    - 2. 데이터에 적합한 레이어층 찾기 (Model(v1, v2 etc..)
    - 3. 데이터 보완 및 과적합 방지 위한 교차 검증 적용(K-Fold)과 정확도 향상을 위한 하이퍼파라미터 최적화(Image Resolution, optimizer, learning rate etc..))


### 1. 데이터에 적절한 모델 선정
###### ./select_model 실행 코드 확인 가능
대표적으로 Convolution Neural Network에서 사용하는 모델을 선택하여 실험하였으며 모델을 제외한 나머지 조건들을 동일하게 설정하여 비교하고자 하였음.

모델은 다음과 같이 사용함.

    - VggNet, ResNet, DenseNet, EfficientNet

하이어파라미터 초기화

      - image size(height, width, channel) : 342, 512, 3
      - epoch : 150
      - batch size(train, validation, test) : 8, 8, 1
      - step(train, validation, test) : 80, 20, 1
      - label : one-hot encoding    -   using Categorical entropy
      - optimizer : Adam(learning rate : 0.01)  -   validation loss에 맞게 laerning rate 조정하였음.
      - Augment
        Train data : Resize, RandomCrop, Resize, Flip, ShiftScaleRotate, HorizontalFlip
        Validation data : Resize
        Test data : Resize
        
위에서 제시하는 조건을 동일하게 하여 각 아래 모델들을 학습하였음.

##### - VggNet     

History 

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/82756782-3b076c00-9e17-11ea-9e7c-204770646c51.png" width="50%"></p>

검증 데이터 기준으로 Maximum Accuracy, Minimum Loss 0.9750, 0.1019 나온 것을 확인할 수 있었음.

Test Evaluation

Train, Validation data 전체를 이용하여 학습 Test 결과 accuracy : 0.9750, loss : 0.1130 확인함과 동시에 학습 결과 클래스 분류에 대해서 확인하였음.

Confusion Matrix 

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/82085133-7b842d00-9727-11ea-9f40-c18a3fddccaf.png" width="35%"></p>

데이터의 수가 너무 적은 multi diseases을 판단하기 어려울 것이라고 예상함. 그리고 multi diseases, scab, rust 이 3가지에 대해서 예측을 제대로 못하고 있음도 확인할 수 있음.

100, 150 epoch 학습 후 kaggle 제출 결과 93.6, 93.7 Accuracy 얻을 수 있었음.

##### - ResNet
 
History

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/82756813-730eaf00-9e17-11ea-80fc-0e99b734b0d5.png" width="50%"></p>

검증 데이터 기준으로 Maximum Accuracy, Minimum Loss 0.7562, 0.5031 나온 것을 확인할 수 있었음.
 
Test Evaluation

Train, Validation data 전체를 이용하여 학습 Test 결과 accuracy : 0.67326, loss : 0.61089 확인함과 동시에 학습 결과 클래스 분류에 대해서 확인하였음.

Confusion Matrix

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/82089570-d0777180-972e-11ea-901d-45d71243c07e.png" width="35%"></p>

여기서도 마찬가지로 multi diseases 클래스에 대한 부분을 예측을 제대로 못하고 있으며 healthy클래스도 주로 scab으로 많은 양의 데이터를 분류하고 있음을 확인이 가능함. 모델이 전반적으로 예측 정확도가 낮음.

150epoch 학습 후kaggle 제출 결과 56.3 Accuracy 얻을 수 있었음.

##### - DenseNet

위의 하이퍼파라미터 같은 조건으로 실행한 결과 모델 학습이 원할하게 진행되지 않아서 따로 기록을 하지 않았음.

##### - EfficientNet

History

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/82756839-99344f00-9e17-11ea-8481-51291a264920.png" width="50%"></p>

검증 데이터 기준으로 Maximum Accuracy, Minimum Loss 0.9750, 0.1019 나온 것을 확인할 수 있었음.

Test Evaluation

Train, Validation data 전체를 이용하여 학습 Test 결과 accuracy : 0.94393, loss : 0.1600 확인함과 동시에 학습 결과 클래스 분류에 대해서 확인하였음.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/82090450-93ac7a00-9730-11ea-9958-392bf93c8b76.png" width="35%"></p>

VggNet모델과 비슷한 성능을 가지고 있음. 그렇지만 VggNet모델보다 데이터가 적은 multi diseases 클래스에 대해서 더 높은 예측을 하고 있음도 동시에 확인 할 수 있음.

100, 150 epoch 학습 후 kaggle 제출 결과 94.8, 94.6 Accuracy 얻을 수 있었음.

전반적으로 multi diseases, scab, healthy, rust 순서대로 잘못 판단하고 있으며 다음 아래와 같이 모델들을 학습한 결과를 그래프를 통해서 확인하여 Train, Validation 데이터의 Accuracy, loss를 비교하고자 같은 플롯에 두고 확인하였음.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/82093860-e38e3f80-9736-11ea-99a8-41147d15090b.png" width="60%"></p>

VggNet, EfficientNet 두 모델이 데이터셋에 최적의 가중치에 빠르고 정확하게 수렴하여서 데이터에 맞는 모델 후보로 생각하였으며 EfficientNet의 Resolution, Width, Depth의 3가지 요소의 확장과 Attetnion Squeeze-and-Excitation을 이용하여 더 높은 성능을 기대할 수 있다고 판단하였으며 또한 데이터 수가 적은      multi_diseases 클래스에 대한 부분을 다른 모델 보다 더 옳은 예측하여 선정하게 되었다.


### 2. 데이터에 적합한 레이어층 찾기
###### ./find_layer 실행 코드 확인 가능.
EfficientNet paper에서 제공하는 b1 ~ b7의 순서대로 학습  비교하여 데이터셋에 적합한 레이어층을 찾고자 하였음.

하이퍼파라미터 초기화

      - image size(height, width, channel) : 342, 512, 3
      - epoch : 100
      - step(train, validation, test) : 80, 20, 1
      - label : one-hot encoding    -   using Categorical entropy
      - optimizer : Adam(learning rate : 0.01)  -   validation loss에 맞게 laerning rate 조정하였음.
      - Augment
        Train data : Resize, RandomCrop, Resize, Flip, ShiftScaleRotate, HorizontalFlip
        Validation data : Resize
        Test data : Resize
        
colab환경에 만족하여 batch size는 각 레이어층에 맞게 최대값을 주었으며 나머지는 위에서 제시하는 조건을 동일하게 설정하여 학습하였음.         

##### efficientnet_b1, b2, b3
    batch size(train, validation, test) - b1 : (24, 24, 1), b2 : (20, 20, 1), b3 : (16, 16, 1)

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/82758700-13b69c00-9e23-11ea-8c2b-ed95f2ded99c.png" width="100%"></p>

각 레이어층 별로 검증 데이터의 Maximum Accuracy, Minimum Loss 아래와 같이 확인 할 수 있음.

    b1 : 0.9625, 0.1044,    b2 : 0.9550, 0.1225,    b3 : 0.9718, 0.0868

##### efficientnet_b4 ~ b7

학습 중 모델 최적화 값의 방향을 찾지 못하여 중간에 학습을 중단함. 찾지 못한다고 생각하는 것은 점점 레이어층이 깊어짐에 따라 Vanishing gradient problem가 발생과 원본 이미지의 해상도 크기 감소로 고유의 픽셀 정보가 끝까지 전달을 못하여 모델이 특징 인식을 하지 못하는 것으로 예상함. 

그래서 다음 실험 순서에서는 배치 사이즈와 이미지 크기 둘 중 하나를 선택하여 진행할 필요가 있을 것 같다.


### 3. 데이터 보완 및 과적합 방지 위한 교차 검증 적용과 정확도 향상을 위한 하이퍼파라미터 최적화

