# Iris-clasification
## Logical  view
- TraningData 
    - 데이터 샘플의 두 가지 리스트(모델 학습/테스트에 사용) 존재 -> KnownSample instance
    - hyperparameter : 모델 튜닝 값
    - metadata : 데이터 업로드 일시, 데이터 이름, 모델 테스트 일시  
    <br/>

- Sample
    - iris sepal(꽃밭침)과 petal(꽃잎)의 길이와 너비를 나타내는 4개의 속성
    - label : iris 종류    
    <br/>

- KnownSample
    - Sample의 하위 클래스
    - 분류되어 할당된 species(종) 속성을 추가 
    - 모델 학습 및 테스트에 사용  
    <br/> 
- Hyperparameter
    - 최근접 이웃의 수 k, 테스트 요약
    - 얼마나 많은 샘플이 정확히 분류되었는지 quality  
    <br/>
- UnknownSample
    - Sample의 하위 클래스
    - sample의 처음 속성 4개를 가지고 있고 분류할 때 객체 제공

### Classification Algorithm
- k-최근접 이웃 알고리즘
    - 알려진 샘플 집합과 알려지지 않은 샘플이 주어졌을 때 미지의 샘플 근처의 이웃을 찾음
    - k개의 이웃을 찾고 이웃의 클래스를 결정하는 방법에 따라 분류
    - 유클리드 거리 계산

## Context view
### actor class
- 식물학자
    - 적절히 분류된 학습 데이터, 테스트 데이터 제공
    - 매개변수 수립 위한 테스트 수행
    - k-nn k값 결정

- 사용자
    - 미지 데이터 분류
    - 분류 결과 확인

## Process view
1. TrainingData를 구성하는 Sample의 초기 데이터셋 업로드
2. 주어진 k값으로 분류기 테스트
3. 새 Sample 객체로 분류 요청

## Development view
- DataModel
- View Functions
    - flask 클래스의 인스턴스 생성
- Tests
    - 모델과 뷰 기능 단위 테스트

## Deployment view
- 클라이언트
    - 분류기 웹 서비스에 연결해 요청
- GUnicorn 웹 서버
- 분류기 애플리케이션
    - flask 애플리케이션