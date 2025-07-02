<h1 style="text-align: center;">🚗HAI(하이)! - Hecto AI Challenge : 2025 상반기 헥토 채용 AI 경진대회</h1>
<hr>
<p style="text-align: center;">
    <a href="https://github.com/donghyun0518/final-project-endoscope/blob/main/%EB%82%B4%EC%8B%9C%EA%B2%BD%EB%AA%A8%EB%8D%B8pdf.pdf" target="_blank">
        <img src="https://github.com/donghyun0518/final-project-endoscope/blob/main/%EC%8B%A4%EC%8B%9C%EA%B0%84%EB%82%B4%EC%8B%9C%EA%B2%BD%ED%91%9C%EC%A7%80.png" alt="Project Cover" style="width: 1000px; border: 1px solid #c9d1d9; border-radius: 8px;">
    </a>
</p>

[[프로젝트 발표 자료](https://github.com/donghyun0518/final-project-endoscope/blob/main/%EB%82%B4%EC%8B%9C%EA%B2%BD%EB%AA%A8%EB%8D%B8pdf.pdf)]

## 🔍 프로젝트 개요
- **목적** : 중고차 이미지로부터 차량의 정확한 차종(총 396개)을 분류하는 AI 모델 개발
- **주제** : 실생활에 밀접한 중고차 거래 및 차량 인식 시스템을 위한 고성능 이미지 분류기 설
- **기간** : 2025.05.19 ~ 2025.06.16 (약 1달)
- **팀 구성** : 3인

## ⚙️ 주요 수행 과정
**1. **문제 정의****
   - 위/ 대장 내시경 검사 중 의료진의 병변 발견을 보조하는 실시간 AI 시스템 필요성 확인.
   - 병변의 종류(용종, 궤양, 암) 및 탐지 정확도를 주요 지표로 설정.

**2. **데이터 수집 및 전처리****
   - 데이터 출처 : AI Hub 내시경 이미지 합성데이터
   - 전처리 작업
     - 이미지 크기 통일화
     - 라벨링 json 파일에서 사용할 라벨링 형식(바운딩 박스) 추출

**3. **모델 설계****
   - 모델 선정 : YOLOv8m 기반의 실시간 객체 탐지 모델 활용
   - 모델 학습:
     - 데이터셋을 훈련, 검증, 테스트 데이터로 분할
     - 크로스엔트로피 손실 함수와 Adam 옵티마이저를 사용해 학습진행.
     - CUDA를 활용하여 모델 학습진행.
   - 평가 지표 :
     - 정확도(Precision), 재현율(Recall), F1 Score 및 mAP50(Mean Average Precision)으로 성능 평가.

**4. 모델 테스트 및 결과 분석**
   - 테스트 데이터셋을 통해 모델의 병변 탐지 정확도 확인.
   - 위/대장 두 장기를 동시에 학습시킬 시 현재 부위가 위인지 대장인지 구분하지 못하는 문제 발견
     - 해결 방법 :
       - 위와 대장을 따로 학습시켜 인터페이스 구현 시 선택 가능하도록 설계
   - 정확도 확인 결과 :
     - 대장
       - Precision : 0.74 , Recall : 0.699, mAP50 : 0.72
     - 위
       - Precision : 0.75 , Recall : 0.63, mAP50 : 0.69

**5. 인터페이스 및 홈페이지 구현**<br>
    [![유튜브에서 시연 영상보기](https://github.com/donghyun0518/final-project-endoscope/blob/main/%EB%82%B4%EC%8B%9C%EA%B2%BD%EC%8B%9C%EC%97%B0%EC%98%81%EC%83%81%ED%91%9C%EC%A7%80.png)](https://www.youtube.com/watch?v=94uCWk3kKMI)
    [이미지 클릭시 유튜브에서 시청 가능합니다.]
   
**6. 한계점**
   - 병변의 종류가 적어 다양한 병변 탐지 불가능
   - 개인 컴퓨터의 성능 제한으로 인해 더 큰 YOLO 모델에 적용하지 못함.

## 🧑🏻‍💻환경
- Python
- VScode
- CUDA

