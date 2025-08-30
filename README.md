<h1 style="text-align: center;">🚗HAI(하이)! - Hecto AI Challenge : 2025 상반기 헥토 채용 AI 경진대회</h1>
<hr>
<p style="text-align: center;">
    <a href="https://github.com/donghyun0518/dacon-carclassification-convnextv2/blob/main/%EC%B0%A8%EB%9F%89%20(2).pdf" target="_blank">
        <img src="https://github.com/donghyun0518/dacon-carclassification-convnextv2/blob/main/car_classifier_main.png" alt="Project Cover" style="width: 1000px; border: 1px solid #c9d1d9; border-radius: 8px;">
    </a>
</p>

[프로젝트 발표 자료](https://github.com/donghyun0518/dacon-carclassification-convnextv2/blob/main/%EC%B0%A8%EB%9F%89%20(2).pdf)

## 🔍 프로젝트 개요
- **목적** : 중고차 이미지로부터 차량의 정확한 차종(총 396개)을 분류하는 AI 모델 개발
- **주제** : 실생활에 밀접한 중고차 거래 및 차량 인식 시스템을 위한 고성능 이미지 분류기 설계
- **기간** : 2025.05.19 ~ 2025.06.16 (약 4주간)
- **팀 구성** : 3인
- **수행 역할** : 데이터 전처리, 모델 학습 파이프라인 구축, 실험 전략 설계 및 성능 개선

## ⚙️ 주요 수행 과정
**1. **문제 정의****
   - 다양한 연식, 모델, 색상, 촬영 조건에서 중고차 이미지가 수집되어 **고정된 패턴이 없고 불균형한 데이터셋 구조** 확인
   - **Log Loss 기반 평가 지표**에 따라 모델은 단순 정확도보다 **확률 기반 예측의 신뢰도**가 중요

**2. **데이터 수집 및 전처리****
   - **데이터 구성**:
     - train : 396개 클래스별 폴더(총 33,137장)
     - test : 테스트 이미지 8,258장
     - test.csv, sample_submission.csv : 추론 결과 제출 형식 포함

   - 전처리 작업:
     - 클래스명 한글 -> 영문 변환 후 매핑 파일 제작
     - 이미지 크기 분포 분석
     - 입력 해상도 512x512로 통일화
     - 클래스별 이미지 수 분석 -> 전체 클래스 50~80장 분포 확인 -> 클래스 수 대비 적은 데이터이므로 데이터 증강으로 보완

**3. **모델 설계 및 학습****
   - **사용 모델**:
     - ConvNeXtV2-Base (384)
       - 사전학습: ImageNet-22K -> 1K
       - 공개일: 2023년 (대회 규정 준수)
       - 적은 데이터에도 강건하고, 차량 디테일 인식에 유리한 구조
   - **적용 기법**:
     - 이미지 증강 : AutoAugment, RandomAugment, Cutout (timm 증강 라이브러리, Torchvision)
     - 혼합 기법 : Mixup, CutMix
     - 최적화 : EarlyStopping, EMA, SWA
     - 스케줄러 : Warmup + CosineAnnealing, ReduceLROnPlateau 등 실험
     - 기타 : Label Smoothing, Temperature Scaling
   - **학습 환경**:
     - GPU : RTX A6000 (VRAM 48GB, RunPod 환경)
     - Batch size: 96, Image size: 512x512

**4. 모델 평가 및 결과 분석**
   - **평가 지표**:
     - Log Loss (정답 클래스 확률 정밀도 중심 평가)
     - 보조 지표 : Accuracy, Top-K Acc, Confidence Gap, Entropy
   - **결과 요약**:
     - 최고 성능 모델 Log Loss: 0.0605
     - LB 제출시 Log Loss: 0.13318

**5. 추론 및 앙상블**
   - Test-Time Augmentation (TTA) 적용
   - Soft Voting 기반 앙상블 방식 도입 (다중 모델 학률 평균)
   - 클래스 확률 분포 CSV 저장 후 제출
   - 최종 제출 결과 Log Loss: 0.1165372043 (Public LB 기준)
   
**6. 배포 및 활용 가능성**
   - 사내 차량 관리, 차량 자동 식별, 중고차 판매 플랫폼에 활용 가능
   - ONNX/TensorRT로 최적화 시 실시간 추론도 가능
   - 모델 압축 및 경량화도 향수 과제로 설정 가능

## 🧑🏻‍💻기술 스택
- 프로그래밍: Python, PyTorch, torchvision, timm
- 데이터 처리: pandas, numpy
- 시각화: matplotlib, seaborn
- 증강/최적화: timm, torch_augmentations, tqdm
- 실행 환경: RunPod (Ubuntu 20.04, CUDA 12.4, RTX A6000)

