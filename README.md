# Olist E-Commerce 고객 이탈 예측

> Kaggle의 [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)를 활용한 고객 이탈 예측 분류 모델

---

## 프로젝트 개요

브라질 최대 이커머스 플랫폼 **Olist**의 2016~2018년 주문 데이터를 분석하여,  
**"특정 고객이 향후 재구매를 하지 않을 것인가"** 를 예측하는 이진 분류 모델을 개발합니다.

재구매율이 약 **3% 미만**인 극심한 불균형 데이터셋에 대해  
**Time-based Split**과 **SMOTE-NC**를 적용하여 현실적인 예측 파이프라인을 구성했습니다.

---

## 프로젝트 구조
```
olist-churn-project/
│
├── data/
│   ├── raw/               # 원본 CSV
│   ├── interim/           # 테이블 병합 후 중간 저장
│   └── processed/         # 최종 피처 테이블 (SMOTE 적용 전)
│
├── notebooks/
│   ├── eda_raw.ipynb             # 원본 데이터 탐색 및 논리적 오류 탐지
│   ├── data_merge.ipynb          # 9개 CSV 테이블 병합
│   ├── feature_engineering.ipynb # RFM 및 파생 변수 생성
│   ├── model_training.ipynb      # SMOTE 적용 및 모델 학습
│   └── model_evaluation.ipynb    # 모델 비교, SHAP 해석
│
├── src/
│   ├── config.py          # 날짜 기준값, 하이퍼파라미터 상수
│   ├── loader.py          # CSV 일괄 로드
│   ├── merge.py           # 테이블 조인 로직
│   ├── features.py        # RFM, 파생 변수 생성 함수
│   ├── preprocess.py      # 인코딩, 스케일링, SMOTE 래퍼
│   ├── evaluate.py        # 공통 평가 함수 (F1, AUC-ROC 등)
│   └── utils.py           # 시각화, 로깅 공통 함수
│
├── models/
│   ├── baseline/          # Logistic Regression
│   ├── tree_based/        # Random Forest, XGBoost, LightGBM
│   └── final/             # 최종 선택 모델
│
├── reports/
│   └── figures/           # EDA 및 평가 결과 이미지
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Time-based Split

날짜 기준으로 **Feature 구간**과 **Label 구간**을 분리하여 데이터 누수를 방지합니다.

| 구간 | 기간 | 역할 |
|------|------|------|
| **Feature** | 2016-09 ~ 2018-03 | RFM, 배송, 리뷰 등 피처 추출 |
| **Label** | 2018-04 ~ 2018-10 | 재주문 있음 → 유지 `0` / 없음 → 이탈 `1` |