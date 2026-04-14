# Olist 고객 이탈 예측 프로젝트

## 프로젝트 개요
Kaggle Olist 브라질 이커머스 데이터셋을 이용한 고객 이탈 예측 ML 모델.
**이탈 정의**: 가장 최근 구매일로부터 90일 이상 재구매 없음 → label=1

---

## 디렉터리 구조
```
olist-churn/
├── data/
│   └── raw/          # Kaggle 원본 CSV (수정 금지)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_preprocessing.ipynb
│   └── 04_modeling.ipynb
├── src/
│   ├── features.py   # 피처 생성 함수
│   ├── preprocess.py # 전처리 함수
│   └── evaluate.py   # 평가 함수
├── skills/           # Claude 세부 지침 (아래 참조)
├── outputs/          # 모델, 피처 중요도, 결과 CSV
└── CLAUDE.md
```

---

## 핵심 설계 원칙

### 시간 기반 Train/Label Split (데이터 리키지 방지 필수)
```
cutoff = max(order_purchase_timestamp) - 90일

피처 계산 구간: order_purchase_timestamp <= cutoff
레이블 판정 구간: cutoff < order_purchase_timestamp <= cutoff + 90일
```
- **cutoff 이전 데이터**로만 피처 계산
- cutoff 이후 구매 여부로 label 생성
- label=0(유지), label=1(이탈)

### 고객 식별 기준
- **반드시 `customer_unique_id` 사용** (`customer_id`는 주문마다 새로 발급되므로 재구매 추적 불가)

---

## 핵심 테이블 관계
```
customers (customer_unique_id)
    └── orders (customer_id → unique_id로 매핑)
            ├── order_reviews (review_score)
            └── order_items (product_id)
                    └── products (product_category_name)
```

---

## 가설 및 피처 매핑

| 가설 | 핵심 피처 | 근거 테이블 |
|------|-----------|-------------|
| 가설1: 배송 지연 → 이탈 | `delivery_delay_days`, `is_delayed` | orders |
| 가설2: 낮은 리뷰 → 이탈 | `last_review_score`, `avg_review_score`, `has_low_review` | order_reviews |
| 가설3: 카테고리 다양성 → 유지 | `unique_categories`, `category_diversity_ratio` | order_items, products |

> 상세 피처 정의 → `skills/feature_engineering.md`

---

## 전처리 규칙 요약
- `order_status != 'delivered'` 행 제거 (분석 대상: 정상 배송 완료 주문만)
- `order_delivered_customer_date` 결측 → 해당 주문 제외
- `review_score` 결측 → 중앙값 대체 (또는 별도 플래그 피처 추가)
- `product_category_name` 결측 → `'unknown'` 대체
- 클래스 불균형 처리: SMOTE 또는 `class_weight='balanced'`

> 상세 처리 방법 → `skills/preprocessing.md`

---

## 모델링 스택
- **베이스라인**: Logistic Regression
- **주력 모델**: LightGBM, XGBoost
- **평가 지표**: AUC-ROC (주), F1-score, Precision, Recall (부)
- **검증 방법**: Stratified K-Fold (k=5)

> 상세 튜닝 전략 → `skills/modeling.md`

---

## 코드 컨벤션
- 노트북은 **피처 그룹별로 독립 실행 가능한 셀** 단위로 작성
- 함수는 `src/`에 정의, 노트북에서 import해서 사용
- 시각화: `plt.style.use('seaborn-v0_8')`, `sns.set_theme(font_scale=2.5)`
- 한국어 주석 사용
- 경로는 항상 프로젝트 루트 기준 상대경로

---

## 작업 요청 시 참고 지침

| 요청 유형 | 참조할 skill 파일 |
|-----------|-------------------|
| EDA / 데이터 탐색 | `skills/eda.md` |
| 피처 엔지니어링 | `skills/feature_engineering.md` |
| 전처리 / 결측치 | `skills/preprocessing.md` |
| 모델 학습 / 튜닝 | `skills/modeling.md` |

---

## 자주 하는 실수 체크리스트
- [ ] cutoff 이후 데이터가 피처에 포함되지 않았는가?
- [ ] `customer_id` 대신 `customer_unique_id`를 쓰고 있는가?
- [ ] 학습/검증 셋 분리 전에 스케일링/인코딩을 적용하진 않았는가?
- [ ] 평가 지표가 AUC-ROC 기준인가? (accuracy만 보지 않았는가?)
- [ ] 클래스 불균형을 처리했는가?
