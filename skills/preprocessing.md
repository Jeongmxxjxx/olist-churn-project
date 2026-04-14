# Skill: 전처리 및 결측치 처리

## 처리 순서
1. 원본 데이터 필터링 (분석 대상 주문 추출)
2. 결측치 처리
3. 인코딩
4. 스케일링
5. 클래스 불균형 처리

---

## 1. 원본 데이터 필터링
```python
# 정상 배송 완료 주문만 사용
orders_clean = orders[orders['order_status'] == 'delivered'].copy()

# order_delivered_customer_date 결측 제거 (배송 지연 계산 불가)
orders_clean = orders_clean.dropna(subset=['order_delivered_customer_date'])

print(f"필터 전: {len(orders):,}건 → 필터 후: {len(orders_clean):,}건")
```

---

## 2. 피처 테이블 결측치 처리

### review_score 결측
```python
# 전략: 중앙값 대체 + 결측 플래그 추가
df['review_missing_flag'] = df['avg_review_score'].isnull().astype(int)
df['avg_review_score']   = df['avg_review_score'].fillna(df['avg_review_score'].median())
df['last_review_score']  = df['last_review_score'].fillna(df['last_review_score'].median())
```

### 배송 지연 관련 결측
```python
# avg_delay, max_delay: 0으로 대체 (배송 지연 없음으로 해석)
delay_cols = ['avg_delay', 'max_delay', 'delay_rate', 'is_ever_delayed']
df[delay_cols] = df[delay_cols].fillna(0)
```

### 카테고리 관련 결측
```python
# product_category_name 결측: 'unknown' 대체는 피처 생성 단계에서 이미 처리
# unique_categories 결측 → 0
cat_cols = ['unique_categories', 'category_diversity_ratio', 'total_items_purchased']
df[cat_cols] = df[cat_cols].fillna(0)
```

---

## 3. 학습/검증 분리 (스케일링 전에 반드시 먼저 수행)

```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=['customer_unique_id', 'label'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```
> ⚠️ 분리 전에 스케일링/인코딩 적용 금지 → 데이터 리키지 발생

---

## 4. 스케일링

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# train 기준으로 fit, test는 transform만
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```
> LightGBM/XGBoost는 스케일링 불필요. Logistic Regression 베이스라인에만 적용.

---

## 5. 클래스 불균형 처리

Olist는 대부분 고객이 1회 구매 → 이탈(label=1) 비율이 매우 높음

### 방법 1: class_weight (권장, 간단)
```python
# LightGBM
import lightgbm as lgb
model = lgb.LGBMClassifier(class_weight='balanced', random_state=42)
```

### 방법 2: SMOTE (오버샘플링)
```python
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print(pd.Series(y_train_res).value_counts())
```
> SMOTE는 **train 셋에만** 적용. test 셋에는 절대 적용 금지.

---

## 처리 결과 체크
```python
# 결측치 잔존 확인
print(df.isnull().sum()[df.isnull().sum() > 0])

# 클래스 비율
print(y_train.value_counts(normalize=True))
```
