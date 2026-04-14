# Skill: EDA (탐색적 데이터 분석)

## 분석 순서

### Step 1. 데이터 로드 및 기본 확인
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

plt.style.use('seaborn-v0_8')
sns.set_theme(font_scale=2.5)

# 핵심 테이블만 로드
orders   = pd.read_csv('data/raw/olist_orders_dataset.csv', parse_dates=[
    'order_purchase_timestamp',
    'order_delivered_customer_date',
    'order_estimated_delivery_date'
])
reviews  = pd.read_csv('data/raw/olist_order_reviews_dataset.csv')
items    = pd.read_csv('data/raw/olist_order_items_dataset.csv')
products = pd.read_csv('data/raw/olist_products_dataset.csv')
customers = pd.read_csv('data/raw/olist_customers_dataset.csv')
```

### Step 2. orders 테이블 우선 확인 (이탈 레이블 생성 기반)
```python
# order_status 분포 → delivered만 사용할 것
print(orders['order_status'].value_counts())

# 날짜 범위 확인
print(orders['order_purchase_timestamp'].min(), orders['order_purchase_timestamp'].max())

# 핵심 날짜 필드 결측치
date_cols = ['order_delivered_customer_date', 'order_estimated_delivery_date']
print(orders[date_cols].isnull().sum())
```

### Step 3. 고객 재구매 분포 확인 (클래스 불균형 사전 파악)
```python
# customer_unique_id 기준 주문 횟수 분포
merged = orders.merge(customers, on='customer_id')
purchase_counts = merged.groupby('customer_unique_id')['order_id'].count()
print(purchase_counts.value_counts().head(10))
# 대부분 1회 구매 → 심각한 클래스 불균형 예상
```

### Step 4. review_score 분포
```python
print(reviews['review_score'].value_counts().sort_index())
sns.countplot(data=reviews, x='review_score')
plt.title('리뷰 점수 분포')
plt.show()

# order당 리뷰 결측 비율
order_with_review = orders[orders['order_status']=='delivered'].merge(
    reviews[['order_id','review_score']], on='order_id', how='left'
)
print("리뷰 결측률:", order_with_review['review_score'].isnull().mean())
```

### Step 5. 배송 지연 분포
```python
delivered = orders[orders['order_status'] == 'delivered'].copy()
delivered['delay_days'] = (
    delivered['order_delivered_customer_date'] -
    delivered['order_estimated_delivery_date']
).dt.days

print(delivered['delay_days'].describe())
print("지연 비율:", (delivered['delay_days'] > 0).mean())
```

### Step 6. 카테고리 다양성 분포
```python
items_with_cat = items.merge(products[['product_id','product_category_name']], on='product_id', how='left')
cat_per_order = items_with_cat.groupby('order_id')['product_category_name'].nunique()
print(cat_per_order.describe())
```

### Step 7. 결측치 시각화
```python
msno.matrix(orders[['order_delivered_customer_date','order_estimated_delivery_date']])
plt.show()
```

## 확인해야 할 핵심 수치

| 항목 | 확인 목적 |
|------|-----------|
| `order_status` 분포 | delivered 비율 파악 |
| `order_delivered_customer_date` 결측률 | 가설1 피처 계산 가능 범위 |
| 1회 구매 고객 비율 | 레이블 클래스 불균형 심각도 |
| `review_score` 1~2점 비율 | 가설2 유효성 |
| 배송 지연 비율 | 가설1 유효성 |
