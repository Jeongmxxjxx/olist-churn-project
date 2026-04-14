# Skill: EDA 공통 — 데이터 로드 및 기본 확인

모든 가설별 EDA 시작 전 반드시 먼저 실행.
결과는 `notebooks/01_eda_common.ipynb`로 저장.

---

## Step 1. 데이터 로드
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

plt.style.use('seaborn-v0_8')
sns.set_theme(font_scale=2.5)

orders    = pd.read_csv('data/raw/olist_orders_dataset.csv', parse_dates=[
    'order_purchase_timestamp',
    'order_delivered_customer_date',
    'order_estimated_delivery_date'
])
reviews   = pd.read_csv('data/raw/olist_order_reviews_dataset.csv')
items     = pd.read_csv('data/raw/olist_order_items_dataset.csv')
products  = pd.read_csv('data/raw/olist_products_dataset.csv')
customers = pd.read_csv('data/raw/olist_customers_dataset.csv')
```

---

## Step 2. orders 기본 확인 (레이블 생성 기반)
```python
# order_status 분포 → delivered만 사용
print(orders['order_status'].value_counts())

# 전체 날짜 범위
print(orders['order_purchase_timestamp'].min(),
      orders['order_purchase_timestamp'].max())

# 날짜 필드 결측치
date_cols = ['order_delivered_customer_date', 'order_estimated_delivery_date']
print(orders[date_cols].isnull().sum())

# 결측치 시각화
msno.matrix(orders[date_cols])
plt.title('날짜 필드 결측치 패턴')
plt.tight_layout()
plt.show()
```

---

## Step 3. 고객 재구매 분포 (클래스 불균형 사전 파악)
```python
merged = orders.merge(customers, on='customer_id')
purchase_counts = merged.groupby('customer_unique_id')['order_id'].count()

print("구매 횟수 분포:")
print(purchase_counts.value_counts().head(10))
print(f"\n1회 구매 고객 비율: {(purchase_counts == 1).mean():.1%}")
```

**체크 포인트**
- `order_status` delivered 비율
- 날짜 필드 결측률 (가설1 피처 계산 가능 범위 결정)
- 1회 구매 고객 비율 → 클래스 불균형 심각도 예측
