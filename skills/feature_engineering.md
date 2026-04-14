# Skill: 피처 엔지니어링

## 전제 조건
- `customer_unique_id` 기준으로 고객을 식별
- **cutoff 이전 데이터만** 사용해 피처 계산 (데이터 리키지 방지)
- 피처 계산 후 cutoff 이후 구매 여부로 레이블 생성

---

## 0. 기준일(cutoff) 설정 및 레이블 생성
```python
import pandas as pd

# cutoff 설정
cutoff = orders['order_purchase_timestamp'].max() - pd.Timedelta(days=90)

# 고객별 마지막 구매일 (cutoff 이전만)
orders_merged = orders.merge(customers, on='customer_id')
before = orders_merged[
    (orders_merged['order_purchase_timestamp'] <= cutoff) &
    (orders_merged['order_status'] == 'delivered')
].copy()

# cutoff 이후 구매한 고객 (재구매 = 유지 = label 0)
after = orders_merged[
    (orders_merged['order_purchase_timestamp'] > cutoff) &
    (orders_merged['order_status'] == 'delivered')
]
retained = set(after['customer_unique_id'].unique())

# 전체 고객 기준 레이블 생성
all_customers = before['customer_unique_id'].unique()
labels = pd.DataFrame({'customer_unique_id': all_customers})
labels['label'] = labels['customer_unique_id'].apply(
    lambda x: 0 if x in retained else 1  # 1=이탈, 0=유지
)
print(labels['label'].value_counts(normalize=True))  # 불균형 확인
```

---

## 1. 가설1 피처 — 배송 지연 경험
```python
def calc_delivery_features(before_df):
    """cutoff 이전 delivered 주문 대상"""
    df = before_df.copy()
    df['delivery_delay_days'] = (
        df['order_delivered_customer_date'] -
        df['order_estimated_delivery_date']
    ).dt.days

    feat = df.groupby('customer_unique_id').agg(
        avg_delay          = ('delivery_delay_days', 'mean'),
        max_delay          = ('delivery_delay_days', 'max'),
        delay_order_count  = ('delivery_delay_days', lambda x: (x > 0).sum()),
        total_orders       = ('order_id', 'nunique'),
    ).reset_index()

    feat['delay_rate'] = feat['delay_order_count'] / feat['total_orders']
    feat['is_ever_delayed'] = (feat['delay_order_count'] > 0).astype(int)

    # 불필요 피처: is_delayed는 delivery_delay_days > 0 과 동일 → 제거
    return feat
```

**포함 피처**
| 피처명 | 설명 |
|--------|------|
| `avg_delay` | 평균 배송 지연 일수 (음수=빠름) |
| `max_delay` | 최대 지연 일수 |
| `delay_rate` | 전체 주문 중 지연된 주문 비율 |
| `is_ever_delayed` | 한 번이라도 지연 경험 여부 |

---

## 2. 가설2 피처 — 리뷰 점수 (만족도)
```python
def calc_review_features(before_df, reviews_df):
    order_review = before_df[['order_id','customer_unique_id']].merge(
        reviews_df[['order_id','review_score']], on='order_id', how='left'
    )
    feat = order_review.groupby('customer_unique_id').agg(
        avg_review_score   = ('review_score', 'mean'),
        last_review_score  = ('review_score', 'last'),   # 가장 최근 주문 리뷰
        low_review_count   = ('review_score', lambda x: (x <= 2).sum()),
    ).reset_index()

    feat['has_low_review'] = (feat['low_review_count'] > 0).astype(int)
    return feat
```

**주의**: `review_score` 결측 처리 → `preprocessing.md` 참조

---

## 3. 가설3 피처 — 카테고리 다양성
```python
def calc_category_features(before_df, items_df, products_df):
    items_cat = items_df.merge(
        products_df[['product_id','product_category_name']], on='product_id', how='left'
    )
    # product_category_name 결측 → 'unknown' 대체
    items_cat['product_category_name'] = items_cat['product_category_name'].fillna('unknown')

    order_cat = before_df[['order_id','customer_unique_id']].merge(
        items_cat, on='order_id', how='left'
    )
    feat = order_cat.groupby('customer_unique_id').agg(
        unique_categories    = ('product_category_name', 'nunique'),
        total_items_purchased = ('product_id', 'count'),
    ).reset_index()

    feat['category_diversity_ratio'] = (
        feat['unique_categories'] / feat['total_items_purchased']
    )
    return feat
```

**제거 피처 근거**
- `total_orders`: Olist에서 대부분 고객이 1회 구매 → 거의 모든 값이 1, 예측력 없음
- `purchase_frequency` 계열: 위와 같은 이유로 분산 없음

---

## 4. 피처 병합
```python
def build_feature_table(labels, feat_delay, feat_review, feat_cat):
    df = labels.copy()
    df = df.merge(feat_delay,  on='customer_unique_id', how='left')
    df = df.merge(feat_review, on='customer_unique_id', how='left')
    df = df.merge(feat_cat,    on='customer_unique_id', how='left')
    return df
```

---

## 최종 피처 목록

| 피처명 | 가설 | 타입 |
|--------|------|------|
| `avg_delay` | 1 | float |
| `max_delay` | 1 | float |
| `delay_rate` | 1 | float |
| `is_ever_delayed` | 1 | int(0/1) |
| `avg_review_score` | 2 | float |
| `last_review_score` | 2 | float |
| `has_low_review` | 2 | int(0/1) |
| `unique_categories` | 3 | int |
| `category_diversity_ratio` | 3 | float |
