# Skill: EDA 가설2 — 낮은 리뷰 점수와 이탈

> **가설**: 최근 주문에서 낮은 리뷰 점수(1~2점)를 남긴 고객은 90일 이내에 다시 구매하지 않을 것이다.

선행 조건: `eda_common.md` 실행 완료 (orders, reviews, customers 로드된 상태)
결과 저장: `notebooks/01_eda_hypothesis2.ipynb`

---

## Step 1. review_score 기본 분포
```python
print(reviews['review_score'].value_counts().sort_index())
print(f"\n낮은 점수(1~2점) 비율: {(reviews['review_score'] <= 2).mean():.1%}")

reviews['review_score'].value_counts().sort_index().plot(
    kind='bar', figsize=(8, 6), edgecolor='white', color='steelblue'
)
plt.title('전체 리뷰 점수 분포')
plt.xlabel('리뷰 점수')
plt.ylabel('건수')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('outputs/eda_h2_review_distribution.png', dpi=150)
plt.show()
```

---

## Step 2. delivered 주문 기준 리뷰 결측률 확인
```python
delivered = orders[orders['order_status'] == 'delivered']

order_review = delivered.merge(
    reviews[['order_id', 'review_score']], on='order_id', how='left'
)
missing_rate = order_review['review_score'].isnull().mean()
print(f"리뷰 결측률: {missing_rate:.1%}")
print(f"리뷰 있는 주문: {order_review['review_score'].notna().sum():,}건")
print(f"리뷰 없는 주문: {order_review['review_score'].isnull().sum():,}건")
```

> 결측률이 높으면 전처리 단계에서 결측 플래그 피처(`review_missing_flag`) 추가 검토

---

## Step 3. 고객별 마지막 리뷰 점수 추출
```python
# order_purchase_timestamp 기준 정렬 후 마지막 리뷰 추출
order_review_ts = delivered.merge(
    reviews[['order_id', 'review_score']], on='order_id', how='left'
).merge(customers, on='customer_id')

# 고객별 가장 최근 주문의 리뷰 점수
last_review = (
    order_review_ts
    .sort_values('order_purchase_timestamp')
    .groupby('customer_unique_id')
    .agg(last_review_score=('review_score', 'last'),
         avg_review_score =('review_score', 'mean'))
    .reset_index()
)

print(last_review['last_review_score'].value_counts().sort_index())
```

---

## Step 4. 마지막 리뷰 점수 × 이탈률 비교
```python
cutoff = orders['order_purchase_timestamp'].max() - pd.Timedelta(days=90)
after  = orders.merge(customers, on='customer_id')
after  = after[after['order_purchase_timestamp'] > cutoff]
retained = set(after['customer_unique_id'])

last_review['churned'] = (~last_review['customer_unique_id'].isin(retained)).astype(int)

# 점수별 이탈률
churn_by_score = last_review.groupby('last_review_score')['churned'].mean()
print(churn_by_score)

churn_by_score.plot(
    kind='bar', figsize=(10, 6), edgecolor='white', color='steelblue'
)
plt.title('마지막 리뷰 점수별 이탈률')
plt.xlabel('리뷰 점수')
plt.ylabel('이탈률')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('outputs/eda_h2_churn_by_score.png', dpi=150)
plt.show()
```

---

## Step 5. 저점수(1~2점) vs 고점수(4~5점) 이탈률 비교
```python
last_review['score_group'] = pd.cut(
    last_review['last_review_score'],
    bins=[0, 2, 3, 5],
    labels=['저점수(1~2)', '중간(3)', '고점수(4~5)']
)

churn_by_group = last_review.groupby('score_group')['churned'].mean()
print(churn_by_group)

churn_by_group.plot(
    kind='bar', figsize=(8, 6),
    color=['tomato', 'gold', 'steelblue'], edgecolor='white'
)
plt.title('리뷰 점수 그룹별 이탈률')
plt.ylabel('이탈률')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('outputs/eda_h2_churn_by_score_group.png', dpi=150)
plt.show()
```

---

## 체크 포인트

| 확인 항목 | 판단 기준 |
|-----------|-----------|
| 리뷰 결측률 | 20% 초과 시 결측 플래그 피처 필수 추가 |
| 저점수 vs 고점수 이탈률 차이 | 5%p 이상이면 가설 유효 |
| 점수별 이탈률 단조 감소 여부 | 점수↑ → 이탈률↓ 패턴 확인 |

→ 단조 감소 패턴이 없으면 점수를 연속값 대신 `has_low_review` 이진 피처로만 사용 검토
