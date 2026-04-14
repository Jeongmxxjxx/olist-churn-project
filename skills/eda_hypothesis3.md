# Skill: EDA 가설3 — 카테고리 다양성과 유지율

> **가설**: 한 가지 카테고리의 상품만 구매한 고객보다 여러 카테고리를 이용한 고객의 유지율이 높을 것이다.

선행 조건: `eda_common.md` 실행 완료 (orders, items, products, customers 로드된 상태)
결과 저장: `notebooks/01_eda_hypothesis3.ipynb`

---

## Step 1. 카테고리 기본 현황
```python
print(f"전체 카테고리 수: {products['product_category_name'].nunique()}")
print(f"카테고리 결측률: {products['product_category_name'].isnull().mean():.1%}")

# 카테고리별 주문 빈도 상위 20개
items_cat = items.merge(
    products[['product_id', 'product_category_name']], on='product_id', how='left'
)
items_cat['product_category_name'] = items_cat['product_category_name'].fillna('unknown')

top_categories = (
    items_cat.groupby('product_category_name')['order_id']
    .nunique()
    .sort_values(ascending=False)
    .head(20)
)
print(top_categories)
```

---

## Step 2. 고객별 구매 카테고리 수 분포
```python
delivered = orders[orders['order_status'] == 'delivered']

order_cat = delivered.merge(customers, on='customer_id').merge(
    items_cat[['order_id', 'product_category_name']], on='order_id', how='left'
)

cat_per_customer = (
    order_cat.groupby('customer_unique_id')['product_category_name']
    .nunique()
    .reset_index()
    .rename(columns={'product_category_name': 'unique_categories'})
)

print(cat_per_customer['unique_categories'].value_counts().sort_index())
print(f"\n1개 카테고리만 구매한 고객 비율: {(cat_per_customer['unique_categories'] == 1).mean():.1%}")

cat_per_customer['unique_categories'].value_counts().sort_index().plot(
    kind='bar', figsize=(10, 6), edgecolor='white', color='steelblue'
)
plt.title('고객별 구매 카테고리 수 분포')
plt.xlabel('카테고리 수')
plt.ylabel('고객 수')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('outputs/eda_h3_category_distribution.png', dpi=150)
plt.show()
```

> **주의**: Olist는 대부분 고객이 1회 구매 → 카테고리 수도 대부분 1개일 가능성 높음.
> 분포 확인 후 피처 유효성 판단 필요.

---

## Step 3. 카테고리 수 × 이탈률 비교
```python
cutoff = orders['order_purchase_timestamp'].max() - pd.Timedelta(days=90)
after  = orders.merge(customers, on='customer_id')
after  = after[after['order_purchase_timestamp'] > cutoff]
retained = set(after['customer_unique_id'])

cat_per_customer['churned'] = (~cat_per_customer['customer_unique_id'].isin(retained)).astype(int)

churn_by_cat = cat_per_customer.groupby('unique_categories')['churned'].mean()
print(churn_by_cat)

churn_by_cat.plot(
    kind='bar', figsize=(10, 6), edgecolor='white', color='steelblue'
)
plt.title('구매 카테고리 수별 이탈률')
plt.xlabel('카테고리 수')
plt.ylabel('이탈률')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('outputs/eda_h3_churn_by_category.png', dpi=150)
plt.show()
```

---

## Step 4. 단일 vs 다중 카테고리 이탈률 비교
```python
cat_per_customer['is_multi_category'] = (cat_per_customer['unique_categories'] > 1).astype(int)

churn_by_multi = cat_per_customer.groupby('is_multi_category')['churned'].mean()
churn_by_multi.rename({0: '단일 카테고리', 1: '다중 카테고리'}).plot(
    kind='bar', figsize=(8, 6),
    color=['tomato', 'steelblue'], edgecolor='white'
)
plt.title('단일 vs 다중 카테고리 구매 고객 이탈률')
plt.ylabel('이탈률')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('outputs/eda_h3_churn_single_vs_multi.png', dpi=150)
plt.show()

print("단일 카테고리 이탈률:", f"{churn_by_multi[0]:.1%}")
print("다중 카테고리 이탈률:", f"{churn_by_multi[1]:.1%}")
print("다중 카테고리 고객 수:", cat_per_customer['is_multi_category'].sum())
```

---

## 체크 포인트

| 확인 항목 | 판단 기준 |
|-----------|-----------|
| 1개 카테고리 고객 비율 | 90% 초과 시 피처 분산 낮음 → 가설 유효성 재검토 |
| 단일 vs 다중 이탈률 차이 | 5%p 이상이면 가설 유효 |
| 다중 카테고리 고객 표본 수 | 너무 적으면 신뢰도 낮음 (n < 100 주의) |

→ 1개 카테고리 고객이 압도적으로 많아 분산이 없다면 `unique_categories` 피처 제거 검토,
  `category_diversity_ratio` 대신 `is_multi_category` 이진 피처로 대체
