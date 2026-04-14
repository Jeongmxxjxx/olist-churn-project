# Skill: EDA 가설1 — 배송 지연 경험과 이탈

> **가설**: 약속된 배송 예정일보다 실제 배송이 늦어진 고객은 이탈률이 높을 것이다.

선행 조건: `eda_common.md` 실행 완료 (orders, customers 로드된 상태)
결과 저장: `notebooks/01_eda_hypothesis1.ipynb`

---

## Step 1. delivered 주문 필터링 및 지연일 계산
```python
delivered = orders[
    (orders['order_status'] == 'delivered') &
    (orders['order_delivered_customer_date'].notna()) &
    (orders['order_estimated_delivery_date'].notna())
].copy()

delivered['delay_days'] = (
    delivered['order_delivered_customer_date'] -
    delivered['order_estimated_delivery_date']
).dt.days

print(delivered['delay_days'].describe())
print(f"\n지연된 주문 비율: {(delivered['delay_days'] > 0).mean():.1%}")
print(f"정시/빠른 배송 비율: {(delivered['delay_days'] <= 0).mean():.1%}")
```

---

## Step 2. 지연일 분포 시각화
```python
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 전체 분포
axes[0].hist(delivered['delay_days'], bins=60, edgecolor='white')
axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='기준선 (0일)')
axes[0].set_title('배송 지연일 분포')
axes[0].set_xlabel('지연일 (음수=빠름, 양수=늦음)')
axes[0].legend()

# 지연 주문만 확대
delayed_only = delivered[delivered['delay_days'] > 0]
axes[1].hist(delayed_only['delay_days'], bins=40, edgecolor='white', color='tomato')
axes[1].set_title(f'지연 주문 분포 (n={len(delayed_only):,})')
axes[1].set_xlabel('지연일')

plt.tight_layout()
plt.savefig('outputs/eda_h1_delay_distribution.png', dpi=150)
plt.show()
```

---

## Step 3. 지연 경험 여부 × 재구매율 비교
```python
# customer_unique_id 기준으로 집계
delivered_merged = delivered.merge(customers, on='customer_id')

# 고객별 지연 경험 여부
customer_delay = delivered_merged.groupby('customer_unique_id').agg(
    ever_delayed=('delay_days', lambda x: (x > 0).any()),
    order_count =('order_id', 'nunique')
).reset_index()

# cutoff 기준 재구매 여부 (레이블)
cutoff = orders['order_purchase_timestamp'].max() - pd.Timedelta(days=90)
after = orders.merge(customers, on='customer_id')
after = after[after['order_purchase_timestamp'] > cutoff]
retained = set(after['customer_unique_id'])

customer_delay['retained'] = customer_delay['customer_unique_id'].isin(retained).astype(int)
customer_delay['churned']  = 1 - customer_delay['retained']

# 그룹별 이탈률 비교
churn_by_delay = customer_delay.groupby('ever_delayed')['churned'].mean()
print("지연 경험 없음 이탈률:", f"{churn_by_delay[False]:.1%}")
print("지연 경험 있음 이탈률:", f"{churn_by_delay[True]:.1%}")

# 시각화
churn_by_delay.rename({False: '지연 없음', True: '지연 있음'}).plot(
    kind='bar', figsize=(8, 6), color=['steelblue', 'tomato'], edgecolor='white'
)
plt.title('배송 지연 경험 여부별 이탈률')
plt.ylabel('이탈률')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('outputs/eda_h1_churn_by_delay.png', dpi=150)
plt.show()
```

---

## Step 4. 지연 정도(구간)별 이탈률
```python
# 지연 구간 분류
bins   = [-999, 0, 3, 7, 14, 999]
labels = ['빠름', '0~3일', '3~7일', '7~14일', '14일 초과']

delivered_merged2 = delivered.merge(customers, on='customer_id').copy()
delivered_merged2['delay_group'] = pd.cut(
    delivered_merged2['delay_days'], bins=bins, labels=labels
)

cust_delay_group = delivered_merged2.groupby('customer_unique_id').agg(
    delay_group=('delay_group', lambda x: x.mode()[0])
).reset_index()
cust_delay_group['churned'] = (~cust_delay_group['customer_unique_id'].isin(retained)).astype(int)

churn_by_group = cust_delay_group.groupby('delay_group')['churned'].mean()
print(churn_by_group)

churn_by_group.plot(kind='bar', figsize=(10, 6), edgecolor='white')
plt.title('배송 지연 구간별 이탈률')
plt.ylabel('이탈률')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('outputs/eda_h1_churn_by_delay_group.png', dpi=150)
plt.show()
```

---

## 체크 포인트

| 확인 항목 | 판단 기준 |
|-----------|-----------|
| 지연된 주문 비율 | 피처 유효 범위 확인 |
| 지연 있음 vs 없음 이탈률 차이 | 5%p 이상이면 가설 유효 |
| 지연 구간별 이탈률 단조 증가 여부 | 가설 방향성 확인 |

→ 이탈률 차이가 미미하면 `delay_days` 연속값보다 `is_ever_delayed` 이진 피처 우선 검토
