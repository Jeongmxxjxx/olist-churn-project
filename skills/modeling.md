# Skill: 모델링 및 평가

## 모델 구성
| 단계 | 모델 | 목적 |
|------|------|------|
| 베이스라인 | Logistic Regression | 피처 방향성 검증 |
| 주력 | LightGBM | 성능 최적화 |
| 비교 | XGBoost | 앙상블 후보 |

---

## 1. 베이스라인: Logistic Regression
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)

y_pred_proba = lr.predict_proba(X_test_scaled)[:, 1]
print("AUC-ROC:", roc_auc_score(y_test, y_pred_proba))
print(classification_report(y_test, lr.predict(X_test_scaled)))
```

---

## 2. 주력: LightGBM + Stratified K-Fold
```python
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

X_all = pd.concat([X_train, X_test])   # 전체 피처 (스케일링 불필요)
y_all = pd.concat([y_train, y_test])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
    X_tr, X_val = X_all.iloc[tr_idx], X_all.iloc[val_idx]
    y_tr, y_val = y_all.iloc[tr_idx], y_all.iloc[val_idx]

    model = lgb.LGBMClassifier(
        class_weight='balanced',
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbose=-1
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    y_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_proba)
    auc_scores.append(auc)
    print(f"Fold {fold+1} AUC: {auc:.4f}")

print(f"\n평균 AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
```

---

## 3. 피처 중요도 시각화
```python
import matplotlib.pyplot as plt

# 마지막 fold 모델 기준
feat_imp = pd.DataFrame({
    'feature': X_all.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(feat_imp['feature'], feat_imp['importance'])
plt.title('피처 중요도 (LightGBM)')
plt.tight_layout()
plt.savefig('outputs/feature_importance.png', dpi=150)
plt.show()
```

---

## 4. 평가 지표 해석 기준

| 지표 | 기준값 | 해석 |
|------|--------|------|
| AUC-ROC | > 0.75 | 양호 / > 0.80 우수 |
| Recall(이탈) | > 0.70 | 실제 이탈 고객 탐지 능력 |
| Precision(이탈) | 참고용 | 낮아도 Recall 우선 |

> 이탈 예측은 **이탈 고객을 놓치는 비용**이 크므로 Recall을 우선 지표로 설정

---

## 5. 임계값(Threshold) 조정
```python
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

# F1 최대화 임계값
f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
best_thresh = thresholds[f1_scores.argmax()]
print(f"최적 임계값: {best_thresh:.3f}")

y_pred_adjusted = (y_pred_proba >= best_thresh).astype(int)
print(classification_report(y_test, y_pred_adjusted))
```

---

## 6. 결과 저장
```python
import joblib

joblib.dump(model, 'outputs/models/lgbm_churn.pkl')
joblib.dump(scaler, 'outputs/models/scaler.pkl')  # LR 베이스라인용

# 예측 결과 CSV
result_df = pd.DataFrame({
    'customer_unique_id': X_test_ids,
    'churn_probability': y_pred_proba,
    'predicted_label': y_pred_adjusted
})
result_df.to_csv('outputs/predictions.csv', index=False)
```
