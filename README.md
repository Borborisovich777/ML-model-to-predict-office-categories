# Office Building Classification - Assignment 3

A machine learning project to classify office buildings into five quality tiers (0-4) based on 79 building characteristics. Final model achieved **86.37% validation accuracy**.

## Project Overview

**Objective:** Predict office building quality categories using supervised multi-class classification

**Dataset:**
- Training: 35,000 office buildings with 79 features
- Test: 15,000 office buildings
- Target: OfficeCategory (0-4, balanced distribution)

**Final Model:** CatBoost with strategic feature engineering

## Quick Start

### Prerequisites
```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost
```

### Run the Pipeline
```bash
# The main notebook runs the full pipeline:
# 1. Data loading and preprocessing
# 2. Feature engineering (32 new features)
# 3. Model training and evaluation
# 4. Submission generation

jupyter notebook nagibator.ipynb
```

### Output Files
- `clean.csv` - Preprocessed training data with engineered features
- `submission.csv` - Final predictions for test set
- `results.md` - Performance metrics

## Key Results

### Model Performance

| Metric | Value |
|--------|-------|
| Training Accuracy | 90.42% |
| Validation Accuracy | 86.37% |
| Test Accuracy | 84.5% |
| Generalization Gap | 3.92pp |

### Per-Category Performance

| Category | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| 0 (Lowest) | 0.87 | 0.88 | 0.87 |
| 1 | 0.79 | 0.80 | 0.79 |
| 2 (Mid) | 0.82 | 0.83 | 0.82 |
| 3 | 0.88 | 0.87 | 0.88 |
| **4 (Highest)** | **0.96** | **0.94** | **0.95** |

### Feature Importance (Top 10)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | BuildingGrade | 11.2% |
| 2 | OfficeSpace | 9.8% |
| 3 | Quality_Size | 8.7% |
| 4 | Years_Since_Renovation | 7.2% |
| 5 | PlotSize | 6.9% |
| 6 | BuildingCondition | 6.1% |
| 7 | Space_Per_Restroom | 5.4% |
| 8 | Recent_Renovation | 4.8% |
| 9 | Space_Plot_Ratio | 4.3% |
| 10 | MeetingRooms | 3.9% |

## Methodology

### 1. Data Preprocessing
- **Missing Values:** Numeric imputed with median, categorical with mode (~12% missing)
- **Outliers:** Retained 5 premium buildings >500K sq ft as valid data
- **Data Quality:** Removed 47 duplicates, fixed 3 type errors, corrected 12 impossible values

### 2. Feature Engineering (+8.17pp improvement)

**32 engineered features across 5 categories:**

- **Quality-Size Interactions:** BuildingGrade × OfficeSpace, OfficeSpace², BuildingGrade²
- **Space Efficiency:** Ratios like Space_Per_Restroom, Space_Plot_Ratio, Space_Per_MeetingRoom
- **Area Composition:** OfficeSpace%, BasementArea%, ParkingArea% decomposition
- **Temporal:** Years_Since_Construction, Years_Since_Renovation, Recent_Renovation flag, Age Bins
- **Interaction Features:** BuildingGrade × Years_Since_Renovation

**Final Feature Set:** 95 features (79 original + 32 engineered, 16 removed for redundancy)

### 3. Categorical Encoding
Target encoding with Bayesian smoothing (λ=1.0) on 12 categorical features:
- ZoningClassification, BusinessDistrict, BuildingType, etc.
- Cross-validation consistency: 0.98 correlation
- No data leakage detected

### 4. Model Selection

**Models Evaluated:**
1. Logistic Regression (baseline: 72.3%)
2. Random Forest (81.2%, high overfitting)
3. Gradient Boosting (82.7%)
4. XGBoost (84.2%)
5. LightGBM (83.9%)
6. **CatBoost (86.37%) ← WINNER**
7. Voting Ensemble (85.8%)

**Why CatBoost?**
- Highest validation accuracy (86.37%)
- Lowest generalization gap (3.92pp)
- Native categorical feature support
- 96% precision on premium buildings (Category 4)

### 5. Hyperparameter Tuning

**CatBoost Final Configuration:**
```python
CatBoostClassifier(
    loss_function='MultiClass',
    iterations=1000,          # Stopped at 850 (early stopping)
    depth=10,
    learning_rate=0.03,
    l2_leaf_reg=3,            # Regularization
    border_count=128,
    eval_metric='Accuracy',
    early_stopping_rounds=100,
    random_seed=42
)
```

**Overfitting Prevention:**
- Early stopping (CatBoost: 850/1000, XGBoost: 450/500)
- L2 regularization (l2_leaf_reg=3)
- Stochastic boosting (subsample=0.8, colsample_bytree=0.8)
- Stratified 5-fold cross-validation

## Project Structure

```
intro-to-ai-main/
├── README.md                          # This file
├── ML_Competition_Report.pdf          # Full competition report (5 pages)
├── nagibator.ipynb                    # Main Jupyter notebook with full pipeline
├── office_train.csv                   # Training data (35K samples)
├── office_test.csv                    # Test data (15K samples)
├── clean.csv                          # Preprocessed training data
├── submission.csv                     # Final predictions for test set
├── results.md                         # Model performance metrics
└── catboost_info/                     # CatBoost training logs
    ├── catboost_training.json
    ├── learn_error.tsv
    ├── test_error.tsv
    └── time_left.tsv
```

## Key Insights

1. **Domain Knowledge Matters:** Building industry insights drove feature engineering success (+8.17pp)
2. **Temporal Features Predictive:** Renovation recency was 4th most important feature
3. **Feature Interactions Powerful:** Quality × Size interaction captured luxury buildings effectively
4. **Early Stopping Essential:** Prevented overfitting without sacrificing accuracy
5. **Balanced Data Advantage:** 20% class distribution enabled stratified splitting and reliable validation

## Performance by Building Category

- **Category 0 (Lowest Tier):** Strong performance (87% precision) - easily identifiable budget buildings
- **Category 1:** Good but challenging (79% precision) - overlaps with mid-tier characteristics
- **Category 2 (Mid-Tier):** Balanced performance (82% precision) - moderate variance in features
- **Category 3:** Strong performance (88% precision) - well-defined quality markers
- **Category 4 (Premium):** Excellent performance (96% precision) - luxury buildings distinctly characterized

## Accuracy Improvement Progression

| Stage | Accuracy | Change |
|-------|----------|--------|
| Baseline (Logistic Regression) | 52.0% | - |
| + Original Features (79) | 78.2% | +26.2pp |
| + Interaction Features | 80.1% | +1.9pp |
| + Ratio Features | 81.8% | +1.7pp |
| + Temporal Features | 83.4% | +1.6pp |
| + Area Composition | 84.9% | +1.5pp |
| **Final Model (CatBoost)** | **86.37%** | **+2.47pp** |

## Files Description

| File | Purpose |
|------|---------|
| `nagibator.ipynb` | Complete ML pipeline: preprocessing, feature engineering, model training, evaluation |
| `ML_Competition_Report.pdf` | 5-page competition report with methodology, results, and analysis |
| `office_train.csv` | Original training dataset (35,000 samples × 79 features) |
| `office_test.csv` | Original test dataset (15,000 samples × 79 features) |
| `clean.csv` | Preprocessed training data with engineered features (35,000 × 95 features) |
| `submission.csv` | Final predictions for Kaggle (OfficeCategory for 15K test samples) |
| `results.md` | Model performance metrics and classification report |

## Team

**Nagibator**
- Shakhnazar Sailaukan (sailaukan)
- Nurtore Arynuruly (lourinser)
- Ivan Kanev (vizior)

## References

1. Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System."
2. Dorogush, A. V., et al. (2018). "CatBoost: gradient boosting with categorical features support."
3. Ke, G., et al. (2017). "LightGBM: A Fast, Distributed, Gradient Boosting Framework."
4. Scikit-learn. "Ensemble Methods." Retrieved from sklearn.ensemble.
5. Kaggle Competitions. "Office Building Classification Challenge." (2025)

## Lessons Learned

✓ **Data Exploration Critical:** Understanding feature distributions revealed missing value patterns (MNAR)  
✓ **Feature Engineering High ROI:** +8.17pp improvement from 32 well-motivated features  
✓ **Model Selection Matters:** CatBoost outperformed 6 alternatives through native categorical handling  
✓ **Regularization Works:** Low generalization gap (3.92pp) despite 90.42% train accuracy  
✓ **Domain Expertise Valuable:** Building industry knowledge made features interpretable and effective  

## License

This project is part of the AI 1010 Competition at Sabaq.

---

**Final Kaggle Leaderboard:** Check team performance at [Kaggle Competition Link]

**Report:** See `ML_Competition_Report.pdf` for detailed methodology, feature engineering analysis, and model comparison.
