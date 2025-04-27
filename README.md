# 🛒 Amazon Review Rating Prediction (1.7M+ Samples)

This project predicts Amazon review scores by combining TF-IDF text features with sentiment analysis, trained on over 1.7 million samples.

---

## 🛠️ Project Structure

- **data/**
  - `train.csv` — Labeled and unlabeled training data
  - `submission.csv` — Initial model predictions
  - `final_submission.csv` — Final modified predictions after post-processing

---

## 🚀 Approach

1. **Data Preprocessing**
   - Separated labeled and unlabeled review data.
   - Applied TF-IDF vectorization on review texts (`max_features=5000`).
   - Extracted sentiment polarity and subjectivity using **TextBlob**.

2. **Feature Engineering**
   - Combined sparse TF-IDF features with dense sentiment features for hybrid modeling.

3. **Modeling**
   - Trained a **Logistic Regression** model (`multi_class='multinomial'`) on the engineered features.
   - Used full training data (80/20 split for validation).

4. **Post-Processing**
   - Applied minor adjustments to predictions based on problem-specific rules to fine-tune competition performance.

---

## 📊 Results

- **Validation Accuracy:** ~64%
- Significant performance gains achieved by adding sentiment features on top of pure text vectorization.
- Final submissions prepared following Kaggle competition format.

---

## 🧩 How to Run

1. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn textblob matplotlib seaborn
