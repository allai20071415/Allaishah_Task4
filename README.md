# Task 4 — Play Store Sentiment Analysis

Machine-learning project that classifies Google Play Store user reviews as **Positive, Neutral, or Negative**.

## Dataset

Use the Kaggle dataset/notebook provided for this assignment:

https://www.kaggle.com/code/mmmarchetti/play-store-sentiment-analysis-of-user-reviews

Download the user-review CSV and place it in:

```text
data/User Reviews.csv
```

The dataset is **not redistributed** in this repository. The program automatically detects common review/sentiment column names.

## Project structure

```text
Task4_PlayStore_Sentiment_GitHub/
├── data/
│   └── User Reviews.csv
├── notebooks/
│   └── sentiment_analysis.ipynb
├── src/
│   └── sentiment_analysis.py
├── outputs/
├── requirements.txt
├── .gitignore
└── README.md
```

## Run on your computer

### 1. Clone your GitHub repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Task4_PlayStore_Sentiment_GitHub
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Add the dataset

Download the CSV from the Kaggle source above and put it inside `data/`.

### 4. Run the Python project

```bash
python src/sentiment_analysis.py
```

### 5. Or run the notebook

```bash
jupyter notebook notebooks/sentiment_analysis.ipynb
```

Run all cells.

## What the project does

1. Loads the Play Store review dataset.
2. Detects the review and sentiment columns.
3. Removes missing and duplicate reviews.
4. Checks Positive/Neutral/Negative class distribution.
5. Preprocesses text:
   - lowercase
   - URL/mention removal
   - punctuation removal
   - tokenisation
   - stopword removal
   - lemmatisation
6. Splits data using an 80/20 stratified split.
7. Converts text to TF-IDF features.
8. Trains:
   - Multinomial Naive Bayes
   - Logistic Regression
9. Calculates accuracy, precision, recall and F1-score.
10. Creates confusion matrices.
11. Creates sentiment distribution and WordCloud visualisations.
12. Shows five misclassified reviews.
13. Saves all generated results to `outputs/`.

## TF-IDF

TF-IDF (Term Frequency–Inverse Document Frequency) converts text into numerical features. It gives more weight to terms that are important to a particular review while reducing the weight of terms that occur across many reviews.

## Evaluation

The models are compared using:

- Accuracy
- Precision
- Recall
- F1-score
- Classification report
- Confusion matrix

The final recommended model is selected from the **actual F1-score produced when the Kaggle dataset is run**, rather than using hard-coded results.

## Outputs

The script creates:

```text
outputs/
├── sentiment_distribution.png
├── positive_wordcloud.png
├── neutral_wordcloud.png
├── negative_wordcloud.png
├── naive_bayes_confusion_matrix.png
├── logistic_regression_confusion_matrix.png
├── model_comparison.png
├── model_comparison.csv
├── error_analysis.csv
└── run_summary.txt
```

## Real-world applications

- Google Play app review monitoring
- Customer feedback analysis
- Detecting negative user experiences
- App quality improvement
- Feature-request analysis
- Product and service satisfaction tracking

## Limitations

TF-IDF models do not fully understand sarcasm, complex negation, emojis, slang or long-range context. Production sentiment systems can use transformer models such as BERT and domain-specific training data.

## References

- Kaggle Play Store Sentiment Analysis: https://www.kaggle.com/code/mmmarchetti/play-store-sentiment-analysis-of-user-reviews
- NLTK: https://www.nltk.org/
- scikit-learn: https://scikit-learn.org/
- TextBlob: https://textblob.readthedocs.io/
