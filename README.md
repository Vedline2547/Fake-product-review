# Fake Product Review Detection

A machine learning project that classifies product reviews as **Computer Generated (CG)** or **Original (OR)** using Natural Language Processing (NLP) and Logistic Regression.

## 📌 Project Overview

Online product reviews play an important role in helping customers make purchasing decisions. However, fake or computer-generated reviews can mislead customers and affect the reliability of online marketplaces.

This project uses **Natural Language Processing (NLP)** to analyze the text of product reviews and classify them into two categories:

* **CG** — Computer Generated
* **OR** — Original review

The project uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert review text into numerical features and **Logistic Regression** as the classification algorithm.

## 🎯 Objectives

* Load and explore a product review dataset.
* Analyze the distribution of review labels.
* Convert text reviews into numerical features using TF-IDF.
* Train a Logistic Regression classification model.
* Predict whether reviews are computer-generated or original.
* Evaluate the model using accuracy and a classification report.

## 📂 Dataset

The dataset contains product reviews with the following main columns:

| Column     | Description                          |
| ---------- | ------------------------------------ |
| `category` | Product category                     |
| `rating`   | Product rating                       |
| `label`    | Review classification (`CG` or `OR`) |
| `text_`    | Review text                          |

The `text_` column is used as the input feature, while `label` is used as the target variable.

## 🛠️ Technologies Used

* Python
* Pandas
* Scikit-learn
* Natural Language Processing (NLP)
* TF-IDF
* Logistic Regression

## 🔄 Machine Learning Workflow

```text
Dataset
   ↓
Data Exploration
   ↓
Select Review Text and Labels
   ↓
Train/Test Split
   ↓
TF-IDF Vectorization
   ↓
Logistic Regression
   ↓
Predictions
   ↓
Model Evaluation
```

## 🧠 TF-IDF Vectorization

TF-IDF is used to transform the text reviews into numerical representations that can be processed by a machine learning algorithm.

The vectorizer is configured with:

```python
TfidfVectorizer(
    stop_words="english",
    max_df=0.7
)
```

English stop words are removed, and terms appearing in more than 70% of the documents are ignored.

## 🤖 Machine Learning Model

The project uses **Logistic Regression** for binary text classification.

```python
model = LogisticRegression()

model.fit(X_train_tfidf, y_train)
```

The model learns patterns in the review text that help distinguish between computer-generated and original reviews.

## 📊 Train/Test Split

The dataset is divided into:

* **80%** training data
* **20%** testing data

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

A `random_state` of `42` is used to make the split reproducible.

## 📈 Model Evaluation

The model is evaluated using:

### Accuracy

Accuracy measures the proportion of predictions that the model classified correctly.

```python
accuracy_score(y_test, y_pred)
```

### Classification Report

The classification report provides:

* Precision
* Recall
* F1-score
* Support

```python
classification_report(y_test, y_pred)
```

After running the project, the resulting accuracy and classification report can be added to this section.

## 📁 Project Structure

```text
Fake-product-review/
│
├── main.py
├── fake reviews dataset.csv
├── README.md
└── .gitignore
```

> **Note:** For large datasets, consider excluding the dataset from GitHub and providing instructions for downloading it instead.

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Vedline2547/Fake-product-review.git
```

### 2. Navigate into the project

```bash
cd Fake-product-review
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install the required libraries

```bash
pip install pandas scikit-learn
```

### 6. Run the project

```bash
python main.py
```

## 📌 Example Output

```text
Dataset Sample:

             category  rating label                                              text_
0  Home_and_Kitchen_5     5.0    CG     Love this! Well made, sturdy...
1  Home_and_Kitchen_5     5.0    CG     love it, a great upgrade...
2  Home_and_Kitchen_5     5.0    CG     This pillow saved my back...

Unique Labels:
['CG' 'OR']

Accuracy: 0.XX

Classification Report:
              precision    recall  f1-score   support

          CG       ...
          OR       ...

    accuracy       ...
```

The exact evaluation results depend on the dataset and train/test split.

## 🚀 Future Improvements

Possible improvements include:

* Compare Logistic Regression with Naive Bayes, SVM, Random Forest, and other classifiers.
* Perform hyperparameter tuning.
* Add text preprocessing such as stemming or lemmatization.
* Analyze class imbalance.
* Add a confusion matrix.
* Experiment with different TF-IDF parameters.
* Use word and character n-grams.
* Build a Flask web application for real-time review classification.
* Deploy the trained model as a web service.
* Experiment with transformer-based NLP models such as BERT.

## 📚 Skills Demonstrated

* Python
* Pandas
* Data preprocessing
* Natural Language Processing
* Text classification
* Feature engineering
* TF-IDF
* Logistic Regression
* Model evaluation
* Scikit-learn

## 👨‍💻 Author

**Vedline Ochieng**

Civil Engineering Student | ML & AI Enthusiast | Python Developer

GitHub: [Vedline2547](https://github.com/Vedline2547)
