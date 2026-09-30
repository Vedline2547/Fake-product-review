# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
# Load dataset
df = pd.read_csv(r"C:\Users\HP\Desktop\ML\Fake-product-review\fake reviews dataset.csv\fake reviews dataset.csv")
# Display the first few rows of the dataset
print("Dataset Sample:")
print(df.head())
# Define features and targets
X = df["text_"]
y = df['label']
# split dataset
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
# Initialize the TfidfVectorizer to convert text to numerical features
vectorizer = TfidfVectorizer(stop_words='english',max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
# Initialize logistic regression classifier
model = LogisticRegression()
# Train the model
model.fit(X_train_tfidf,y_train)
# Make predictions
y_pred = model.predict(X_test_tfidf)
# Evaluate the model
accuracy = accuracy_score(y_test,y_pred)
report = classification_report(y_test,y_pred)
print(f"\nAccuracy : {accuracy}")
print("\nClassification Report:\n",report)