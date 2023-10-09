# email_spam_detection

- This dataset contains 5172 rows of email data, each classified as spam or ham. The dataset is in CSV format, with the following columns:

- Email name: A unique identifier for each email.
- Label: The classification of the email, either 1 for spam or 0 for ham.
- Features: 3000 features representing the most common words in all the emails, after excluding non-alphabetical characters/words.
- 
## Usage

- This dataset can be used to train and evaluate machine learning models for email spam detection. To use the dataset, you can follow these steps:

- Split the dataset into train and test sets.
- Train a machine learning model on the train set.
- Evaluate the model on the test set to assess its performance.
- You can use a variety of machine learning algorithms for email spam detection, such as Naive Bayes, Logistic Regression, Support Vector Machines, and Random Forests.

-Example

- Here is an example of how to use the dataset to train a Naive Bayes model for email spam detection using Python and other models are attached within this repository :

Python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

## Load the dataset
df = pd.read_csv('email_spam_detection.csv')

## Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df[['Feature 1', 'Feature 2', ..., 'Feature 3000']], df['Label'], test_size=0.25, random_state=42)

## Train a Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

## Evaluate the model on the test set
y_pred = model.predict(X_test)
accuracy = (y_pred == y_test).mean()

print('Accuracy:', accuracy)

- This code will train a Naive Bayes model on the train set and evaluate it on the test set. The output of the code will be the accuracy of the model, which is the percentage of emails that the model correctly classified as spam or ham.

- Conclusion
This dataset can be used to train and evaluate machine learning models for email spam detection. The dataset is in CSV format and contains 5172 rows of email data, each classified as spam or ham.
