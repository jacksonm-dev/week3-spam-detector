# Train a spam classifier and see how good it is
# Run prepare_data.py first

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

data = pd.read_csv("data/features.csv")

features = ["word_count", "exclamations", "money_words", "all_caps",
            "urgency", "call_to_action"]
X = data[features]   # the numbers the model sees
y = data["label"]    # the answers: "spam" or "legit"


# Part 2: split the data and train the model
# 75% of the emails to learn from, 25% kept back to test it
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))


# Part 3.1: confusion matrix

print("\nConfusion matrix (rows = real answer, columns = model's guess)")
print("        spam  legit")
print(confusion_matrix(y_test, predictions, labels=["spam", "legit"]))


# Part 3.2: precision and recall

print("\nPrecision:", precision_score(y_test, predictions, pos_label="spam"))
print("Recall:", recall_score(y_test, predictions, pos_label="spam"))


# Part 3.4: feature weights

# Positive weight = pushes towards spam, negative = pushes towards legit
print("\nFeature weights:")
for name, weight in zip(features, model.coef_[0]):
    print(name, round(weight, 2))


# Part 3.5: try to fool the model

my_emails = pd.read_csv("data/attack_emails.csv")
guesses = model.predict(my_emails[features])

print("\nMy emails:")
for i in range(len(my_emails)):
    print(my_emails["text"][i])
    print("   really:", my_emails["label"][i], "   model says:", guesses[i])


# Part 3.6: are copies of the same email in both train and test?

train_ids = data["source_id"][X_train.index].unique()
test_ids = data["source_id"][X_test.index].unique()

count = 0
for email_id in test_ids:
    if email_id in train_ids:
        count = count + 1

print("\nEmails with copies in both training and test:", count, "out of 16")