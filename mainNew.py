import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import naive_bayes, model_selection, feature_extraction, metrics, base
from sklearn.model_selection import GridSearchCV
from joblib import dump

path = "./Dados/combined_data.csv"

df = pd.read_csv(path)
df.rename(columns={'label':'type','text':'email'}, inplace= True)

labels = {0 : "Not Spam", 1 : "Spam"}
label_counts = df['type'].value_counts()

# X are the features in this case "email"
emails = df.drop('type', axis = 1).values

# y are the labels in this "type" case where 1 is Spam and 0 Ham
types = df['type'].values

# Mostrar Contagem de Ham e Spam
plt.pie(label_counts, labels = labels.values(), autopct = "%.2f%%")
plt.show()

# Check and remove null values
null_values = df.isna().sum().sum()
print(f"Number of null values: {null_values}")
if null_values > 0:
    df = df.dropna()
    print(f"Number of null values (after dropping null values): {df.isna().sum().sum()}")

# Check and remove duplicated values
dup_values = df.duplicated().sum()
print(f"Number of duplicated urls : {dup_values}")
if dup_values > 0:
    df.drop_duplicates(inplace = True)
    print(f"Number of duplicated urls (after dropping duplicates) : {df.duplicated().sum()}")


# Count Vectorization ( Converting text into features ) 
print("Generating features...")
vectorizer = feature_extraction.text.CountVectorizer()
emails = vectorizer.fit_transform(emails.reshape(-1))  
print("Total number of features :", len(vectorizer.get_feature_names_out()))

# Splitting the data into training and testing data
X_train, X_test, y_train, y_test = model_selection.train_test_split(emails, types, test_size = 0.15, stratify = types)

# Hyperparameter Tuning using GridSearchCV
param_grid = {'alpha': [0.1, 0.5, 1.0, 1.5, 2.0]}
nb_model = naive_bayes.MultinomialNB()
grid_search = GridSearchCV(nb_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best hyperparameters
best_params = grid_search.best_params_
print(f"Best hyperparameters: {best_params}")

# Evaluation of the model with the best hyperparameters in the test set
best_model = grid_search.best_estimator_
accuracy = best_model.score(X_test, y_test)
print(f"Accuracy with Better Hyperparameters: {accuracy}")

# Pie Graphic
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.title("Training Set")
value_counts_train = pd.Series(y_train).value_counts()
plt.pie(value_counts_train, labels=labels.values(), autopct=lambda p: '{:.0f}'.format(p * sum(value_counts_train) / 100))

plt.subplot(2, 2, 2)
plt.title("Testing Set")
value_counts_test = pd.Series(y_test).value_counts()
plt.pie(value_counts_test, labels=labels.values(), autopct=lambda p: '{:.0f}'.format(p * sum(value_counts_test) / 100))

plt.subplot(2, 2, 3)
plt.title("Training Set")
plt.pie(pd.Series(y_test).value_counts(), labels=labels.values(), autopct="%.2f%%")

plt.subplot(2, 2, 4)
plt.title("Testing Set")
plt.pie(pd.Series(y_test).value_counts(), labels=labels.values(), autopct="%.2f%%")

plt.tight_layout()
plt.show()


#NB
models = [naive_bayes.MultinomialNB(alpha=best_params['alpha']), naive_bayes.BernoulliNB()]
scores = []

def evaluate(model):
    try:
        cross_val_scores = model_selection.cross_val_score(base.clone(model), X_train, y_train, cv=5, error_score='raise')
        print(f"\nCross Validation Scores for model {model} : ")
        print(cross_val_scores, cross_val_scores.mean())
        scores.append(cross_val_scores.mean())
    except Exception as e:
        print(f"Error evaluating model {model}: {str(e)}")
    

for model in models:
    evaluate(model)

# Training the Model
model = models[np.argmax(scores)]
print(f"Best Model : {model}")
model.fit(X_train, y_train)

# Making predictions
y_train_pred = model.predict(X_train)

# Evaluating the performance on train set
accuracy = np.sum(y_train == y_train_pred) / len(y_train)
print("Accuracy (Train) : ", accuracy)

# Confusion Matrix
col = ["Positive", "Negative"]
ln = ["Prediction Positive","Prediction Negative"]
cm = metrics.confusion_matrix(y_train, y_train_pred)

sns.heatmap(cm, annot = True, fmt='d', xticklabels = ln, yticklabels = col)
plt.show()

# Classification Report
clf_report = metrics.classification_report(y_train, y_train_pred)
print(clf_report)

# Model Validation
y_test_pred = model.predict(X_test)
print(y_test[:10], y_test_pred[:10])

# Evaluating the performance on test set
accuracy = np.sum(y_test == y_test_pred) / len(y_test)
print("Accuracy (Test) : ", accuracy)

# Confusion Matrix
cm = metrics.confusion_matrix(y_test, y_test_pred)
print("Heat Map: ")
sns.heatmap(cm, annot = True, fmt='d', xticklabels = ln, yticklabels = col)
plt.show()

# Classification Report
clf_report = metrics.classification_report(y_train, y_train_pred)
print(clf_report)

dump(model, "clf_model.h5")
dump(vectorizer, "count_vectorizer.h5")