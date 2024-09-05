import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('bank-full.csv', sep=';')

# List of columns to encode
columns_to_encode = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

# Convert categorical variables to numerical
le = LabelEncoder()
for col in columns_to_encode:
    if col in df.columns:
        df[col] = le.fit_transform(df[col])

# Convert target variable to binary (yes/no)
df['y'] = df['y'].map({'yes': 1, 'no': 0})

X = df.drop(['y'], axis=1)
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

feature_importance = clf.feature_importances_
feature_names = X_train.columns

print("\nFeature Importance:")
for feature, importance in zip(feature_names, feature_importance):
    print(f"{feature}: {importance:.3f}")

# Create a figure and axes object
fig, ax = plt.subplots(figsize=(20, 10))

# Draw the tree
plot_tree(clf, ax=ax, feature_names=X_train.columns, class_names=['no', 'yes'], filled=True)

# Save the figure
plt.savefig('decision_tree.png')
print("Decision tree saved as 'decision_tree.png' in the current directory")

# Optionally, display the plot
plt.show()

# Close the figure to free up memory
plt.close(fig)
