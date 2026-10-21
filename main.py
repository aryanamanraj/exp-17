
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# 1. Load Titanic dataset
df = sns.load_dataset("titanic")

# 2. Select required columns
df = df[
    ["survived", "pclass", "sex", "age",
     "sibsp", "parch", "fare"]
].copy()

# 3. Handle missing values
df["age"] = df["age"].fillna(df["age"].median())
df["fare"] = df["fare"].fillna(df["fare"].median())

# 4. Convert sex into numeric values
df["sex"] = df["sex"].map({"male": 0, "female": 1})

# 5. Separate input and output
X = df[
    ["pclass", "sex", "age", "sibsp", "parch", "fare"]
]
y = df["survived"]

# 6. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,
    random_state=42, stratify=y
)

# 7. Train Decision Tree model
model = DecisionTreeClassifier(
    max_depth=4, random_state=42
)
model.fit(X_train, y_train)

# 8. Make predictions
y_pred = model.predict(X_test)

# 9. Evaluate model
print("Accuracy:",
      round(accuracy_score(y_test, y_pred) * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["Did Not Survive", "Survived"]
))

# 10. Plot confusion matrix
sns.heatmap(
    confusion_matrix(y_test, y_pred),
    annot=True, fmt="d", cmap="Blues"
)
plt.title("Titanic Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 11. Visualize Decision Tree
plt.figure(figsize=(18, 8))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Did Not Survive", "Survived"],
    filled=True, rounded=True
)
plt.title("Titanic Survival Decision Tree")
plt.show()

# 12. Predict for a new passenger
new_passenger = pd.DataFrame({
    "pclass": [2],
    "sex": [1],
    "age": [25],
    "sibsp": [0],
    "parch": [0],
    "fare": [20.0]
})

prediction = model.predict(new_passenger)[0]

print("\nNew Passenger Prediction:",
      "Survived" if prediction == 1
      else "Did Not Survive")
