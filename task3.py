import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load the dataset
df = pd.read_csv("bank-additional-full.csv", sep=';')

print("Dataset Shape:", df.shape)
print("\nSample Data:\n", df.head())

# Encode categorical features
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Separate features and target
X = df.drop("y", axis=1)
y = df["y"]

# SMOTE for balancing
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluation
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Optional: Plot a single decision tree from the forest
plt.figure(figsize=(20,10))
plot_tree(clf.estimators_[0], 
          feature_names=X.columns, 
          class_names=label_encoders['y'].classes_,
          filled=True, 
          max_depth=3)
plt.title("Random Forest - One Tree Visualization")
plt.savefig("random_forest_tree.png", dpi=300, bbox_inches='tight')

plt.show()
