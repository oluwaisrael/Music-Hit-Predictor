import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


print("--- Loading Spotify Tracks Dataset ---")
df = pd.read_csv("spotify_tracks_full.csv")
df = df[df['track_genre'] == 'afrobeat'].reset_index(drop=True)
print(f"Afrobeat tracks found: {len(df)}")


threshold = df['popularity'].median()
print(f"Popularity median (threshold): {threshold}")
df['is_hit'] = (df['popularity'] > threshold).astype(int)


drop_cols = ['Unnamed: 0', 'is_hit', 'popularity', 'track_id', 'artists', 'album_name', 'track_name', 'track_genre']
X = df.drop(columns=[c for c in drop_cols if c in df.columns])
y = df['is_hit']

X['explicit'] = X['explicit'].astype(int)

print(f"Features shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts(normalize=True)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


print("\n--- Training Random Forest Model ---")
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.dummy import DummyClassifier
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)
dummy_acc = accuracy_score(y_test, dummy.predict(X_test))

print("\n=== Model Evaluation Metrics ===")
print(f"Baseline (majority class) Accuracy: {dummy_acc:.4f}")
print(f"Random Forest Accuracy:             {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n--- Training Logistic Regression ---")
logreg = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
logreg.fit(X_train_scaled, y_train)
logreg_pred = logreg.predict(X_test_scaled)
logreg_acc = accuracy_score(y_test, logreg_pred)

print("\n=== Model Comparison ===")
print(f"Baseline (majority class):  {dummy_acc:.4f}")
print(f"Logistic Regression:        {logreg_acc:.4f}")
print(f"Random Forest:               {accuracy_score(y_test, y_pred):.4f}")
        
from sklearn.model_selection import cross_val_score

print("\n--- 5-Fold Cross-Validation ---")
rf_cv = cross_val_score(model, X, y, cv=5, scoring='accuracy')
logreg_cv = cross_val_score(logreg, scaler.fit_transform(X), y, cv=5, scoring='accuracy')

print(f"Random Forest CV:        {rf_cv.mean():.4f} (+/- {rf_cv.std():.4f})")
print(f"Logistic Regression CV:  {logreg_cv.mean():.4f} (+/- {logreg_cv.std():.4f})")

importances = model.feature_importances_
feature_imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)

print("\nTop Predictors for a Hit:")
print(feature_imp_df.head(10))
#
import joblib


joblib.dump(logreg, 'logreg_afrobeats_model.pkl')

joblib.dump(scaler, 'afrobeats_scaler.pkl')

print("\nModel and Scaler successfully saved to disk! Ready for Streamlit.")