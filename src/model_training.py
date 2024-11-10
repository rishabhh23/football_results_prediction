from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import joblib  # to save the model
from sklearn.preprocessing import OneHotEncoder

def train_model(results_df):
    # Separate features and target
    X = results_df[['home_team', 'away_team', 'home_score', 'away_score', 'neutral', 'tournament']]
    y = results_df['outcome']

    # OneHotEncode categorical features
    categorical_features = ['home_team', 'away_team', 'tournament']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
    X_encoded = encoder.fit_transform(X[categorical_features])

    # Combine encoded features with numeric features
    X_numeric = X[['home_score', 'away_score', 'neutral']].reset_index(drop=True)
    X_final = pd.concat([pd.DataFrame(X_encoded), X_numeric], axis=1)
    X_final.columns = X_final.columns.astype(str)  # Convert all column names to strings

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

    # Initialize and train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    # Save model and encoder for future use
    joblib.dump(model, "./models/random_forest_model.pkl")
    joblib.dump(encoder, "./models/encoder.pkl")
    print("Model and encoder saved.")
