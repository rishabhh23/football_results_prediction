from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import joblib  # to save the model

def train_model(results_df):
    # Select features and target
    X = results_df[['home_team', 'away_team', 'home_score', 'away_score', 'neutral', 'tournament']]
    y = results_df['outcome']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    # Save model
    joblib.dump(model, "./models/random_forest_model.pkl")
    print("Model saved as random_forest_model.pkl")

# Example usage
# from data_preprocessing import load_data, preprocess_results
# results_df, shootouts_df, _ = load_data()
# results_df = preprocess_results(results_df, shootouts_df)
# train_model(results_df)
