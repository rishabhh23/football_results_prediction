import pandas as pd
import joblib

# Load the trained model and encoder
model = joblib.load("./models/random_forest_model.pkl")
encoder = joblib.load("./models/encoder.pkl")


def predict_match_outcome(home_team, away_team, home_score, away_score, neutral, tournament):
    # Create a DataFrame for the new match
    new_match = pd.DataFrame({
        'home_team': [home_team],
        'away_team': [away_team],
        'home_score': [home_score],
        'away_score': [away_score],
        'neutral': [neutral],
        'tournament': [tournament]
    })

    # Encode categorical features
    categorical_features = new_match[['home_team', 'away_team', 'tournament']]
    encoded_features = encoder.transform(categorical_features)

    # Combine encoded categorical features with numerical ones
    numeric_features = new_match[['home_score', 'away_score', 'neutral']].reset_index(drop=True)
    match_features = pd.concat([pd.DataFrame(encoded_features), numeric_features], axis=1)
    match_features.columns = match_features.columns.astype(str)  # Convert all column names to strings

    # Predict the outcome
    outcome = model.predict(match_features)[0]
    return outcome


# Example usage for predicting the outcome of a specific match
print(predict_match_outcome("Spain", "Wales", 1, 0, False, "Friendly"))
