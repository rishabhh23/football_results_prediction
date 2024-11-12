import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
from src.data_preprocessing import calculate_team_stats, scale_features

def prepare_training_data(df, team_stats):
    data = []
    for _, row in df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        home_stats = team_stats[home_team]
        away_stats = team_stats[away_team]

        # 1 for home win, 0 for draw, -1 for away win
        outcome = 1 if row['home_score'] > row['away_score'] else (0 if row['home_score'] == row['away_score'] else -1)

        data.append([
            home_stats['win_rate'], away_stats['win_rate'],
            home_stats['avg_score'], away_stats['avg_score'],
            outcome
        ])

    model_df = pd.DataFrame(data, columns=['home_win_rate', 'away_win_rate', 'home_avg_score', 'away_avg_score', 'outcome'])
    model_df = model_df[model_df['outcome'] != 0]
    return model_df

def train_model(training_data_path):
    team_stats = calculate_team_stats(pd.read_csv(training_data_path))
    model_df = prepare_training_data(pd.read_csv(training_data_path), team_stats)

    X = model_df[['home_win_rate', 'away_win_rate', 'home_avg_score', 'away_avg_score']]
    y = model_df['outcome'].apply(lambda x: 1 if x == 1 else 0)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Train the model
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    # Save the model, scaler, and team stats
    joblib.dump(model, "./models/win_predictor_model.pkl")
    joblib.dump(scaler, "./models/scaler.pkl")
    joblib.dump(team_stats, "./models/team_stats.pkl")
    print("Model, scaler, and team stats saved.")

    # Make predictions on the scaled test set
    y_pred = model.predict(X_test_scaled)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Print the results
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    return model, team_stats, scaler

# Example usage
# train_model("path/to/your/data.csv")
