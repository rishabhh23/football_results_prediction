import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
from src.data_preprocessing import calculate_team_stats

def prepare_training_data(df, team_stats):
    data = []
    for _, row in df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        home_stats = team_stats[home_team]
        away_stats = team_stats[away_team]

        outcome = 1 if row['home_score'] > row['away_score'] else (0 if row['home_score'] == row['away_score'] else -1)

        data.append([
            home_stats['win_rate'], away_stats['win_rate'],
            home_stats['avg_score'], away_stats['avg_score'],
            outcome
        ])

    model_df = pd.DataFrame(data,
                            columns=['home_win_rate', 'away_win_rate', 'home_avg_score', 'away_avg_score', 'outcome'])
    model_df = model_df[model_df['outcome'] != 0]
    return model_df


def train_model(training_data_path):
    # Prepare data
    team_stats = calculate_team_stats(pd.read_csv(training_data_path))
    model_df = prepare_training_data(pd.read_csv(training_data_path), team_stats)

    X = model_df[['home_win_rate', 'away_win_rate', 'home_avg_score', 'away_avg_score']]
    y = model_df['outcome'].apply(lambda x: 1 if x == 1 else 0)

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Save model and stats
    joblib.dump(model, "./models/win_predictor_model.pkl")
    joblib.dump(team_stats, "./models/team_stats.pkl")
    print("Model and team stats saved.")

    return model, team_stats
