import joblib
import pandas as pd


def load_model_and_stats():
    model = joblib.load("./models/win_predictor_model.pkl")
    team_stats = joblib.load("./models/team_stats.pkl")
    return model, team_stats


def predict_match_winner(team1, team2):
    model, team_stats = load_model_and_stats()

    if team1 not in team_stats or team2 not in team_stats:
        return "One or both team names not found in the data."

    t1_stats = team_stats[team1]
    t2_stats = team_stats[team2]

    # Ensure valid feature names
    match_data = pd.DataFrame([[
        t1_stats['win_rate'],
        t2_stats['win_rate'],
        t1_stats['avg_score'],
        t2_stats['avg_score']
    ]], columns=['home_win_rate', 'away_win_rate', 'home_avg_score', 'away_avg_score'])

    prediction = model.predict(match_data)[0]
    return team1 if prediction == 1 else team2


# Example usage
print(predict_match_winner("Scotland", "Argentina"))
