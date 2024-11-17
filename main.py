import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.data_preprocessing import load_data, calculate_team_stats, prepare_training_data
from src.visualization import plot_class_distribution, plot_cost_distribution

# Load data and calculate team stats
data_path = "C:/Users/rish2/OneDrive/Desktop/project1/data/raw/results_copy.csv"
df = load_data(data_path)
team_stats = calculate_team_stats(df)

def load_model_and_stats():
    model = joblib.load("./models/win_predictor_model.pkl")
    scaler = joblib.load("./models/scaler.pkl")
    team_stats = joblib.load("./models/team_stats.pkl")
    return model, scaler, team_stats

def predict_match_winner(team1, team2):
    model, scaler, team_stats = load_model_and_stats()

    if team1 not in team_stats or team2 not in team_stats:
        return "One or both team names not found in the data."

    t1_stats = team_stats[team1]
    t2_stats = team_stats[team2]

    match_data = pd.DataFrame(
        [[t1_stats['win_rate'], t2_stats['win_rate'], t1_stats['avg_score'], t2_stats['avg_score']]],
        columns=['home_win_rate', 'away_win_rate', 'home_avg_score', 'away_avg_score']
    )

    # Scale the data using the saved scaler
    match_data_scaled = scaler.transform(match_data)

    prediction = model.predict(match_data_scaled)[0]
    return team1 if prediction == 1 else team2

def train_model(training_data_path):
    team_stats = calculate_team_stats(pd.read_csv(training_data_path))
    model_df = prepare_training_data(pd.read_csv(training_data_path), team_stats)

    X = model_df[['home_win_rate', 'away_win_rate', 'home_avg_score', 'away_avg_score']]
    y = model_df['outcome'].apply(lambda x: 1 if x == 1 else 0)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model iteratively to simulate loss
    model = LogisticRegression(max_iter=100, warm_start=True)
    epochs = 50
    accuracies, losses = [], []

    for epoch in range(epochs):
        model.fit(X_train_scaled, y_train)
        y_pred_train = model.predict(X_train_scaled)

        # Calculate training accuracy and loss for each epoch
        accuracy = accuracy_score(y_train, y_pred_train)
        accuracies.append(accuracy)
        loss = 1 - accuracy
        losses.append(loss)

    # Save model, scaler, and team stats
    joblib.dump(model, "./models/win_predictor_model.pkl")
    joblib.dump(scaler, "./models/scaler.pkl")
    joblib.dump(team_stats, "./models/team_stats.pkl")
    print("Model, scaler, and team stats saved.")

    # Evaluation on test set
    y_pred = model.predict(X_test_scaled)
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

    # Plotting the loss and accuracy graphs
    plt.figure(figsize=(12, 5))
    plt.plot(range(1, epochs + 1), losses, label='Loss', color='red')
    plt.plot(range(1, epochs + 1), accuracies, label='Accuracy', color='blue')
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.title("Loss and Accuracy over Epochs")
    plt.legend()
    plt.show()

    # Plot individual metric graphs
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [accuracy, precision, recall, f1]

    plt.figure(figsize=(8, 6))
    plt.bar(metrics, values, color=['skyblue', 'orange', 'green', 'purple'])
    plt.ylim(0, 1)
    plt.title("Model Evaluation Metrics")
    plt.ylabel("Score")
    plt.show()

    # Confusion matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu", cbar=False,
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.show()

    # Return the training labels for visualization
    return model, team_stats, scaler, y_train

if __name__ == "__main__":
    # Train the model and retrieve y_train
    model, team_stats, scaler, y_train = train_model(data_path)

    # Plot class distribution in y_train
    plot_class_distribution(y_train)

    # Define cost data and plot cost distribution
    # cost_data = {
    #     'Production': 300,
    #     'Shipping': 150,
    #     'Marketing': 100,
    #     'Miscellaneous': 50
    # }
    # plot_cost_distribution(cost_data)  # Visualize cost distribution

    # Example prediction usage
    print("Predicted Match Winner is:", predict_match_winner("Kazakhstan", "Austria"))
