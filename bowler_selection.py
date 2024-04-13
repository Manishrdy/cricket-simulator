import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

df = pd.read_csv("modified_dataset.csv")
df.head()

# Preprocessing
df['isPlayerOut'] = df['isPlayerOut'].astype(int)

# Calculate the bowler's performance against each batsman
performance_metrics = ['runs_off_bat', 'extras', 'isPlayerOut', 'striker_total_runs', 'non_striker_total_runs', 'bowler_wickets', 'bowler_runs']
bowler_performance = df.groupby(['bowler', 'striker', 'non_striker']).agg({
    'runs_off_bat': 'mean',
    'extras': 'mean',
    'isPlayerOut': 'sum',  # Assuming 'sum' will give the number of times the bowler got the batsman out
    'striker_total_runs': 'mean',
    'non_striker_total_runs': 'mean',
    'bowler_wickets': 'mean',
    'bowler_runs': 'mean'
}).reset_index()

# Encode categorical features
label_encoders = {}
for column in ['bowler', 'striker', 'non_striker']:
    le = LabelEncoder()
    bowler_performance[column] = le.fit_transform(bowler_performance[column])
    label_encoders[column] = le

# Split the dataset
X = bowler_performance.drop('isPlayerOut', axis=1)
y = bowler_performance['isPlayerOut']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = classifier.predict(X_test_scaled)

# Performance metrics - for multiclass classification
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')  # Change average to 'macro', 'micro', or 'weighted' based on your requirement
recall = recall_score(y_test, y_pred, average='macro')  # Same here
f1 = f1_score(y_test, y_pred, average='macro')  # And here
conf_matrix = confusion_matrix(y_test, y_pred)

# Print the performance metrics
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print(f'Confusion Matrix:\n{conf_matrix}')

# Function to predict the probability for each bowler
def predict_bowler_probabilities(striker, non_striker, bowlers_list, classifier, label_encoders, scaler):
    probabilities = {}
    striker_encoded = label_encoders['striker'].transform([striker])[0]
    non_striker_encoded = label_encoders['non_striker'].transform([non_striker])[0]

    for bowler in bowlers_list:
        bowler_encoded = label_encoders['bowler'].transform([bowler])[0]
        # Create a DataFrame with the same structure as the model's features
        # Ensure all features are present and in the correct order
        features = pd.DataFrame([[bowler_encoded, striker_encoded, non_striker_encoded, 0, 0, 0, 0, 0, 0]],
                                columns=['bowler', 'striker', 'non_striker', 'runs_off_bat', 'extras', 'striker_total_runs', 'non_striker_total_runs', 'bowler_wickets', 'bowler_runs'])
        features_scaled = scaler.transform(features)
        prob = classifier.predict_proba(features_scaled)[0][1]
        probabilities[bowler] = prob

    return probabilities

# Example usage
striker = 'AJ Finch'  # Replace with actual striker name
non_striker = 'M Klinger'  # Replace with actual non-striker name
bowlers_list = ['SL Malinga', 'KMDN Kulasekara', 'JRMVB Sanjaya']  # Replace with actual bowler names

probabilities = predict_bowler_probabilities(striker, non_striker, bowlers_list, classifier, label_encoders, scaler)
print(probabilities)