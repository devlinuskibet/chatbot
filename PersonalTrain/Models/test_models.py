import joblib

# Function to merge similar symptoms (same as before)
def merge_similar_symptoms(symptoms_list):
    merged_symptoms = []
    for symptom in symptoms_list:
        if symptom in ['Headache', 'Migraine']:
            merged_symptoms.append('Headache/Migraine')
        elif symptom in ['Fever', 'Chills']:
            merged_symptoms.append('Fever/Chills')
        else:
            merged_symptoms.append(symptom)
    return merged_symptoms

# Load the KNN model
knn_model = joblib.load('C:\\Users\\linzs\\Desktop\\PersonalTrain\\Models\\knn_model.pkl')

# Load the Naive Bayes model
nb_model = joblib.load('C:\\Users\\linzs\\Desktop\\PersonalTrain\\Models\\nb_model.pkl')

# Load the MultiLabelBinarizer
mlb = joblib.load('C:\\Users\\linzs\\Desktop\\PersonalTrain\\Models\\mlb.pkl')

# Test with new symptoms
new_symptoms = ['Fever', 'Chills', 'Headache', 'Sweating']  # You can change this with any new symptoms
new_symptoms = merge_similar_symptoms(new_symptoms)  # Apply the same feature engineering
new_symptoms_transformed = mlb.transform([new_symptoms])

# Make predictions using the KNN model
knn_prediction = knn_model.predict(new_symptoms_transformed)
print(f"Predicted Disease (KNN): {knn_prediction[0]}")

# Make predictions using the Naive Bayes model
nb_prediction = nb_model.predict(new_symptoms_transformed)
print(f"Predicted Disease (Naive Bayes): {nb_prediction[0]}")
