# Load the saved models and binarizer
import joblib

# Load the KNN model
knn_model = joblib.load('C:\\Users\\linzs\\Desktop\\PersonalTrain\\Models\\knn_model.pkl')

# Load the Naive Bayes model
nb_model = joblib.load('C:\\Users\\linzs\\Desktop\\PersonalTrain\\Models\\nb_model.pkl')

# Load the MultiLabelBinarizer
mlb = joblib.load('C:\\Users\\linzs\\Desktop\\PersonalTrain\\Models\\mlb.pkl')

# Define the merge_similar_symptoms function (used in the original model)
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

# Define some new test cases with different sets of symptoms
test_cases = [
    ['Fever', 'Chills', 'Headache', 'Sweating'],
    ['Cough', 'Chest pain', 'Shortness of breath'],
    ['Diarrhea', 'Abdominal pain', 'Nausea'],
    ['Rash', 'Itching', 'Swelling'],
    ['Fatigue', 'Joint pain', 'Muscle pain'],
    ['Dizziness', 'Blurred vision', 'Numbness']
]

# Loop through each test case
for i, symptoms in enumerate(test_cases):
    symptoms = merge_similar_symptoms(symptoms)  # Apply feature engineering
    symptoms_transformed = mlb.transform([symptoms])  # Transform using MultiLabelBinarizer

    # Make predictions using KNN and Naive Bayes
    knn_prediction = knn_model.predict(symptoms_transformed)
    nb_prediction = nb_model.predict(symptoms_transformed)

    # Display results for each test case
    print(f"\nTest Case {i+1}: Symptoms = {symptoms}")
    print(f"Predicted Disease (KNN): {knn_prediction[0]}")
    print(f"Predicted Disease (Naive Bayes): {nb_prediction[0]}")
