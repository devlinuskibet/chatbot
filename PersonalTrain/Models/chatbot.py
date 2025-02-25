import joblib
from sklearn.preprocessing import MultiLabelBinarizer

# Load the models
# Load the KNN model
knn_model = joblib.load('C:\\Users\\linzs\\Desktop\\PersonalTrain\\Models\\knn_model.pkl')

# Load the Naive Bayes model
nb_model = joblib.load('C:\\Users\\linzs\\Desktop\\PersonalTrain\\Models\\nb_model.pkl')

# Load the MultiLabelBinarizer
mlb = joblib.load('C:\\Users\\linzs\\Desktop\\PersonalTrain\\Models\\mlb.pkl')
# Known symptoms (this list can be expanded to include all symptoms the model was trained on)
known_symptoms = [
    'Fever/Chills', 'Headache/Migraine', 'Sweating', 'Cough', 'Chest pain',
    'Shortness of breath', 'Diarrhea', 'Abdominal pain', 'Nausea', 'Rash',
    'Itching', 'Swelling', 'Fatigue', 'Joint pain', 'Muscle pain',
    'Dizziness', 'Blurred vision', 'Numbness'
]

# Feature Engineering: Merge similar symptoms and normalize input
def merge_similar_symptoms(symptoms_list):
    merged_symptoms = []
    for symptom in symptoms_list:
        symptom = symptom.capitalize()  # Normalize the symptom to capitalize the first letter
        if symptom in ['Headache', 'Migraine']:
            merged_symptoms.append('Headache/Migraine')
        elif symptom in ['Fever', 'Chills']:
            merged_symptoms.append('Fever/Chills')
        elif symptom not in known_symptoms:
            print(f"Warning: '{symptom}' is not recognized and will be ignored.")
        else:
            merged_symptoms.append(symptom)
    return merged_symptoms

# Chatbot logic to predict the disease based on symptoms
def predict_disease(symptoms):
    # Feature Engineering
    symptoms = merge_similar_symptoms(symptoms)
    
    if not symptoms:
        return None, None

    # Transform symptoms using the MultiLabelBinarizer
    symptoms_transformed = mlb.transform([symptoms])
    
    # Predict with KNN model
    knn_prediction = knn_model.predict(symptoms_transformed)[0]
    
    # Predict with Naive Bayes model
    nb_prediction = nb_model.predict(symptoms_transformed)[0]
    
    return knn_prediction, nb_prediction

# Chatbot interaction
def chatbot():
    print("Welcome to the Medical Chatbot!")
    print("Please enter your symptoms separated by commas (e.g., 'Fever, Headache, Sweating').")
    print(f"Known Symptoms: {', '.join(known_symptoms)}")
    
    while True:
        user_input = input("Enter symptoms (or type 'exit' to quit): ").strip().lower()
        if user_input == 'exit':
            print("Goodbye!")
            break
        
        symptoms = [symptom.strip() for symptom in user_input.split(',')]
        knn_prediction, nb_prediction = predict_disease(symptoms)
        
        if knn_prediction and nb_prediction:
            print(f"KNN Prediction: {knn_prediction}")
            print(f"Naive Bayes Prediction: {nb_prediction}")
        else:
            print("No valid symptoms were provided. Please try again.")

if __name__ == "__main__":
    chatbot()