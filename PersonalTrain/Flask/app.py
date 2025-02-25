from flask import Flask, render_template, request
import joblib

# Load the KNN model
knn_model = joblib.load('C:\\Users\\linzs\\Desktop\\PersonalTrain\\Models\\knn_model.pkl')

# Load the Naive Bayes model
nb_model = joblib.load('C:\\Users\\linzs\\Desktop\\PersonalTrain\\Models\\nb_model.pkl')

# Load the MultiLabelBinarizer
mlb = joblib.load('C:\\Users\\linzs\\Desktop\\PersonalTrain\\Models\\mlb.pkl')

# Initialize Flask app
app = Flask(__name__)

def process_symptoms(input_symptoms):
    """Process input symptoms and return predictions from both models."""
    # Assuming symptoms are comma-separated in the form field
    symptoms = [symptom.strip() for symptom in input_symptoms.split(',')]
    
    # Transform symptoms using the same method as during training
    transformed_symptoms = mlb.transform([symptoms])
    
    # Make predictions
    knn_prediction = knn_model.predict(transformed_symptoms)[0]
    nb_prediction = nb_model.predict(transformed_symptoms)[0]

    return knn_prediction, nb_prediction

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get symptoms input from user
        user_input = request.form['symptoms']
        
        # Process symptoms and get predictions
        knn_pred, nb_pred = process_symptoms(user_input)
        
        # Render predictions in the HTML
        return render_template('index.html', knn_prediction=knn_pred, nb_prediction=nb_pred, user_symptoms=user_input)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
