from flask import Flask,render_template,request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl",'rb'))
data = pd.read_csv("Cleaned_Dataset 1.csv")

def suggest_adaptive_improvements(new_data, model):
    def get_prob(data):
        return model.predict_proba(data)[0][1]

    current_prob = round(get_prob(new_data), 2)
    current_cgpa = new_data['CGPA'].values[0]
    current_intern = new_data['Internships'].values[0]

    suggestions = []
    if current_prob < 0.5:
        target_prob = 0.5
        stage = "to get a fair chance"
    elif current_prob < 0.8:
        target_prob = 0.8
        stage = "to be surely placed"
    else:
        suggestions.append("You have a high chance of getting placed. Keep it up!")
        target_prob = min(current_prob + 0.1, 1.0)
        stage = "for further improvement"

    improved = False

    data_both = new_data.copy()
    data_both['Internships'] = 1
    cgpa_val = current_cgpa
    while cgpa_val <= 10.0:
        data_both['CGPA'] = cgpa_val
        prob = round(get_prob(data_both), 2)
        if prob >= target_prob:
            suggestions.append(f"Do an internship and raise CGPA to {cgpa_val:.2f} {stage} [Prob: {prob:.2f}]")
            improved = True
            break
        cgpa_val = round(cgpa_val + 0.1, 2)

    return current_prob, suggestions


@app.route("/")
def index():
    age = sorted(data['Age'].unique())
    gender = sorted(data['Gender'].unique())
    stream = sorted(data['Stream'].unique())
    internships = sorted(data['Internships'].unique())
    cgpa = sorted(data['CGPA'].unique())
    hostel = sorted(data['Hostel'].unique())
    backlogs = sorted(data['HistoryOfBacklogs'].unique())
    return render_template('index.html',age=age,gender=gender,stream=stream,internships=internships,
                           cgpa=cgpa,hostel=hostel,backlogs=backlogs)


@app.route("/predict", methods=['POST'])
def predict():
    features = {
        'Age': int(request.form.get('age')),
        'Gender': request.form.get('gender'),
        'Stream': request.form.get('stream'),
        'Internships': int(request.form.get('internships')),
        'CGPA': float(request.form.get('cgpa')),
        'Hostel': request.form.get('hostel'),
        'HistoryOfBacklogs': request.form.get('backlogs')
    }

    input_df = pd.DataFrame([features])

    prob, suggestions = suggest_adaptive_improvements(input_df, model)

    return {
        "probability": round(prob * 100, 2),
        "suggestions": suggestions
    }

if __name__ == "__main__":
    app.run(debug=True)