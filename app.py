from flask import Flask,render_template,request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl",'rb'))
data = pd.read_csv("Cleaned_Dataset 1.csv")

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
    age = int(request.form.get('age'))
    gender = request.form.get('gender')
    stream = request.form.get('stream')
    internships = int(request.form.get('internships'))
    cgpa = float(request.form.get('cgpa'))
    hostel = request.form.get('hostel')
    backlogs = request.form.get('backlogs')

    prediction = model.predict(pd.DataFrame([[age,gender,stream,internships,cgpa,hostel,backlogs]], 
                                            columns=['Age','Gender','Stream','Internships','CGPA','Hostel','HistoryOfBacklogs']))
    prob = model.predict_proba(pd.DataFrame([[age,gender,stream,internships,cgpa,hostel,backlogs]], 
                                            columns=['Age','Gender','Stream','Internships','CGPA','Hostel','HistoryOfBacklogs']))
    return str(np.round(prob[0][1]*100,2))

if __name__ == "__main__":
    app.run(debug=True)