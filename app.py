
from flask import Flask,render_template,request
import pickle
import json
import numpy as np

model = pickle.load(open("artifacts/model.pkl","rb"))

with open("artifacts/columns_name.json","r") as json_file:
    col_name = json.load(json_file)
col_name_list =col_name['col_name']

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods= ["GET","POST"])
def predict():
    data = request.form
    input_data = np.zeros(len(col_name_list))

    input_data[0] = data['gender']
    input_data[1] = data['married']
    input_data[2] = data['dependents']
    input_data[3] = data['education']
    input_data[4] = data['self_employed']
    input_data[5] = data['loan_term']
    input_data[6] = data['credit_history']
    input_data[7] = data['property_area']
    ######### Log_Transformation for loan amount 
    loan_amount = int(data['loan_amt'])
    loan_amount_log = np.log(loan_amount)
    input_data[8] = loan_amount_log
    ######### Log_transformation on Total income 
    total_income = int(data['applicant_income']) + int(data['co_applicant_income'])
    total_income_log = np.log(total_income)
    input_data[9] = total_income_log

    
    
    
    print(input_data)
    result = model.predict([input_data])

    if result[0] == 0:
        loan_result = "Loan Rejected"
    else: 
        loan_result = "Loan Accepted"
    
    print(loan_result)
    
    return render_template("index.html",prediction =loan_result)

if __name__== "__main__":
    app.run(host= "0.0.0.0",port=8080, debug=True) ##### AWS Deployment host = 0.0.0.0 port= 8080 debug= False

