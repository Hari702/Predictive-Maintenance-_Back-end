from flask import Flask,request,jsonify, session,make_response,Response,json
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import time
from datetime import datetime, timedelta
from flask_session import Session
from sklearn.preprocessing import OrdinalEncoder
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold,cross_val_score,cross_val_predict,StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,ConfusionMatrixDisplay,roc_curve,roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler,MinMaxScaler
import secrets


app = Flask(__name__)
CORS(app)

# Load dataset and model
df = pd.read_csv("D:/Digiverz_Data_Engineering_and_science/Accelerator-Development/Notebook/dataset/Final_dataset.csv")

df_model = df.copy()
columns_failure_type = df_model['Failure_type'].unique().tolist()
ordinal_en = OrdinalEncoder(categories=[["L", "M", "H"], columns_failure_type])

transformed_data = ordinal_en.fit_transform(df_model[['Type', 'Failure_type']])
df_model.drop(['Type', 'Failure_type'], axis=1, inplace=True)
transformed_df = pd.DataFrame(transformed_data, index=df_model.index, columns=['Type', 'Failure_type'])
df_model = pd.concat([df_model, transformed_df], axis=1)

x = df_model.drop(columns=["UDI", "Product_Id", "Target", "Failure_type", "Type"], axis=1)
y = df_model["Failure_type"]

ssKfold = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
for train_index, test_index in ssKfold.split(x, y):
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

with open("D:/Digiverz_Data_Engineering_and_science/Accelerator-Development/Notebook/Model/failure_prediction_model.pkl", 'rb') as file:
    random_forest = pickle.load(file)

priority_mapping = {
    "Power Failure": "High",
    "Tool Wear Failure": "Medium",
    "Overstrain Failure": "High",
    "Heat Dissipation Failure": "Medium"
}

def get_timestamp():
    return pd.to_datetime(datetime.now())

def predict_maintenance(new_data):
    predicted_probabilities = random_forest.predict_proba([new_data])[0]
    max_probability = max(predicted_probabilities)
    failure_type_index = np.argmax(predicted_probabilities)
    failure_type = ordinal_en.categories_[1][failure_type_index]
    priority = priority_mapping.get(failure_type, "Low")
    schedule_days = {
        "High": 2,
        "Medium": 4,
        "Low": 90
    }
    days_for_maintenance = schedule_days.get(priority, 14)
    return priority, failure_type, days_for_maintenance

def process_data(new_data):
    timestamp = get_timestamp()
    priority, failure_type, days_for_maintenance = predict_maintenance(new_data)
    maintenance_time = timestamp + timedelta(days=days_for_maintenance)
    # if(priority=="Low" & failure_type == "No Failure"){
    #     maintenance_time=timestamp+timedelta(dayys)
    # }
    return {
        "Air temperature":new_data[0],
        "Process temperature":new_data[1],
        "Rotational speed":new_data[2],
        "Torque":new_data[3],
        "Tool wear":new_data[4],
        "DateTime":timestamp.strftime('%d-%m-%Y'),
        "Reading time":timestamp.strftime('%H:%M:%S'),
        "priority": priority,
        "failure_type": failure_type,
        "maintenance_time": maintenance_time.strftime('%d-%m-%Y')
    }
    
# parameter_limits = {
#     "Air temperature": {"min": 10, "max": 40},
#     "Process temperature": {"min": 50, "max": 100},
#     "Rotational speed": {"min": 500, "max": 1500},
#     "Torque": {"min": 20, "max": 100},
#     "Tool wear": {"min": 0, "max": 500},
# }

# airTempMin=200
# airTempMax=300
# processTempMin=200
# processTempMax=300
# rotationalSpeedMin=1000
# roationalSpeedMax=2000
# torqueMin=30
# torqueMax=50
# toolWearMin=15
# toolWearMax=50
    
# def rule_based_maintenance(new_data):
    
    
    

# Load test data
test_data = pd.read_csv('D:/Digiverz_Data_Engineering_and_science/Accelerator-Development/Notebook/dataset/realtime_data.csv')

@app.route('/stream')
def stream():
    def event_stream():
        index = 0
        while index < len(test_data):
            row = test_data.iloc[index]
            new_data = row[['Air_temperature', 'Process_temperature', 'Rotational_speed', 'Torque', 'Tool_wear']].tolist()
            response = process_data(new_data)
            yield f"data: {json.dumps(response)}\n\n"
            index += 1
            time.sleep(10)  # Wait 30 seconds before sending the next row

    return Response(event_stream(), mimetype="text/event-stream")




@app.route("/config",methods=['POST'])
def get_config_data():
    global config_data
    config_data=request.json
    print(config_data)
    return jsonify(config_data)
    

def check_parameters(data):
    alerts = []

    airTempMin = config_data['air-temperature-min']
    airTempMax=config_data['air-temperature-max']
    processTempMin = config_data['process-temperature-min']
    processTempMax = config_data['process-temperature-max']
    rotationalSpeedMin=config_data['rotational-speed-min']
    rotationalSpeedMax=config_data['rotational-speed-max']
    torqueMin=config_data['torque-min']
    torqueMax=config_data['torque-max']
    toolWearMin=config_data['toolwear-min']
    toolWearMax=config_data['toolwear-max']


 
    if data[0] < airTempMin or data[0] > airTempMax:
        alerts.append('Critical: Air temperature out of bound')
    if data[1] < processTempMin or data[1] > processTempMax:
        alerts.append('Critical: Process temperature out of bound')
    if data[2] < rotationalSpeedMin or data[2] > rotationalSpeedMax:
        alerts.append('Critical:Rotational speed out of bound')
    if data[3] < torqueMin or data[3] > torqueMax:
        alerts.append('Critical: Torque out of bound')
    if data[4]< toolWearMin or data[4]>toolWearMax:
        alerts.append('Critical: Tool wear out of bound')

    # Process Temperature Check
    # if data[1] < process_tempmin or data[1] > process_tempmax:
    #     alerts.append('Warning: Process temperature is approaching critical limits')
    # if process_temp < 200 or process_temp > 300:
    #     alerts.append('Critical: Process temperature out of bounds!')

    # # Rotational Speed Check
    # if rotational_speed < 1100 or rotational_speed > 1900:
    #     alerts.append('Warning: Rotational speed is approaching critical limits')
    # if rotational_speed < 1000 or rotational_speed > 2000:
    #     alerts.append('Critical: Rotational speed out of bounds!')

    # # Torque Check
    # if torque < 35 or torque > 45:
    #     alerts.append('Warning: Torque is approaching critical limits')
    # if torque < 30 or torque > 50:
    #     alerts.append('Critical: Torque out of bounds!')

    # # Tool Wear Check
    # if tool_wear < 20 or tool_wear > 45:
    #     alerts.append('Warning: Tool wear is approaching critical limits')
    # if tool_wear < 15 or tool_wear > 50:
    #     alerts.append('Critical: Tool wear out of bounds!')

    return alerts

def process_data_config(new_data):
    timestamp = get_timestamp()
    alerts= check_parameters(new_data)
    # maintenance_time = timestamp + timedelta(days=days_for_maintenance)
    # if(priority=="Low" & failure_type == "No Failure"){
    #     maintenance_time=timestamp+timedelta(dayys)
    # }
    return {
        "Air temperature":new_data[0],
        "Process temperature":new_data[1],
        "Rotational speed":new_data[2],
        "Torque":new_data[3],
        "Tool wear":new_data[4],
        "DateTime":timestamp.strftime('%d-%m-%Y'),
        "Reading time":timestamp.strftime('%H:%M:%S'),
        "alerts":alerts
    }

@app.route('/streamConfig')
def streamConfig():
    def event_stream_another():
        index = 0
        while index < len(test_data):
            row = test_data.iloc[index]
            new_data = row[['Air_temperature', 'Process_temperature', 'Rotational_speed', 'Torque', 'Tool_wear']].tolist()
            response = process_data_config(new_data)
            yield f"data: {json.dumps(response)}\n\n"
            index += 1
            time.sleep(15)  # Wait 30 seconds before sending the next row

    return Response(event_stream_another(), mimetype="text/event-stream")



    









if __name__ == '__main__':
    app.run(debug=True)








# if __name__ == "__main__":
#     app.run()

# if __name__ == "__main__":
#     socketio.run(app, debug=True)

