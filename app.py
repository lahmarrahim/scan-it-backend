from datetime import datetime
import json
import os
from flask import Flask, flash, request, redirect, jsonify,json
from werkzeug.utils import secure_filename
import pickle 
import pandas as pd
from preproc import encodage
from preproc_UNSW import apply_log_transformation
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import sqlalchemy
from sqlalchemy.orm import sessionmaker
from flask_migrate import Migrate
import sqlite3
import numpy as np
app=Flask(__name__)
app.secret_key = "YOUR_SECRET_KEY"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

path = os.getcwd()
# file Upload
UPLOAD_FOLDER = os.path.join(path, 'uploads')

if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

CORS(app)

ALLOWED_EXTENSIONS = set(['csv'])
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(path, 'results.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
migrate = Migrate(app, db)

class Result(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255))
    result = db.Column(db.String(50))
    dataset =db.Column(db.String(50))
    classifier = db.Column(db.String(50))

    date = db.Column(db.DateTime, default=datetime.utcnow)
    archive = db.Column(db.Boolean, nullable=False, default=False)

    def __init__(self, filename, result,dataset,classifier):
        self.filename = filename
        self.result = result
        self.dataset = dataset
        self.classifier=classifier
   
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#preprocessing
with open('preprocessing_transformer.pkl', 'rb') as file:
    preprocessing_transformer = pickle.load(file)

#Models
with open('DT_estimator.pkl', 'rb') as file:
    estimator_model_dt = pickle.load(file)

with open('RF_estimator.pkl', 'rb') as file:
    estimator_model_rf = pickle.load(file)

with open('KNN_estimator.pkl', 'rb') as file:
    estimator_model_knn = pickle.load(file)

with open('SVM_estimator.pkl', 'rb') as file:
    estimator_model_svm = pickle.load(file)

with open('LR_estimator.pkl', 'rb') as file:
    estimator_model_lr = pickle.load(file)

with open('xgb_estimator.pkl', 'rb') as file:
    estimator_model_xgb = pickle.load(file)


with open('preprocessing_transformer_UNSW.pkl', 'rb') as file:
    preprocessing_transformer_UNSW = pickle.load(file)

#Models
with open('DT_estimator_UNSW.pkl', 'rb') as file:
    estimator_model_dt_UNSW = pickle.load(file)

with open('RF_estimator_UNSW.pkl', 'rb') as file:
    estimator_model_rf_UNSW = pickle.load(file)

with open('KNN_estimator_UNSW.pkl', 'rb') as file:
    estimator_model_knn_UNSW = pickle.load(file)

with open('SVM_estimator_UNSW.pkl', 'rb') as file:
    estimator_model_svm_UNSW= pickle.load(file)

with open('LR_estimator_UNSW.pkl', 'rb') as file:
    estimator_model_lr_UNSW = pickle.load(file)

with open('xgb_estimator_UNSW.pkl', 'rb') as file:
    estimator_model_xgb_UNSW = pickle.load(file)



@app.route('/ScanWithDT', methods=['POST','GET'])
def dtt():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Get the directory path of the uploaded file
            #file_directory = os.path.dirname(file_path)

            # Read the .txt file into a DataFrame
            df = pd.read_csv(file_path)
            
            if 'duration' in df.columns:
                df.loc[df['class'] != 'Normal', 'class'] = 1
                df.loc[df['class'] == 'Normal', 'class'] = 0 
                df['class'] = df['class'].astype(int)
                encoded_instance = preprocessing_transformer['encoder'](df)
                selected_features =  preprocessing_transformer['selected_features']
                scaler = preprocessing_transformer['scaler']
                normalized_instance = pd.DataFrame(scaler.transform(encoded_instance), columns=encoded_instance.columns)
                normalized_instance =normalized_instance[selected_features] 
                result = estimator_model_dt.predict(normalized_instance)
                prediction=estimator_model_dt.predict_proba(normalized_instance)
                prob1=prediction[0][0]
                prob2=prediction[0][1]
                if prob1 < prob2 :
                    prob = prob2 
                else :
                    prob = prob1
                if result == [0]:
                    result= 'Normal'
                else:
                    result = 'Attack'
                dataset='NSL-KDD'
                
            else :
                if 'dur' in df.columns : 
                    df.loc[df['attack_cat'] != 'Normal', 'attack_cat'] = 1
                    df.loc[df['attack_cat'] == 'Normal', 'attack_cat'] = 0  
                    df['attack_cat'] = df['attack_cat'].astype(int)
                    log_instance = preprocessing_transformer_UNSW['log_transformation'](df)
                    encoded_instance=log_instance
                    encoder = preprocessing_transformer_UNSW['label_encoder']
                    for feature, le in encoder.items():
                        encoded_instance[feature] = le.transform(encoded_instance[feature])

                    selected_features =  preprocessing_transformer_UNSW['selected_features']

                    scaler = preprocessing_transformer_UNSW['scaler']
                    normalized_instance = pd.DataFrame(scaler.transform(encoded_instance), columns=encoded_instance.columns)
                    normalized_instance =normalized_instance[selected_features] 
                    result = estimator_model_dt_UNSW.predict(normalized_instance)
                    prediction=estimator_model_dt_UNSW.predict_proba(normalized_instance)
                    prob1=prediction[0][0]
                    prob2=prediction[0][1]
                    if prob1 <prob2 :
                        prob = prob2 
                    else :
                        prob = prob1
                    if result == [0]:
                        result= 'Normal'
                    else:
                        result = 'Attack'
                    dataset='UNSW-NB15'
                else:
                    return jsonify({'error': 'Verify your CSV file'})

            last_feature = df.iloc[:, -1]  # Get the last column

            if last_feature.iloc[-1]==0 and result=='Normal' :
                pred= 'True  Normal'
            if last_feature.iloc[-1] ==0 and result=='Attack' :
                pred= 'False Attack'
            if last_feature.iloc[-1]==1 and result=='Normal' :
                pred= 'False  Normal'
            if last_feature.iloc[-1]==1 and result=='Attack' :
                 pred='True Attack'

            classifier='Decision Tree'
            new_result = Result(filename, result, dataset, classifier)
            db.session.add(new_result)
            db.session.commit()
            prob = prob*100
        response = {
            'result': result,
            'filename': filename,
            'pred':pred,
            'prob':round(prob, 2)
        }
        return json.dumps(response)


@app.route('/ScanWithRF', methods=['GET', 'POST'])
def rff():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Get the directory path of the uploaded file
            #file_directory = os.path.dirname(file_path)

            # Read the .txt file into a DataFrame
            df = pd.read_csv(file_path)
            
            if 'duration' in df.columns:
                df.loc[df['class'] != 'Normal', 'class'] = 1
                df.loc[df['class'] == 'Normal', 'class'] = 0 
                df['class'] = df['class'].astype(int)
                encoded_instance = preprocessing_transformer['encoder'](df)
                selected_features =  preprocessing_transformer['selected_features']
                scaler = preprocessing_transformer['scaler']
                normalized_instance = pd.DataFrame(scaler.transform(encoded_instance), columns=encoded_instance.columns)
                normalized_instance =normalized_instance[selected_features] 
                result = estimator_model_rf.predict(normalized_instance)
                prediction=estimator_model_rf.predict_proba(normalized_instance)
                prob1=prediction[0][0]
                prob2=prediction[0][1]
                if prob1 < prob2 :
                    prob = prob2 
                else :
                    prob = prob1
                if result == [0]:
                    result= 'Normal'
                else:
                    result = 'Attack'
                dataset='NSL-KDD'
                
            else :
                if 'dur' in df.columns: 
                    df.loc[df['attack_cat'] != 'Normal', 'attack_cat'] = 1
                    df.loc[df['attack_cat'] == 'Normal', 'attack_cat'] = 0  
                    df['attack_cat'] = df['attack_cat'].astype(int)
                    log_instance = preprocessing_transformer_UNSW['log_transformation'](df)
                    encoded_instance=log_instance
                    encoder = preprocessing_transformer_UNSW['label_encoder']
                    for feature, le in encoder.items():
                        encoded_instance[feature] = le.transform(encoded_instance[feature])

                    selected_features =  preprocessing_transformer_UNSW['selected_features']

                    scaler = preprocessing_transformer_UNSW['scaler']
                    normalized_instance = pd.DataFrame(scaler.transform(encoded_instance), columns=encoded_instance.columns)
                    normalized_instance =normalized_instance[selected_features] 
                    result = estimator_model_rf_UNSW.predict(normalized_instance)
                    prediction=estimator_model_rf_UNSW.predict_proba(normalized_instance)
                    prob1=prediction[0][0]
                    prob2=prediction[0][1]
                    if prob1 < prob2 :
                        prob = prob2 
                    else :
                        prob = prob1
                    if result == [0]:
                        result= 'Normal'
                    else:
                        result = 'Attack'
                    dataset='UNSW-NB15'
                else:
                    return jsonify({'error': 'Verify your CSV file'}) 

            last_feature = df.iloc[:, -1]  # Get the last column

            if last_feature.iloc[-1]==0 and result=='Normal' :
                pred= 'True Normal'
            if last_feature.iloc[-1] ==0 and result=='Attack' :
                pred= 'False Attack'
            if last_feature.iloc[-1]==1 and result=='Normal' :
                pred= 'False Normal'
            if last_feature.iloc[-1]==1 and result=='Attack' :
                 pred='True Attack'

            classifier='Random Forest'
            new_result = Result(filename, result, dataset, classifier)
            db.session.add(new_result)
            db.session.commit()
            prob=prob*100
        response = {
            'result': result,
            'filename': filename,
            'pred':pred,
            'prob':round(prob, 2)
        }
        return json.dumps(response)


@app.route('/ScanWithKNN', methods=['GET', 'POST'])
def knnn():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Get the directory path of the uploaded file
            #file_directory = os.path.dirname(file_path)

            # Read the .txt file into a DataFrame
            df = pd.read_csv(file_path)
            
            if 'duration' in df.columns:
                df.loc[df['class'] != 'Normal', 'class'] = 1
                df.loc[df['class'] == 'Normal', 'class'] = 0 
                df['class'] = df['class'].astype(int)
                encoded_instance = preprocessing_transformer['encoder'](df)
                selected_features =  preprocessing_transformer['selected_features']
                scaler = preprocessing_transformer['scaler']
                normalized_instance = pd.DataFrame(scaler.transform(encoded_instance), columns=encoded_instance.columns)
                normalized_instance =normalized_instance[selected_features] 
                result = estimator_model_knn.predict(normalized_instance)
                prediction=estimator_model_knn.predict_proba(normalized_instance)
                prob1=prediction[0][0]
                prob2=prediction[0][1]
                if prob1 < prob2 :
                    prob = prob2 
                else :
                    prob = prob1
                if result == [0]:
                    result= 'Normal'
                else:
                    result = 'Attack'
                dataset='NSL-KDD'
            else : 
                if 'dur'in df.columns:
                    df.loc[df['attack_cat'] != 'Normal', 'attack_cat'] = 1
                    df.loc[df['attack_cat'] == 'Normal', 'attack_cat'] = 0  
                    df['attack_cat'] = df['attack_cat'].astype(int)
                    log_instance = preprocessing_transformer_UNSW['log_transformation'](df)
                    encoded_instance=log_instance
                    encoder = preprocessing_transformer_UNSW['label_encoder']
                    for feature, le in encoder.items():
                        encoded_instance[feature] = le.transform(encoded_instance[feature])

                    selected_features =  preprocessing_transformer_UNSW['selected_features']

                    scaler = preprocessing_transformer_UNSW['scaler']
                    normalized_instance = pd.DataFrame(scaler.transform(encoded_instance), columns=encoded_instance.columns)
                    normalized_instance =normalized_instance[selected_features] 
                    result = estimator_model_knn_UNSW.predict(normalized_instance)
                    prediction=estimator_model_knn_UNSW.predict_proba(normalized_instance)
                    prob1=prediction[0][0]
                    prob2=prediction[0][1]
                    if prob1 < prob2 :
                        prob = prob2 
                    else :
                        prob = prob1
                    if result == [0]:
                        result= 'Normal'
                    else:
                        result = 'Attack'
                    dataset='UNSW-NB15'  
                else:
                    return jsonify({'error': 'Verify your CSV file'})

            last_feature = df.iloc[:, -1]  # Get the last column

            if last_feature.iloc[-1]==0 and result=='Normal' :
                pred= 'True Normal'
            if last_feature.iloc[-1] ==0 and result=='Attack' :
                pred= 'False Attack'
            if last_feature.iloc[-1]==1 and result=='Normal' :
                pred= 'False Normal'
            if last_feature.iloc[-1]==1 and result=='Attack' :
                 pred='True Attack'
            classifier='K-nearest Neighbors'
            new_result = Result(filename, result, dataset, classifier)
            db.session.add(new_result)
            db.session.commit()
            prob=prob*100
        response = {
            'result': result,
            'filename': filename,
            'pred':pred,
            'prob':round(prob, 2)
        }
        return json.dumps(response)


@app.route('/ScanWithSVM', methods=['GET', 'POST'])
def svmm():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Get the directory path of the uploaded file
            #file_directory = os.path.dirname(file_path)

            # Read the .txt file into a DataFrame
            df = pd.read_csv(file_path)
            
            if 'duration' in df.columns:
                df.loc[df['class'] != 'Normal', 'class'] = 1
                df.loc[df['class'] == 'Normal', 'class'] = 0 
                df['class'] = df['class'].astype(int)
                encoded_instance = preprocessing_transformer['encoder'](df)
                selected_features =  preprocessing_transformer['selected_features']
                scaler = preprocessing_transformer['scaler']
                normalized_instance = pd.DataFrame(scaler.transform(encoded_instance), columns=encoded_instance.columns)
                normalized_instance =normalized_instance[selected_features] 
                estimator_model_svm.probability = True
                result = estimator_model_svm.predict(normalized_instance)
                prediction=estimator_model_svm.predict_proba(normalized_instance)
                prob1=prediction[0][0]
                prob2=prediction[0][1]
                if prob1 < prob2 :
                    prob = prob2 
                else :
                    prob = prob1
                if result == [0]:
                    result= 'Normal'
                else:
                    result = 'Attack'
                dataset='NSL-KDD'
            else : 
                if 'dur' in df.columns:
                    df.loc[df['attack_cat'] != 'Normal', 'attack_cat'] = 1
                    df.loc[df['attack_cat'] == 'Normal', 'attack_cat'] = 0  
                    df['attack_cat'] = df['attack_cat'].astype(int)
                    log_instance = preprocessing_transformer_UNSW['log_transformation'](df)
                    encoded_instance=log_instance
                    encoder = preprocessing_transformer_UNSW['label_encoder']
                    for feature, le in encoder.items():
                        encoded_instance[feature] = le.transform(encoded_instance[feature])

                    selected_features =  preprocessing_transformer_UNSW['selected_features']

                    scaler = preprocessing_transformer_UNSW['scaler']
                    normalized_instance = pd.DataFrame(scaler.transform(encoded_instance), columns=encoded_instance.columns)
                    normalized_instance =normalized_instance[selected_features] 
                    estimator_model_svm_UNSW.probability = True
                    result = estimator_model_svm_UNSW.predict(normalized_instance)
                    prediction=estimator_model_svm_UNSW.predict_proba(normalized_instance)
                    prob1=prediction[0][0]
                    prob2=prediction[0][1]
                    if prob1 < prob2 :
                        prob = prob2 
                    else :
                        prob = prob1
                    if result == [0]:
                        result= 'Normal'
                    else:
                        result = 'Attack'
                    dataset='UNSW-NB15' 
                else:
                    return jsonify({'error': 'Verify your CSV file'})

            last_feature = df.iloc[:, -1]  # Get the last column

            if last_feature.iloc[-1]==0 and result=='Normal' :
                pred= 'True Normal'
            if last_feature.iloc[-1] ==0 and result=='Attack' :
                pred= 'False Attack'
            if last_feature.iloc[-1]==1 and result=='Normal' :
                pred= 'False Normal'
            if last_feature.iloc[-1]==1 and result=='Attack' :
                 pred='True Attack'
            classifier='Support Vector Machine'
            new_result = Result(filename, result, dataset, classifier)
            db.session.add(new_result)
            db.session.commit()
            prob=prob*100
        response = {
            'result': result,
            'filename': filename,
            'pred':pred,
            'prob':round(prob, 2)
        }
        return json.dumps(response)
    

@app.route('/ScanWithXGB', methods=['GET', 'POST'])
def xgbb():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Get the directory path of the uploaded file
            #file_directory = os.path.dirname(file_path)

            # Read the .txt file into a DataFrame
            df = pd.read_csv(file_path)
            
            if 'duration' in df.columns:
                df.loc[df['class'] != 'Normal', 'class'] = 1
                df.loc[df['class'] == 'Normal', 'class'] = 0 
                df['class'] = df['class'].astype(int)
                encoded_instance = preprocessing_transformer['encoder'](df)
                selected_features =  preprocessing_transformer['selected_features']
                scaler = preprocessing_transformer['scaler']
                normalized_instance = pd.DataFrame(scaler.transform(encoded_instance), columns=encoded_instance.columns)
                normalized_instance =normalized_instance[selected_features] 
                result = estimator_model_xgb.predict(normalized_instance)
                prediction=estimator_model_xgb.predict_proba(normalized_instance)
                prob1=prediction[0][0]
                prob2=prediction[0][1]
                if prob1 < prob2 :
                    prob=prob2 
                else :
                    prob = prob1
                #float32_value = np.float32(prob)
                # Convert float32 to float and serialize to JSON in one line
                prob = json.dumps(float(prob))
                if result == [0]:
                    result= 'Normal'
                else:
                    result = 'Attack'
                dataset='NSL-KDD'
            else : 
                if 'dur' in df.columns:
                    df.loc[df['attack_cat'] != 'Normal', 'attack_cat'] = 1
                    df.loc[df['attack_cat'] == 'Normal', 'attack_cat'] = 0  
                    df['attack_cat'] = df['attack_cat'].astype(int)
                    log_instance = preprocessing_transformer_UNSW['log_transformation'](df)
                    encoded_instance=log_instance
                    encoder = preprocessing_transformer_UNSW['label_encoder']
                    for feature, le in encoder.items():
                        encoded_instance[feature] = le.transform(encoded_instance[feature])

                    selected_features =  preprocessing_transformer_UNSW['selected_features']

                    scaler = preprocessing_transformer_UNSW['scaler']
                    normalized_instance = pd.DataFrame(scaler.transform(encoded_instance), columns=encoded_instance.columns)
                    normalized_instance =normalized_instance[selected_features] 
                    result = estimator_model_xgb_UNSW.predict(normalized_instance)
                    prediction=estimator_model_xgb_UNSW.predict_proba(normalized_instance)
                    prob1=prediction[0][0]
                    prob2=prediction[0][1]
                    if prob1 < prob2 :
                        prob=prob2 
                    else :
                        prob = prob1
                    prob = json.dumps(float(prob))

                    #float32_value = np.float32(prob)

                    # Convert float32 to float and serialize to JSON in one line
                    prob = json.dumps(float(prob))

                    if result == [0]:
                        result= 'Normal'
                    else:
                        result = 'Attack'
                    dataset='UNSW-NB15' 
                else:
                    return jsonify({'error': 'Verify your CSV file'})

            last_feature = df.iloc[:, -1]  # Get the last column

            if last_feature.iloc[-1]==0 and result=='Normal' :
                pred= 'True Normal'
            if last_feature.iloc[-1] ==0 and result=='Attack' :
                pred= 'False Attack'           
            if last_feature.iloc[-1]==1 and result=='Normal' :
                pred= 'False Normal'
            if last_feature.iloc[-1]==1 and result=='Attack' :
                 pred='True Attack'
            classifier='XGBoost'
            new_result = Result(filename, result, dataset, classifier)
            db.session.add(new_result)
            db.session.commit()
            prob = "{:.2%}".format(float(prob) )
        response = {
            'result': result,
            'filename': filename,
            'pred':pred,
            'prob':prob
        }
        return json.dumps(response)
    
    
@app.route('/ScanWithLR', methods=['GET', 'POST'])
def lrr():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Get the directory path of the uploaded file
            #file_directory = os.path.dirname(file_path)

            # Read the .txt file into a DataFrame
            df = pd.read_csv(file_path)
            
            if 'duration' in df.columns:
                df.loc[df['class'] != 'Normal', 'class'] = 1
                df.loc[df['class'] == 'Normal', 'class'] = 0 
                df['class'] = df['class'].astype(int)
                encoded_instance = preprocessing_transformer['encoder'](df)
                selected_features =  preprocessing_transformer['selected_features']
                scaler = preprocessing_transformer['scaler']
                normalized_instance = pd.DataFrame(scaler.transform(encoded_instance), columns=encoded_instance.columns)
                normalized_instance =normalized_instance[selected_features] 
                result = estimator_model_lr.predict(normalized_instance)
                prediction=estimator_model_lr.predict_proba(normalized_instance)
                prob1=prediction[0][0]
                prob2=prediction[0][1]
                if prob1 < prob2 :
                    prob=prob2 
                else :
                    prob = prob1
                if result == [0]:
                    result= 'Normal'
                else:
                    result = 'Attack'
                dataset='NSL-KDD'
            else : 
                if 'dur' in df.columns:
                    df.loc[df['attack_cat'] != 'Normal', 'attack_cat'] = 1
                    df.loc[df['attack_cat'] == 'Normal', 'attack_cat'] = 0  
                    df['attack_cat'] = df['attack_cat'].astype(int)
                    log_instance = preprocessing_transformer_UNSW['log_transformation'](df)
                    encoded_instance=log_instance
                    encoder = preprocessing_transformer_UNSW['label_encoder']
                    for feature, le in encoder.items():
                        encoded_instance[feature] = le.transform(encoded_instance[feature])

                    selected_features =  preprocessing_transformer_UNSW['selected_features']

                    scaler = preprocessing_transformer_UNSW['scaler']
                    normalized_instance = pd.DataFrame(scaler.transform(encoded_instance), columns=encoded_instance.columns)
                    normalized_instance =normalized_instance[selected_features] 
                    result = estimator_model_lr_UNSW.predict(normalized_instance)
                    prediction=estimator_model_lr_UNSW.predict_proba(normalized_instance)
                    prob1=prediction[0][0]
                    prob2=prediction[0][1]
                    if prob1 < prob2 :
                        prob=prob2 
                    else :
                        prob = prob1
                    if result == [0]:
                        result= 'Normal'
                    else:
                        result = 'Attack'
                    dataset='UNSW-NB15' 
                else:
                    return jsonify({'error': 'Verify your CSV file'})

            last_feature = df.iloc[:, -1]  # Get the last column

            if last_feature.iloc[-1]==0 and result=='Normal' :
                pred= 'True Normal'
            if last_feature.iloc[-1] ==0 and result=='Attack' :
                pred= 'False Attack'            
            if last_feature.iloc[-1]==1 and result=='Normal' :
                pred= 'False Normal'
            if last_feature.iloc[-1]==1 and result=='Attack' :
                 pred='True Attack'
            classifier='Logistic Regression'
            new_result = Result(filename, result, dataset, classifier)
            db.session.add(new_result)
            db.session.commit()
            prob=prob*100
        response = {
            'result': result,
            'filename': filename,
            'pred':pred,
            'prob':round(prob, 2)
        }
        return json.dumps(response)


@app.route('/results', methods=['GET'])
def get_results():
    results = Result.query.all()
    result_list = []
    for result in results:
        result_data = {
            'id': result.id,
            'filename': result.filename,
            'result': result.result,
            'dataset': result.dataset,
            'classifier':result.classifier,
            'date': result.date.strftime('%Y-%m-%d %H:%M:%S') , # Format the date as a string
            'archive': result.archive
        }
        result_list.append(result_data)
    return jsonify(result_list)


@app.route('/Dashboard', methods=['GET'])
def get_data():
    conn = sqlite3.connect('results.db')
    cursor = conn.cursor()
    # Execute a query to retrieve data from the table
    cursor.execute("SELECT id,strftime('%d-%m-%Y', date), filename, dataset, result, classifier FROM Result WHERE archive=False")

    data = cursor.fetchall()
    
    # Close the database connection
    cursor.close()
    conn.close()
    
    # Convert the data to a JSON response
    result = [
        {
            'id': row[0],
            'date': row[1],
            'filename': row[2],
            'datasetname': row[3],
            'classifier':row[5],
            'result': row[4]
        }
        for row in data
    ]
    return jsonify(result)


@app.route('/Archive', methods=['GET'])
def get_archeive():
    conn = sqlite3.connect('results.db')
    cursor = conn.cursor()
    # Execute a query to retrieve data from the table
    cursor.execute("SELECT id,strftime('%d-%m-%Y', date), filename, dataset, result ,classifier FROM Result where archive= True")
    data = cursor.fetchall()
    # Close the database connection
    cursor.close()
    conn.close()
    # Convert the data to a JSON response
    result = [
        {
            'id': row[0],
            'date': row[1],
            'filename': row[2],
            'datasetname': row[3],
            'classifier':row[5],
            'result': row[4]
        }
        for row in data
    ]
    return jsonify(result)
    

@app.route('/Dashboard/<int:id>', methods=['PATCH'])
def archive_result(id):
    result = Result.query.get(id)
    if result:
        result.archive = True
        db.session.commit()
        return jsonify({'message': 'Result archived successfully'})
    else:
        return jsonify({'message': 'Result not found'}), 404


# to delete all rows in table results

# @app.route('/delete_all_rows', methods=['POST','GET'])
# def delete_all_rows():
#     try:
#         conn = sqlite3.connect('results.db')

#         # Create a cursor
#         cursor = conn.cursor()

#         # Execute the DELETE statement
#         cursor.execute('DELETE FROM Result')

#         # Commit the changes
#         conn.commit()

#         # Close the cursor and connection
#         cursor.close()
#         conn.close()
#         return 'All rows deleted successfully'
#     except Exception as e:
#         db.session.rollback()
#         return f'Error: {str(e)}'


@app.route('/api/filter', methods=['GET'])
def filter_data():
    filters = {
        'name': request.args.get('name'),
        'date': request.args.get('date'),
        'filename': request.args.get('filename')
    }

    # Perform any necessary data validation or parsing here

    # Construct your database query based on the filter parameters
    query =  sessionmaker.Result.query

    if filters['name']:
        query = query.filter( sessionmaker.Result.name.ilike(f'%{filters["name"]}%'))

    if filters['date']:
        date_value = datetime.strptime(filters['date'], '%Y-%m-%d').date()
        query = query.filter( sessionmaker.Result.date_field == date_value)

    if filters['filename']:
        query = query.filter( sessionmaker.Result.filename.ilike(f'%{filters["filename"]}%'))

    # Execute the query and retrieve the filtered data
    filtered_data = query.all()

    # Convert the filtered data to a format suitable for sending as a response
    serialized_data = [item.serialize() for item in filtered_data]

    return jsonify(serialized_data)



if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=False)