from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from models.model import model_absa, tokenizer, no_grad
from celery import Celery
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import datetime
from celery.exceptions import SoftTimeLimitExceeded

# Inisialisasi Firebase Admin SDK
cred = credentials.Certificate('firebase-admin-sdk.json')
firebase_admin.initialize_app(cred)

# Inisialisasi Firestore client
db = firestore.client()

app = Flask(__name__)

# Konfigurasi Celery
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)
celery.conf.update(
    CELERY_INCLUDE=['main']
)

@celery.task(soft_time_limit=600, time_limit=620, autoretry_for=(SoftTimeLimitExceeded,), retry_kwargs={'max_retries': 3, 'countdown': 20})
def predict_reviews(review_data, max_sequence_length, id):
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
    doc_ref = db.collection('users').document(id).collection('task_list').document(id + "_" + str(predict_reviews.request.id))
    data = {
        'createdAt': formatted_time,
        'status': 'Pending'
    }
    doc_ref.set(data)

    model_absa.eval()
    results = []

    for review_text in review_data:
        # Tokenize the review text
        encoded_input = tokenizer(review_text, padding='max_length', truncation=True, max_length=max_sequence_length, return_tensors='pt')

        input_ids = encoded_input['input_ids']
        attention_mask = encoded_input['attention_mask']

        with no_grad:
            sentiment_pred, confidence_pred = model_absa(input_ids, attention_mask)

        sentiment_pred = sentiment_pred.cpu().numpy()
        confidence_pred = confidence_pred.cpu().numpy()

        sentiment_pred = np.round(sentiment_pred).astype(int)
        confidence_pred = np.round(confidence_pred).astype(int)

        # Convert arrays to JSON strings
        sentiment_pred_json = sentiment_pred.tolist()[0]
        confidence_pred_json = confidence_pred.tolist()[0]

        row = {"Text": review_text, "Sentiment": sentiment_pred_json, "Confidence": confidence_pred_json}
        results.append(row)

    doc_ref = db.collection('results').document(id+"_"+str(predict_reviews.request.id))
    doc_ref.set({"data" : results})
    doc_ref = db.collection('users').document(id).collection('task_list').document(id + "_" + str(predict_reviews.request.id))
    doc_ref.update({
        'status' : 'Finish'
    })

    # return results

@app.route('/predict', methods=['POST'])
def predict():
    id = request.form.get('id')
    string = request.form.get('text')

    if string:
        result = predict_reviews.apply_async(args=[[text], 64, id])

        response = {
            'message': 'Your Request Has been Submitted',
            'id_task': id + "_" + str(result)
        }
        return jsonify(response), 202

    if 'file' not in request.files:
        return jsonify({'message': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No file selected for uploading'}), 400

    if file and (file.filename.endswith('.csv') or file.filename.endswith('.xlsx')):
        # try:
        # Try Read CSV
        df = None
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        # print(df.values.flatten())
        result = predict_reviews.apply_async(args=[df['text'].tolist(), 64, id])
        # predict_reviews(df['text'].tolist(), 64, id)

        response = {
            'message': 'Your Request Has been Submitted',
            'id_task': id+"_"+str(result)
        }
        return jsonify(response), 202
        # except Exception as e:
        #     return jsonify({'message': f'Error processing file: {str(e)}'}), 500
    else:
        return jsonify({'message': 'Invalid file format. Please upload a CSV file.'}), 400

@app.route('/check-task/<task_id>', methods=['GET'])
def check_task(task_id):
    task = predict_reviews.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'status': 'Pending'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'result': task.result
        }
    else:
        response = {
            'state': task.state,
            'status': str(task.info)
        }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)