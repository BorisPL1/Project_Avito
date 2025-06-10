import requests
from flask import Flask, request, jsonify, render_template_string
import logging
import polars as pl
from datasets import Dataset
from model_class import SentenceClassifier
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
model = SentenceClassifier()
model.load_from_file("/Studying/AAA/Term2/First_model/First_model")

def convert_col(data):
    tone_col = list(data["tone"])
    for i in range(len(tone_col)):
        if tone_col[i] == 0:
            tone_col[i] = 'Оскорбление'
        elif tone_col[i] == 1:
            tone_col[i] = 'Другое'
        elif tone_col[i] == 2:
            tone_col[i] = 'Домогательство'
        elif tone_col[i] == 3:
            tone_col[i] = 'Угроза'
    data["tone"] = tone_col
    return data

@app.route('/read_msgs', methods=['POST'])
def read_msg_many():
    data = request.get_json()
    if not data:
        return 'No data received', 400
    
    data1 = Dataset.from_dict(data)
    prediction = model.predict_tone(data1)
    prediction = convert_col(prediction)
    prediction = {"tone": prediction["tone"].values.tolist()}
    return json.dumps(prediction)

if __name__ == '__main__':
    logging.basicConfig(
        format='[%(levelname)s] [%(asctime)s] %(message)s',
        level=logging.INFO,
    )

    app.run(host='0.0.0.0', port=8080, debug=True)
