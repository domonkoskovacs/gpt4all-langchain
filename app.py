import time

from flask import Flask, request, jsonify
from flask_cors import CORS
from chat import answer_question
from ingest_text import ingest_text

app = Flask(__name__)
CORS(app)
CORS(app, origins='http://localhost:3000')


@app.route('/ingest', methods=["POST"])
def ingest():
    """
    This function indexes a text, role must be "USER" or "EMPLOYEE", example request body:
    { "text": "In germany it
    is illegal to go faster than 460 km/h on the highway because of some airline laws. The laws say that under 3000
    meters nothing can go faster than 460 km/h.", "content_name": "german_speed_limit", "role": "USER" }
    :return:
    """
    req = request.get_json()
    text = req.get('text')
    content_name = req.get('content_name')
    role = req.get('role')

    ingest_text(text, content_name, role)
    return "Text indexed successfully"


@app.route('/answer', methods=['POST'])
def get_answer():
    """
    This function answers a question, example request body:
    {
        "question":"What is langchain?",
        "prompt_type":"SHORT",
        "role":"USER"
    }
    :return: answer, answer_time
    """
    if request.method == 'POST':
        req = request.get_json()
        question = req.get('question')
        prompt_type = req.get('prompt_type')
        role = req.get('role')

        start_time = time.time()
        answer = answer_question(question, prompt_type, role)
        end_time = time.time()
        response_time = end_time - start_time

        response = {
            'answer_message': answer,
            'response_time': response_time
        }
        return jsonify(response)


if __name__ == '__main__':
    app.run()
