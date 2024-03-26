from flask import Flask, request, jsonify
from gliner import GLiNER
from collections import defaultdict

app = Flask(__name__)

model = GLiNER.from_pretrained("urchade/gliner_largev2")

@app.route('/', methods=['POST'])
def gliner():
    # Get the JSON data from the request
    if request.is_json:
        # Handle JSON data
        data = request.get_json()
    else:
        # Handle form data
        data = request.form.to_dict()
    # Extract context and question from the JSON data

    text = data.get('text')
    labels = data.get('labels')

    entities = model.predict_entities(text, labels)
    answer = defaultdict(list)
    for entity in entities:
        answer[entity["label"]].append(entity["text"])
    # Return the answer as JSON response
    return jsonify({"data": answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)