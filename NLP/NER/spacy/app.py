import spacy
from flask import Flask, request, jsonify
from collections import defaultdict

app = Flask(__name__)

nlp = spacy.load("en_core_web_sm")
# load the best model
nlp = spacy.load(r"/content/drive/MyDrive/output/model-best")


@app.route('/', methods=['POST'])
def spacy_ner():
    # Get the JSON data from the request
    if request.is_json:
        # Handle JSON data
        data = request.get_json()
    else:
        # Handle form data
        data = request.form.to_dict()
    print(data)
    # Extract context and question from the JSON data
    text = data.get('text')
    labels = data.get('labels')
    doc = nlp(text)
    answer = defaultdict(list)
    for ent in doc.ents:
        answer[ent.label_].append(ent.text)
    print(answer)
    # Return the answer as JSON response
    return jsonify({"data": answer})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5004)