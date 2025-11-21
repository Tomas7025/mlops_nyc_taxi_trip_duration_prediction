from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

def load_model(path):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

MODEL_PATH = "models/random_forest.pkl"
model = load_model(MODEL_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        user_input = data["inputs"]

        X = np.array(user_input).reshape(1, -1)

        prediction = model.predict(X).tolist()

        return jsonify({
            "model": model.__class__.__name__,
            "prediction": prediction
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


def main():
    app.run(host="0.0.0.0", port=5000, debug=True)


if __name__ == "__main__":
    main()

#curl -X POST http://127.0.0.1:5000/predict \
#     -H "Content-Type: application/json" \
#     -d '{"inputs": [5.1, 3.5, 1.4, 0.2]}'
