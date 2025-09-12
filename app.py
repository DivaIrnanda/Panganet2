from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# ====== LOAD MODEL DAN SCALER ======
try:
    model = pickle.load(open("best_model_XGB.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
except FileNotFoundError:
    class MockModel:
        def predict(self, X):
            return [np.mean(X) * 0.8 + 20]
    class MockScaler:
        def transform(self, X):
            return X
    model = MockModel()
    scaler = MockScaler()

# ====== ROUTE HALAMAN ======
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/prediksi')
def prediksi_page():
    return render_template("prediksi.html")

# ====== ROUTE PREDIKSI ======
@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_data = request.get_json()
        data = [
            json_data['fitur1'], json_data['fitur2'], json_data['fitur3'],
            json_data['fitur4'], json_data['fitur5'], json_data['fitur6'],
            json_data['fitur7'], json_data['fitur8'], json_data['fitur9']
        ]
        features = np.array(data).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        output = float(prediction[0])
        return jsonify({"prediction": round(output, 2)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
