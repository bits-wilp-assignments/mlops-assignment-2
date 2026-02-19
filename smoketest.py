import requests

def test_health():
    resp = requests.get("http://localhost:8000/health")
    assert resp.status_code == 200
    print("Health OK")

def test_predict():
    files = {"file": open("data/raw/PetImages/Cat/cat.0.jpg","rb")}
    resp = requests.post("http://localhost:8000/predict", files=files)
    assert resp.status_code == 200
    print("Prediction OK", resp.json())

if __name__ == "__main__":
    test_health()
    test_predict()
