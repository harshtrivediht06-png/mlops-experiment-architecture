# MLOps Pipeline: Breast Cancer Classification

**Developed by:** Harsh Trivedi

This repository contains a complete, modular Machine Learning Operations (MLOps) pipeline. It encompasses model training, evaluation, and batch prediction workflows, alongside a fully containerized FastAPI service for real-time inference.

## 1. Local Environment Setup
Ensure your virtual environment is activated, then install the required dependencies:
```cmd
pip install --no-cache-dir -r requirements.txt
```

## 2. ML Pipeline Execution (Training, Evaluation, Prediction)
The core machine learning logic is split into modular scripts for maintainability.

* **Training:** 
  Run the training script to build the Scikit-learn pipeline, train the Random Forest model, and save the artifact (`models/model.joblib`).
  ```cmd
  python -m src.train
  ```
* **Evaluation:** 
  Generate accuracy metrics and classification reports on the test dataset.
  ```cmd
  python -m src.evaluate
  ```
* **Batch Prediction:** 
  Load the saved `.joblib` model to generate predictions on sample processed test data.
  ```cmd
  python -m src.predict
  ```

## 3. API Schema and Run Instructions
The trained model is served via a FastAPI REST interface. 

### Running the API Locally
Launch the Uvicorn server as a module:
```cmd
python -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```
Access the auto-generated Swagger UI for interactive testing at `http://127.0.0.1:8000/docs`.

### API Schema Documentation
* **`GET /health`**
  * **Description:** Health check endpoint to verify service uptime.
  * **Response:** `200 OK` with JSON `{"status": "ok"}`.
* **`POST /predict`**
  * **Description:** Generates a breast cancer classification prediction based on input features.
  * **Payload Schema (`PredictionInput`):** Requires exactly 30 numerical features (float) inside a JSON array.
    ```json
    {
      "features": [
        17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
        1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
        25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
      ]
    }
    ```
  * **Success Response:** `200 OK` with JSON `{"prediction": 0}` (or `1`).
  * **Error Handling:** Returns `422 Unprocessable Entity` if the array contains anything other than exactly 30 features.

## 4. Docker Containerization 
To guarantee consistent execution without dependency conflicts across different machines, the API is packaged into a Linux-based Docker container.

### Build the Image
Ensure Docker Desktop is running, then execute the following command from the project root directory:
```cmd
docker build -t ml-inference-api:1.0 .
```

### Run the Container
Launch the container and map the internal port 8000 to your local machine:
```cmd
docker run -p 8000:8000 ml-inference-api:1.0
```
The API is now fully isolated and accessible at `http://127.0.0.1:8000/docs`.