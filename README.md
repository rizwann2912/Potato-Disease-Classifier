Potato Disease Classification 🌱
This project uses machine learning to assist farmers in identifying common potato plant diseases from leaf images, including early blight, late blight, and healthy plants. The project is built with FastAPI for the backend and React for the frontend and is deployed on Google Cloud Platform (GCP) for easy access and scalability.

Table of Contents
Overview
Features
Model
Tech Stack
Project Structure
Setup and Installation
API Documentation
Usage
Contributing
License
Overview
This Potato Disease Classification project aims to help farmers easily detect potato diseases by simply capturing a photo of potato leaves. The model accurately predicts whether the plant has early blight, late blight, or is healthy, with an accuracy of 98%. The goal is to support farmers in making informed decisions to manage their crops effectively.

Features
Image-Based Disease Detection: Upload a photo of potato leaves, and the model classifies it.
REST API: FastAPI backend handles requests, making it scalable and fast.
Frontend Interface: Built with React for an intuitive, user-friendly experience.
Google Cloud Integration: The model and API are deployed on GCP, allowing seamless access to the prediction service.
Model
Algorithm: A CNN (Convolutional Neural Network) trained to classify potato leaf images into three categories.
Accuracy: 98% on test data.
Deployment: Model is hosted on Google Cloud Storage, with prediction API available via FastAPI.
Tech Stack
Backend: FastAPI, Python
Frontend: React
Cloud Platform: Google Cloud Platform (GCP)
Machine Learning: CNN Model
Database: Google Cloud Storage for model storage
Project Structure
bash
Copy code
potato-disease-classification/
│
├── frontend/                   # React frontend files
├── backend/                    # FastAPI backend files
│   ├── main.py                 # Main API file
│   ├── requirements.txt        # Python dependencies
├── model/                      # Model files stored on GCP
├── gcp-config/                 # GCP configuration files
│
└── README.md                   # Project documentation
Setup and Installation
Prerequisites
Python 3.12 or above
Node.js (for frontend)
Google Cloud SDK
Backend Setup (FastAPI)
Clone the repository:
bash
Copy code
git clone https://github.com/your-username/potato-disease-classification.git
cd potato-disease-classification/backend
Install dependencies:
bash
Copy code
pip install -r requirements.txt
Run the FastAPI backend locally:
bash
Copy code
uvicorn main:app --reload
Frontend Setup (React)
Navigate to the frontend directory:
bash
Copy code
cd ../frontend
Install dependencies:
bash
Copy code
npm install
Start the React app:
bash
Copy code
npm start
GCP Deployment
Model Deployment: Upload the trained model to Google Cloud Storage.
API Deployment: Deploy the FastAPI backend on GCP (e.g., Cloud Run).
Frontend Deployment: Deploy the React frontend to Google App Engine or another hosting service on GCP.
API Documentation
The backend API endpoints are defined using FastAPI and include endpoints for:

/predict: Accepts an image file and returns the predicted class (early blight, late blight, or healthy).
Example request using Postman:

http
Copy code
POST /predict
Content-Type: multipart/form-data
Body: [Upload an image file]
Response:

json
Copy code
{
  "class": "early_blight",
  "confidence": 0.98
}
Usage
Start both the backend and frontend locally or access the deployed version online.
Upload an image of a potato leaf via the frontend.
View the classification result, indicating if the leaf is healthy or has a disease (early or late blight).
Contributing
Contributions are welcome! Please fork the repository and create a pull request to suggest improvements or add features.

License
This project is licensed under the MIT License. See the LICENSE file for more details.
