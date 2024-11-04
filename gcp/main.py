from google.cloud import storage
import tensorflow as tf
from flask import jsonify
from PIL import Image
import numpy as np
import os
import sys

BUCKET_NAME = "potatso-disease"
class_names = ["Late Blight", "Early Blight", "Healthy"]
model = None


def verify_model_file():
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob("models/model.h5")

        if not blob.exists():
            print("ERROR: Model file not found in bucket")
            return False

        print(f"Model file exists in bucket, size: {blob.size} bytes")
        return True
    except Exception as e:
        print(f"Error checking model file: {str(e)}")
        return False


def download_blob_if_needed():
    global model
    try:
        if model is None:
            print("Starting model download...")
            if not verify_model_file():
                raise Exception("Model file verification failed")

            model_path = "/tmp/model.h5"
            storage_client = storage.Client()
            bucket = storage_client.bucket(BUCKET_NAME)
            blob = bucket.blob("models/model.h5")
            blob.download_to_filename(model_path)

            print(f"Model downloaded to {model_path}")
            print(f"Model file size: {os.path.getsize(model_path)} bytes")

            model = tf.keras.models.load_model(model_path)
            print("Model architecture:")
            model.summary(print_fn=lambda x: print(x))
            print("Model loaded successfully")
    except Exception as e:
        print(f"Error in download_blob_if_needed: {str(e)}")
        print(f"Python version: {sys.version}")
        print(f"TensorFlow version: {tf.__version__}")
        raise e


def predict(request):
    try:
        global model
        print("\n=== Starting new prediction request ===")

        download_blob_if_needed()

        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        image = request.files["file"]
        if not image.filename:
            return jsonify({"error": "No selected file"}), 400

        print(f"Processing file: {image.filename}")

        # Save incoming image for debugging
        debug_path = "/tmp/debug_image.jpg"
        with open(debug_path, 'wb') as f:
            image.save(f)
        print(f"Saved debug image, size: {os.path.getsize(debug_path)} bytes")

        # Detailed preprocessing logging
        image = Image.open(image)
        print(f"Original image size: {image.size}, mode: {image.mode}")

        image = image.convert("RGB")
        print("Converted to RGB")

        image = image.resize((256, 256), Image.BICUBIC)
        print(f"Resized image: {image.size}")

        image_array = np.array(image, dtype=np.float32)
        print(f"NumPy array shape before normalization: {image_array.shape}")
        print(
            f"Array stats before norm - min: {image_array.min()}, max: {image_array.max()}, mean: {image_array.mean()}")

        image_array = image_array / 255.0
        print(
            f"Array stats after norm - min: {image_array.min()}, max: {image_array.max()}, mean: {image_array.mean()}")

        img_array = np.expand_dims(image_array, axis=0)
        print(f"Final input shape: {img_array.shape}")

        # Make prediction with detailed logging
        print("Starting prediction...")
        predictions = model.predict(img_array, verbose=1)
        print(f"Raw prediction output: {predictions}")
        print(f"Prediction shape: {predictions.shape}")

        predicted_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_index]
        confidence = float(np.max(predictions[0])) * 100

        print(f"Predicted class index: {predicted_index}")
        print(f"Predicted class name: {predicted_class}")
        print(f"Confidence: {confidence}%")

        # Detailed predictions for all classes
        class_predictions = {
            class_name: float(pred) * 100
            for class_name, pred in zip(class_names, predictions[0])
        }
        print("All class predictions:", class_predictions)

        return jsonify({
            "class": predicted_class,
            "confidence": confidence,
            "all_predictions": class_predictions
        })

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        print(f"Stack trace:", sys.exc_info())
        return jsonify({"error": str(e)}), 400