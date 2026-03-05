from speciesnet import SpeciesNet
import requests
from PIL import Image
from io import BytesIO
import os

model_path = './model'
try:
    model = SpeciesNet(model_path)
    print("SpeciesNet model instantiated.")
except Exception as e:
    print(f"Error instantiating SpeciesNet model: {e}")
image_save_path = 'https://inaturalist-open-data.s3.amazonaws.com/photos/605060662/large.jpg'
instances = [{
    'filepath': image_save_path,
    'latitude': 34.08,
    'longitude': 74.8056
}]

print("Running prediction...")
try:
    predictions_dict = model.predict(instances_dict={"instances": instances}) # type: ignore
    if predictions_dict and "predictions" in predictions_dict and predictions_dict["predictions"]:
        prediction = predictions_dict["predictions"][0]
        print("\n--- India Species Prediction ---")
        print(f"Result: {prediction['classifications']['classes'][0]}")
    else:
        print("No predictions returned.")
except Exception as e:
    print(f"Error during prediction: {e}")