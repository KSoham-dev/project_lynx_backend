from speciesnet import SpeciesNet
import requests
from PIL import Image
from io import BytesIO
import os

model_path = '/content/'
try:
    model = SpeciesNet(model_path)
    print("SpeciesNet model instantiated.")
except Exception as e:
    print(f"Error instantiating SpeciesNet model: {e}")
image_save_path = 'https://cdn.discordapp.com/attachments/1384169300545503302/1477200741818171478/image.png?ex=69a3e5e5&is=69a29465&hm=f15986a34a5c4d3aa220bcff9f528978a6036fa7cea5533eb7dbd7d8c40d1323&'
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
        print(f"Result: {prediction['classifications']["classes"][0]}")
    else:
        print("No predictions returned.")
except Exception as e:
    print(f"Error during prediction: {e}")