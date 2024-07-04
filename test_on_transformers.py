from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch.nn.functional as F

# Load the image processor and model
processor = AutoImageProcessor.from_pretrained("RavenOnur/Sign-Language")
model = AutoModelForImageClassification.from_pretrained("RavenOnur/Sign-Language")

# Load a local image
image_path = "./test_1.png"
image = Image.open(image_path)

# Preprocess the image
inputs = processor(images=image, return_tensors="pt")

# Perform inference
outputs = model(**inputs)

# Get the predicted class with the highest probability
probs = F.softmax(outputs.logits, dim=1)
predicted_class = probs.argmax(dim=1).item()

print(f"Predicted class: {predicted_class}")
