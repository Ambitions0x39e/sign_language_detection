import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog
from PySide6.QtGui import QPixmap
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch.nn.functional as F

class SignLanguageClassifier(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.processor = AutoImageProcessor.from_pretrained("RavenOnur/Sign-Language")
        self.model = AutoModelForImageClassification.from_pretrained("RavenOnur/Sign-Language")
        self.class_dict = {
            "0": "A", "1": "B", "10": "L", "11": "M", "12": "N", "13": "O", "14": "P",
            "15": "Q", "16": "R", "17": "S", "18": "T", "19": "U", "2": "C", "20": "V",
            "21": "W", "22": "X", "23": "Y", "3": "D", "4": "E", "5": "F", "6": "G",
            "7": "H", "8": "I", "9": "K"
        }

    def initUI(self):
        self.setWindowTitle("Sign Language Classifier")
        self.setGeometry(100, 100, 600, 400)
        
        self.layout = QVBoxLayout()

        self.label = QLabel("Select an image to classify")
        self.layout.addWidget(self.label)

        self.button = QPushButton("Choose Image")
        self.button.clicked.connect(self.openFileDialog)
        self.layout.addWidget(self.button)

        self.resultLabel = QLabel("")
        self.layout.addWidget(self.resultLabel)

        self.setLayout(self.layout)

    def openFileDialog(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Images (*.png *.xpm *.jpg *.jpeg)")
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            pixmap = QPixmap(file_path)
            pixmap = self.scalePixmap(pixmap, max_size=512)
            self.label.setPixmap(pixmap)
            self.classifyImage(file_path)

    def scalePixmap(self, pixmap, max_size=512):
        width = pixmap.width()
        height = pixmap.height()
        if width > max_size or height > max_size:
            ratio = max_size / max(width, height)
            width = int(width * ratio)
            height = int(height * ratio)
            pixmap = pixmap.scaled(width, height)
        return pixmap

    def classifyImage(self, image_path):
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        predicted_class = probs.argmax(dim=1).item()
        self.resultLabel.setText(f"Predicted class: {self.class_dict[str(predicted_class)]}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SignLanguageClassifier()
    window.show()
    sys.exit(app.exec())
