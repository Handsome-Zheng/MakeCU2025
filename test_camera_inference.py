import cv2
import torch
import timm
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from picamera2 import Picamera2

# ===================== CONFIG =====================
MODEL_PATH = 'best_waste_classifier.pth'
MODEL_NAME = 'efficientnet_b3'
IMG_SIZE = 300

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===================== TRANSFORMS =====================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ===================== MODEL DEFINITION =====================
class WasteClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super(WasteClassifier, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        num_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

# ===================== LOAD MODEL =====================
checkpoint = torch.load(MODEL_PATH, map_location=device)
class_names = checkpoint['class_names']
num_classes = len(class_names)

model = WasteClassifier(MODEL_NAME, num_classes)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

print(f"Model loaded successfully with {num_classes} classes.")
print("Classes:", class_names)

# ===================== CAMERA SETUP =====================
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

print("Camera started. Press 'q' to quit.")

# ===================== LIVE INFERENCE LOOP =====================
while True:
    frame = picam2.capture_array()  # Capture frame as NumPy array (RGB)
    img = Image.fromarray(frame).convert('RGB')

    # Preprocess
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = class_names[predicted.item()]
    confidence_score = confidence.item() * 100

    # Display result
    display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    text = f"{predicted_class} ({confidence_score:.1f}%)"
    cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Waste Classifier - Raspberry Pi Camera", display_frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()