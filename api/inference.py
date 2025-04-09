import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
import os

# ✅ Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model("efficientnet_b5", pretrained=False, num_classes=131)

model_path = r"D:\projects\smartpest\smartpest_backend\models\best_model_b5.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval().to(device)

# ✅ Define transforms
transform = transforms.Compose([
    transforms.Resize((600, 600)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ✅ Class labels
class_file = r"D:\projects\smartpest\smartpest_backend\classes.txt"
with open(class_file, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

def predict_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {"error": f"Failed to open image: {e}"}

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()

    return {
        "class": class_names[pred_idx],
        "confidence": round(probs[0][pred_idx].item(), 4)
    }
