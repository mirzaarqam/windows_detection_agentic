import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import cv2

# -------------------------------
# Load Qwen2-VL on CPU (GPU not compatible)
# -------------------------------
model_name = "Qwen/Qwen2-VL-7B-Instruct"

# Force CPU usage since GPU has incompatible CUDA capability
model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    device_map="cpu",
    torch_dtype=torch.float32
)

processor = AutoProcessor.from_pretrained(model_name)

# -------------------------------
# Load image
# -------------------------------
image_path = "room.jpg"
image = Image.open(image_path).convert("RGB")

# -------------------------------
# Ask Qwen2-VL to find window positions
# -------------------------------
prompt = """
You are a vision model. 
Look at the image and give me bounding boxes of all windows. 

Format output ONLY as a Python list like:
[[x1, y1, x2, y2], [x1, y1, x2, y2]]
"""

inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

output = model.generate(**inputs, max_new_tokens=200)
response = processor.decode(output[0], skip_special_tokens=True)

print("\nRAW MODEL OUTPUT:\n", response)

# -------------------------------
# Parse bounding boxes
# -------------------------------
try:
    boxes = eval(response)
except:
    print("Could not parse bounding boxes. Response was:")
    print(response)
    exit()

print("\nDETECTED BOXES:\n", boxes)

# -------------------------------
# Draw boxes on image
# -------------------------------
img_cv = cv2.imread(image_path)

for (x1, y1, x2, y2) in boxes:
    cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imwrite("windows_detected_qwen.png", img_cv)

print("\nSaved: windows_detected_qwen.png\n")
