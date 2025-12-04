import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import cv2
from qwen_vl_utils import process_vision_info

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
print(f"Image size: {image.size}")
print(f"Image mode: {image.mode}")

# Qwen2-VL requires a specific conversation format
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image,
            },
            {
                "type": "text", 
                "text": "Describe what you see in this image. What is the main object? What are its dimensions and position?"
            },
        ],
    }
]

# Apply chat template
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cpu")

output = model.generate(**inputs, max_new_tokens=500)
response = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print("\nRAW MODEL OUTPUT:\n", response)
print("\n" + "="*80 + "\n")

# Extract assistant's response
if "assistant" in response:
    assistant_response = response.split("assistant")[-1].strip()
else:
    assistant_response = response

print("ASSISTANT RESPONSE:", assistant_response)
print("\nThe model is describing the image. Run the script again to proceed with detection.")
exit()

# -------------------------------
# Parse bounding boxes (disabled for now - testing vision first)
# -------------------------------
try:
    boxes = eval(assistant_response)
except Exception as e:
    print(f"Could not parse bounding boxes. Error: {e}")
    print("Response was:")
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
