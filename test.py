# ===== Test model using an image with the same center crop as mainRunner.py =====

import cv2
from ultralytics import YOLO

# Load model
model = YOLO(r".\runs\classify\train\weights\best.pt")

# Load image
image_path = r"test_4.png"
frame = cv2.imread(image_path)

if frame is None:
    raise FileNotFoundError(f"Cannot load image: {image_path}")

# ===== Center crop: keep the middle 50% of the image width =====
height, width = frame.shape[:2]

crop_ratio = 0.5
crop_width = int(width * crop_ratio)

x1 = (width - crop_width) // 2
x2 = x1 + crop_width

cropped = frame[:, x1:x2]

# Resize to the same input size as mainRunner.py
input_image = cv2.resize(cropped, (384, 384))  

# ===== Predict =====
results = model.predict(source=input_image, verbose=False)

result = results[0]
class_id = result.probs.top1
confidence = float(result.probs.top1conf)
forest_class = model.names[class_id]

print(f"Class ID: {class_id}")
print(f"Forest density: {forest_class}")
print(f"Confidence: {confidence:.4f}")

# ===== Display original image, center crop, and prediction =====
# cv2.imshow("Original Image", frame)
# cv2.imshow("Center Crop (50%)", cropped)
# cv2.imshow(
#     f"Prediction: {forest_class} ({confidence:.2%})",
#     input_image
# )
result.show()  # Display the prediction result with bounding boxes and labels

cv2.waitKey(0)
cv2.destroyAllWindows()