# Computer Vision - Image Processing and Analysis

## 📖 Table of Contents
1. [OpenCV Fundamentals](#opencv-fundamentals)
2. [Image Preprocessing](#image-preprocessing)
3. [Object Detection](#object-detection)
4. [Image Segmentation](#image-segmentation)
5. [Face Recognition](#face-recognition)
6. [Image Classification](#image-classification)
7. [Complete Examples](#complete-examples)

---

## OpenCV Fundamentals

### Installation
```bash
pip install opencv-python opencv-contrib-python
```

### Basic Operations

```python
import cv2
import numpy as np

# Read image
img = cv2.imread('image.jpg')

# Display image
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save image
cv2.imwrite('output.jpg', img)

# Image properties
print(f"Shape: {img.shape}")      # (height, width, channels)
print(f"Size: {img.size}")         # Total pixels
print(f"Data type: {img.dtype}")   # uint8

# Convert color spaces
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```

### Image Manipulation

```python
# Resize
resized = cv2.resize(img, (width, height))
resized = cv2.resize(img, None, fx=0.5, fy=0.5)  # Scale by factor

# Rotate
(h, w) = img.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, 45, 1.0)  # 45 degrees
rotated = cv2.warpAffine(img, M, (w, h))

# Crop
cropped = img[y:y+h, x:x+w]

# Flip
flipped_h = cv2.flip(img, 1)  # Horizontal
flipped_v = cv2.flip(img, 0)  # Vertical
```

---

## Image Preprocessing

### Noise Reduction

```python
# Gaussian Blur
blurred = cv2.GaussianBlur(img, (5, 5), 0)

# Median Blur (removes salt-and-pepper noise)
median = cv2.medianBlur(img, 5)

# Bilateral Filter (preserves edges)
bilateral = cv2.bilateralFilter(img, 9, 75, 75)
```

### Edge Detection

```python
# Canny Edge Detection
edges = cv2.Canny(img, 100, 200)  # minVal, maxVal

# Sobel
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel = np.sqrt(sobelx**2 + sobely**2)

# Laplacian
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
```

### Thresholding

```python
# Binary Threshold
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Adaptive Threshold
adaptive = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 11, 2
)

# Otsu's Threshold
_, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

### Morphological Operations

```python
# Kernel
kernel = np.ones((5, 5), np.uint8)

# Erosion
erosion = cv2.erode(img, kernel, iterations=1)

# Dilation
dilation = cv2.dilate(img, kernel, iterations=1)

# Opening (erosion followed by dilation)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# Closing (dilation followed by erosion)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
```

### Histogram Equalization

```python
# Grayscale
equalized = cv2.equalizeHist(gray)

# Color (CLAHE - Contrast Limited Adaptive Histogram Equalization)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
l = clahe.apply(l)
equalized_color = cv2.merge([l, a, b])
equalized_color = cv2.cvtColor(equalized_color, cv2.COLOR_LAB2BGR)
```

---

## Object Detection

### Haar Cascades

```python
# Face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

# Draw rectangles
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
```

### YOLO (You Only Look Once)

```python
# Load YOLO
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load classes
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Prepare image
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Process detections
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            
            # Draw rectangle
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
```

### Using Pre-trained Models

```python
# Using TensorFlow Hub or PyTorch models
import torch
from torchvision import transforms

# Load pre-trained model (e.g., Faster R-CNN)
model = torch.hub.load('pytorch/vision', 'fasterrcnn_resnet50_fpn', pretrained=True)
model.eval()

# Preprocess
transform = transforms.Compose([transforms.ToTensor()])
img_tensor = transform(img).unsqueeze(0)

# Predict
with torch.no_grad():
    predictions = model(img_tensor)

# Draw bounding boxes
for box, score, label in zip(
    predictions[0]['boxes'],
    predictions[0]['scores'],
    predictions[0]['labels']
):
    if score > 0.5:
        x1, y1, x2, y2 = box.numpy()
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
```

---

## Image Segmentation

### Threshold-based Segmentation

```python
# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
```

### Watershed Algorithm

```python
# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Find sure foreground area
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Find unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Marker labelling
_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

# Apply watershed
markers = cv2.watershed(img, markers)
img[markers == -1] = [0, 0, 255]
```

### Deep Learning Segmentation (U-Net)

```python
from tensorflow.keras import layers

def unet_model(input_size=(256, 256, 3)):
    inputs = layers.Input(input_size)
    
    # Encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    # Bottleneck
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    
    # Decoder
    u1 = layers.UpSampling2D((2, 2))(c3)
    u1 = layers.concatenate([u1, c2])
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u1)
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c4)
    
    u2 = layers.UpSampling2D((2, 2))(c4)
    u2 = layers.concatenate([u2, c1])
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c5)
    
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c5)
    
    model = keras.Model(inputs, outputs)
    return model

model = unet_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

---

## Face Recognition

### Face Detection and Recognition

```python
import face_recognition

# Load images
known_image = face_recognition.load_image_file("known_person.jpg")
unknown_image = face_recognition.load_image_file("unknown_person.jpg")

# Get face encodings
known_encoding = face_recognition.face_encodings(known_image)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

# Compare faces
results = face_recognition.compare_faces([known_encoding], unknown_encoding)
distance = face_recognition.face_distance([known_encoding], unknown_encoding)

if results[0]:
    print("Match found!")
```

### Using OpenCV DNN

```python
# Load face detection model
net = cv2.dnn.readNetFromTensorflow('opencv_face_detector_uint8.pb', 'opencv_face_detector.pbtxt')

# Prepare image
blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123])

# Detect faces
net.setInput(blob)
detections = net.forward()

# Process detections
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.5:
        x1 = int(detections[0, 0, i, 3] * width)
        y1 = int(detections[0, 0, i, 4] * height)
        x2 = int(detections[0, 0, i, 5] * width)
        y2 = int(detections[0, 0, i, 6] * height)
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
```

---

## Image Classification

### Using Pre-trained Models

```python
from tensorflow.keras.applications import ResNet50, imagenet_utils
import numpy as np

# Load pre-trained model
model = ResNet50(weights='imagenet')

# Preprocess image
img = cv2.imread('image.jpg')
img = cv2.resize(img, (224, 224))
img = np.expand_dims(img, axis=0)
img = imagenet_utils.preprocess_input(img)

# Predict
predictions = model.predict(img)
results = imagenet_utils.decode_predictions(predictions)

# Display top 5 predictions
for (imagenetID, label, prob) in results[0]:
    print(f"{label}: {prob * 100:.2f}%")
```

### Custom Image Classifier

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Load data
train_generator = datagen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Build model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)
```

---

## Complete Examples

### Example 1: Image Preprocessing Pipeline

```python
def preprocess_image(img_path):
    # Read image
    img = cv2.imread(img_path)
    
    # Resize
    img = cv2.resize(img, (224, 224))
    
    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize
    img = img.astype('float32') / 255.0
    
    # Apply CLAHE
    lab = cv2.cvtColor((img * 255).astype('uint8'), cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img = cv2.merge([l, a, b])
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    img = img.astype('float32') / 255.0
    
    return img
```

### Example 2: Object Detection Pipeline

```python
def detect_objects(img_path):
    # Load image
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    
    # Load YOLO
    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    # Prepare blob
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    # Process detections
    boxes = []
    confidences = []
    class_ids = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Non-maximum suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    # Draw boxes
    for i in indices:
        i = i[0]
        x, y, w, h = boxes[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return img
```

---

## Key Takeaways

1. **OpenCV**: Essential for image preprocessing
2. **Preprocessing**: Crucial for model performance
3. **Object Detection**: Use pre-trained models (YOLO, Faster R-CNN)
4. **Segmentation**: U-Net for semantic segmentation
5. **Transfer Learning**: Use pre-trained models for classification

---

## Next Steps

- **[09_MLOps_and_Deployment.md](09_MLOps_and_Deployment.md)** - Deploy computer vision models
- **[10_Practical_Projects_Guide.md](10_Practical_Projects_Guide.md)** - Build CV projects

---

**Master computer vision to build image-based AI applications!**

