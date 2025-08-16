# 🔍 010-sliding-window-detector
**Basic Object Detection using Sliding Windows**


## 📋 Project Overview
This project implements a basic object detection system using sliding windows and template matching in Python. Features include HOG-based feature extraction, cosine similarity classification, and non-maximum suppression for accurate bounding box detection. Built with OpenCV and NumPy, it provides a foundation for understanding computer vision object detection techniques.

## 🎯 Objectives
- Implement sliding window object detection from scratch
- Extract meaningful features using Histogram of Oriented Gradients (HOG)
- Perform template matching with cosine similarity classification
- Apply non-maximum suppression to eliminate duplicate detections
- Visualize detection results with bounding boxes and confidence scores
- Provide educational examples for computer vision learning

## 🔧 Technical Implementation

### 📌 Core Components
- **Sliding Windows**: Multi-scale window generation across input images
- **HOG Features**: Histogram of Oriented Gradients for shape description
- **Template Matching**: Cosine similarity-based object classification
- **Non-Maximum Suppression**: IoU-based duplicate detection removal

### 🧹 Feature Extraction
**HOG Features**:
- Gradient magnitude and orientation calculation
- Orientation binning (9 bins, 0-180 degrees)
- Histogram normalization for robustness

**Additional Features**:
- Mean intensity statistics
- Standard deviation measures
- Edge density using Canny edge detection

### ⚙️ Detection Pipeline
1. **Window Generation**: Create multi-scale sliding windows
2. **Feature Extraction**: Extract HOG + statistical features per window
3. **Classification**: Compare features using cosine similarity
4. **Post-processing**: Apply non-maximum suppression
5. **Visualization**: Draw bounding boxes with confidence scores

### 📏 Evaluation Metrics
- **Detection Accuracy**: Template matching similarity scores
- **IoU Calculation**: Intersection over Union for overlap measurement
- **Non-Maximum Suppression**: Precision improvement through duplicate removal

## 📊 Key Functions

| Function | Description |
|----------|-------------|
| `generate_sliding_windows()` | Creates multi-scale sliding windows |
| `extract_hog_features()` | Extracts HOG features from image regions |
| `extract_window_features()` | Combines HOG, intensity, and edge features |
| `simple_classifier()` | Template matching with cosine similarity |
| `non_maximum_suppression()` | Removes overlapping detections |
| `detect_objects()` | Main detection pipeline |
| `draw_detections()` | Visualizes results with bounding boxes |

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- OpenCV 4.0+
- NumPy
- Matplotlib (for visualization)

### Installation
1. Clone the repository:
```bash
git clone https://github.com/zubair-csc/010-sliding-window-detector.git
cd 010-sliding-window-detector
```

2. Install required libraries:
```bash
pip install opencv-python numpy matplotlib
```

### Usage Example
```python
import cv2
from sliding_window_detector import detect_objects, draw_detections

# Load images
image = cv2.imread('test_image.jpg')
template = cv2.imread('template.jpg')

# Detect objects
detections = detect_objects(
    image, 
    template,
    similarity_threshold=0.6,
    nms_threshold=0.3
)

# Visualize results
result = draw_detections(image, detections)
cv2.imshow('Detections', result)
cv2.waitKey(0)
```

### Running the Demo
```python
from sliding_window_detector import demo_object_detection

# Run demonstration with synthetic data
original, template, result, detections = demo_object_detection()
print(f"Found {len(detections)} objects")
```

## 📈 Results
- **Multi-scale Detection**: Handles objects of different sizes
- **Feature Robustness**: HOG features provide shape-based matching
- **Duplicate Removal**: Non-maximum suppression improves precision
- **Visual Feedback**: Clear bounding box visualization with confidence scores

## 🛠️ Customization Options
- **Window Sizes**: Adjust `window_sizes` parameter for different object scales
- **Step Size**: Control detection density with `step_size` parameter
- **Similarity Threshold**: Fine-tune detection sensitivity
- **NMS Threshold**: Adjust overlap tolerance for duplicate removal

## 📚 Educational Value
This implementation serves as an excellent learning resource for:
- Understanding sliding window detection principles
- Learning feature extraction techniques (HOG)
- Implementing template matching algorithms
- Applying non-maximum suppression
- Computer vision pipeline development

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 🙌 Acknowledgments
- OpenCV community for computer vision tools
- scikit-image for HOG implementation inspiration
- Computer vision research community for sliding window techniques
