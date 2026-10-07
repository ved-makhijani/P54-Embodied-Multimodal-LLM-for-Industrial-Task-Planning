# Vision Framework Comparison: OpenCV vs. YOLO Variants

**Author:** Kaveesha Dharmadasa (KD)  
**Task ID:** PITPP-7  
**Project:** Multimodal LLM for Industrial Task Planning  

## 1. Overview
This document provides a technical comparison between OpenCV and YOLO (You Only Look Once) variants to determine the most suitable visual perception framework for our simulated industrial environment. The goal is to accurately detect objects, map their coordinates, and feed this data to the LLM Task Planner.

## 2. OpenCV (Open Source Computer Vision Library)
OpenCV is a foundational library for traditional computer vision and image processing.

* **Strengths:** * Extremely lightweight and fast for basic operations.
  * Excellent for image preprocessing (blurring, edge detection, color space conversion like HSV filtering).
  * Highly effective for camera calibration and raw pixel manipulation.
* **Weaknesses:** * Struggles with complex object recognition in dynamic lighting or unstructured environments.
  * Requires extensive manual feature engineering (e.g., manually defining color ranges for a "blue block").
  * Not inherently designed for modern deep-learning-based classification without external models.

## 3. YOLO Variants (e.g., YOLOv8)
YOLO represents the state-of-the-art in real-time, deep-learning-based object detection.

* **Strengths:**
  * Exceptional accuracy in detecting and classifying multiple objects simultaneously.
  * Outputs highly precise 2D bounding boxes and class labels out-of-the-box.
  * Robust against lighting changes, partial occlusions, and varied object orientations.
* **Weaknesses:**
  * Computationally heavier than OpenCV (benefits significantly from GPU acceleration).
  * Requires a trained dataset (though pre-trained models handle basic industrial objects well).

## 4. Comparative Summary & Conclusion

| Feature | OpenCV (Traditional) | YOLOv8 (Deep Learning) |
| :--- | :--- | :--- |
| **Object Classification** | Low (Requires manual tuning) | High (Automated, robust) |
| **Bounding Box Accuracy** | Moderate (Contour-based) | Very High (Neural Network) |
| **Computational Load** | Very Low (CPU friendly) | High (GPU recommended) |
| **Setup Complexity** | High (Code-heavy logic) | Low (Plug-and-play inference) |

**Conclusion:** Relying solely on OpenCV is insufficient for the dynamic nature of intent-based robotic picking. However, YOLO alone lacks the low-level camera feed manipulation required for simulation APIs. Therefore, a hybrid architecture is required: OpenCV should handle the camera feed preprocessing and geometric coordinate mapping, while a YOLO variant (like YOLOv8) should be utilized for the core object detection and classification pipeline.