 Architectural Recommendations: Tool choice in Perception and language modules.

Author: Kaveesha Dharmadasa (KD)  
Task ID: PITPP-9  
Project: Multimodal LLM in Planning Industrial Tasks.  
Date: 13 April 2026  

Overview
This document presents the ultimate choice of the tool to be used in the Visual Perception and Natural Language Processing modules, as a result of the comparative analysis performed during the current sprint.

2. Vision Framework Recommendation: YOLOv8 + OpenCV.
Having evaluated a set of computer vision models that can be applied to our simulated industrial setting, a hybrid model based on YOLOv8 and OpenCV is recommended.

* YOLOv8 (Primary Object Detection): YOLOv8 is chosen as the main framework because it is faster and more accurate in real-time object detection. It can easily find particular parts of manufacturing and produce accurate 2D bounding boxes, which is essential in the manipulation functions of the robot.
* OpenCV (Secondary and Preprocessing): Although OpenCV is not the solution to a strong object recognition in the dynamic environment, it will be kept to support the preprocessing of images, the coordinate system, and the management of the camera feed before feeding the frames to the YOLO model.

2. OpenAI API with Local Fallback LLM Pipeline Recommendation.
The essence of this project is based on natural language translation to executable robotic intents. The proposed pipeline consists of a cloud-first solution with a rigid local fallback structure.

* OpenAI API / GPT-4 (Primary Reasoning Engine): To test the first prototype, it is suggested to use OpenAI API (GPT-4). Our study shows that it has the best zero-shot intent extraction as of now. It is much more effective at complex reasoning (with ReAct and Chain-of-Thought prompting) than smaller local models, and has a much better ability to interpret objects, actions, and spatial relations out of ambiguous operator instructions.
Local Open-Source Model / LLaMA-3 (Privacy Fallback): To fully adhere to the Australian Privacy Principles (APPs) as described in our ethical considerations, a local open-source model (like LLaMA-3) will have to be incorporated as a backup. This makes sure that in subsequent deployments in the real world, any prompts that have sensitive or proprietary factory instructions or layouts are not required to be sent to external cloud servers.

Conclusion
The chosen tools offer the best tradeoff between the highest level of AI performance (GPT-4, YOLOv8) and ethical/functional adherence (Local LLMs, OpenCV). The results of these devices will be normalized in the newly developed Scene Abstraction JSON format to make them easily integrated with the Central Task Planner.