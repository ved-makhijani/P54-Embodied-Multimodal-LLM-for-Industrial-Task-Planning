import pybullet
import time
import pybullet_data
import math
import pybullet as p
import numpy as np
import cv2

physicsClient = pybullet.connect(pybullet.GUI)

# Set the search path to find the URDF files
pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load the plane
planeId = pybullet.loadURDF("plane.urdf")

# Load the robotic arm (Kuka IIWA)
robotId = pybullet.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0])

# Set gravity
pybullet.setGravity(0, 0, -9.81)

# Get the number of joints
numJoints = pybullet.getNumJoints(robotId)
print(f"Number of joints: {numJoints}")

# Simulation loop
i = 0

# Create a red cube
cube1 = p.loadURDF("cube_small.urdf", [0.8, 0, 0.1], globalScaling=2.0)

# Create a blue cube
cube2 = p.loadURDF("cube_small.urdf", [1.3, 0.2, 0.1], globalScaling=2.0)

# Red cube
p.changeVisualShape(cube1, -1, rgbaColor=[1, 0, 0, 1])

# Blue cube
p.changeVisualShape(cube2, -1, rgbaColor=[0, 0, 1, 1])

while True:
    for joint in range(numJoints):
        jointPos = 0.5 * (1 + math.sin(i * 0.01 + joint * 0.5))
        pybullet.setJointMotorControl2(
            robotId,
            joint,
            pybullet.POSITION_CONTROL,
            targetPosition=jointPos
        )
    # Camera views setting
    viewMatrix = p.computeViewMatrix(
        cameraEyePosition=[1, 2, 2],
        cameraTargetPosition=[1, 0, 0],
        cameraUpVector=[0, 0, 1]
    )
    
    projectionMatrix = p.computeProjectionMatrixFOV(
        fov=60,
        aspect=640/480,
        nearVal=0.1,
        farVal=100
    )
    
    width, height, img, depth, seg = p.getCameraImage(
        width=640,
        height=480,
        viewMatrix=viewMatrix,
        projectionMatrix=projectionMatrix
    )
    # camera view end

    
    # Convert to OpenCV format
    img = np.reshape(img, (height, width, 4))
    img = img.astype(np.uint8)
    img = img[:, :, :3]  # remove alpha channel
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Gray", gray)

    cv2.imshow("Simulation View", img)
    cv2.waitKey(1)

    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Red color range
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    
    # Find contours for red blocks
    contours, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, "Red Block", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    
    # Blue color range
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])
    
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Find contours for blue blocks
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours_blue:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img, "Blue Block", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

    scene = []
    
    # Store detected objects and their positions in the scene list
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        scene.append({
            "object": "red_block",
            "position": [x, y]
        })
    for cnt in contours_blue:
        x, y, w, h = cv2.boundingRect(cnt)
        scene.append({
            "object": "blue_block",
            "position": [x, y]
        })

    keys = pybullet.getKeyboardEvents()
    if ord('q') in keys and keys[ord('q')] & pybullet.KEY_IS_DOWN:
        print("Exiting simulation...")
        break
    pybullet.stepSimulation()
    time.sleep(1./240.)
    i += 1


print(scene)

# Disconnect
pybullet.disconnect()


