# RoboMaster EP – Complete Setup & Development Guide

This comprehensive guide covers everything you need to set up Python, install the RoboMaster SDK, test your connection, work with sensors, and develop advanced applications for your RoboMaster EP robot.

---

## 📦 1. Install Python 3.8

Run the following commands in your terminal:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.8 python3.8-venv python3.8-distutils
```

---

## ⚙️ 2. Set Up RoboMaster SDK

Open a terminal in **VS Code** and run:

```bash
# Create a virtual environment
python3.8 -m venv ~/robomaster-env

# Activate the environment
source ~/robomaster-env/bin/activate

# Upgrade tools
pip install --upgrade pip setuptools wheel

# Install SDK
pip install robomaster
```

---

## 🚀 3. Test Connection

Save this script as `test_connection.py` and run it:

```python
from robomaster import robot
import time

def main():
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    # Example: move backward 5 meters
    ep_robot.chassis.move(x=-5, y=0, z=0)
    time.sleep(2)

    # Example (optional): check battery
    # battery = ep_robot.battery
    # val = battery.get_battery().wait_for_completed()
    # print(f"Battery: {val}%")

    ep_robot.close()

if __name__ == '__main__':
    main()
```

---

## 🔌 4. Adding Sensors

### Button Sensor
- Connect to **IO + 3.3V** on the sensor adapter.
- Read value in Python:

```python
ep_robot.sensor_adapter.get_io(adapter_id=3, port=1)
```

### Analog Sensor
- Connect to **3.3V + AD + GND** on the sensor adapter.
- Read value in Python:

```python
ep_robot.sensor_adapter.get_adc(adapter_id=3, port=1)
```

### Function Parameters
- `adapter_id` → Sensor adapter number (e.g., `3`)
- `port` → Specific port on the adapter (e.g., `1`)

---

## 📱 5. Monitor Sensors in the RoboMaster App

1. Connect to the robot using the RoboMaster app.
2. Tap **Settings** (top-right corner).
3. Go to **Extension Module**.
4. Select the **Sensor Adapter** you want to monitor.
5. View the real-time values on the right side of the screen.

---

## 📚 6. Official Examples Repository

**Highly Recommended**: The official RoboMaster SDK examples repository contains extensive code samples for almost every feature of the robot.

### Clone the Examples Repository

```bash
git clone https://github.com/dji-sdk/RoboMaster-SDK.git
cd RoboMaster-SDK
```

### What's Included

The examples folder contains samples for:
- **Basic Movement**: Chassis control, gimbal movement, and blaster operations
- **Computer Vision**: Camera streaming, image processing, and marker detection
- **Sensors**: All sensor types including distance, temperature, and custom sensors
- **Advanced Features**: Multi-robot coordination, custom protocols, and real-time data streaming
- **Game Development**: Battle modes, scoring systems, and interactive gameplay
- **AI Integration**: Machine learning models, autonomous navigation, and decision-making algorithms

### Structure Overview

```
RoboMaster-SDK/
├── examples/
│   ├── 01_robot/           # Basic robot control
│   ├── 02_chassis/         # Movement and navigation
│   ├── 03_gimbal/          # Camera gimbal control
│   ├── 04_blaster/         # Shooting mechanisms
│   ├── 05_armor/           # Hit detection
│   ├── 06_vision/          # Computer vision
│   ├── 07_sensor/          # Sensor integration
│   ├── 08_stream/          # Video streaming
│   └── 09_advanced/        # Complex applications
└── docs/                   # Detailed documentation
```

### Running Examples

```bash
# Navigate to specific example
cd examples/01_robot

# Run with your virtual environment activated
source ~/robomaster-env/bin/activate
python basic_ctrl.py
```

---

## 🛠️ 7. Essential Python Modules for RoboMaster Development

### Core Modules to Install

```bash
# Activate your virtual environment first
source ~/robomaster-env/bin/activate

# Computer vision and image processing
pip install opencv-python opencv-contrib-python

# Threading and concurrent programming
pip install threading-timeout

# Numerical computing (often needed for sensors)
pip install numpy

# Data visualization (useful for sensor data)
pip install matplotlib

# Optional: Advanced image processing
pip install pillow scikit-image
```

### OpenCV for Computer Vision

OpenCV is essential for advanced RoboMaster applications:

```python
import cv2
from robomaster import robot, camera

def vision_example():
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    
    ep_camera = ep_robot.camera
    ep_camera.start_video_stream(display=False)
    
    while True:
        frame = ep_camera.read_cv2_image(strategy="newest")
        
        # Process frame with OpenCV
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Display results
        cv2.imshow('Original', frame)
        cv2.imshow('Edges', edges)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    ep_camera.stop_video_stream()
    ep_robot.close()
    cv2.destroyAllWindows()
```

### Threading for Concurrent Operations

Threading allows you to handle multiple tasks simultaneously:

```python
import threading
import time
from robomaster import robot

def sensor_monitoring(ep_robot):
    """Run sensor monitoring in background"""
    while True:
        try:
            # Read sensors continuously
            io_val = ep_robot.sensor_adapter.get_io(adapter_id=3, port=1)
            adc_val = ep_robot.sensor_adapter.get_adc(adapter_id=3, port=1)
            print(f"IO: {io_val}, ADC: {adc_val}")
            time.sleep(0.1)
        except:
            break

def main_control():
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    
    # Start sensor monitoring thread
    sensor_thread = threading.Thread(target=sensor_monitoring, args=(ep_robot,))
    sensor_thread.daemon = True
    sensor_thread.start()
    
    # Main control loop
    for i in range(10):
        ep_robot.chassis.move(x=0.5, y=0, z=0)
        time.sleep(1)
        ep_robot.chassis.move(x=-0.5, y=0, z=0)
        time.sleep(1)
    
    ep_robot.close()
```

---

## 🎯 8. Vision Markers and ArUco Detection

### RoboMaster Official Markers

RoboMaster provides official printable markers for detection and navigation. These can be downloaded from the DJI developer resources and printed for use in competitions or development.

**Benefits of Official Markers:**
- Optimized for RoboMaster camera systems
- Pre-configured detection parameters
- Standardized for competitions
- Built-in support in the SDK

### Using RoboMaster Markers

```python
from robomaster import robot, vision

def detect_robomaster_markers():
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    
    ep_vision = ep_robot.vision
    ep_camera = ep_robot.camera
    
    # Enable marker detection
    ep_vision.sub_detect_info(name="marker", callback=marker_callback)
    ep_camera.start_video_stream(display=False)
    
    time.sleep(10)  # Detection for 10 seconds
    
    ep_vision.unsub_detect_info(name="marker")
    ep_camera.stop_video_stream()
    ep_robot.close()

def marker_callback(sub_info):
    x, y = sub_info
    print(f"Marker detected at: x={x}, y={y}")
```

### Alternative: OpenCV ArUco Markers

**ArUco markers from OpenCV provide similar functionality and are often more flexible:**

#### Advantages of ArUco:
- **Customizable**: Generate any number of unique markers
- **Robust Detection**: Excellent performance in various lighting conditions  
- **Flexible Sizing**: Can be printed at any size
- **Rich Information**: Can encode additional data
- **Pose Estimation**: Calculate 3D position and orientation

#### Generate ArUco Markers

```python
import cv2
import numpy as np

def generate_aruco_markers():
    # Create ArUco dictionary
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    
    # Generate markers (IDs 0-9)
    for marker_id in range(10):
        marker_img = cv2.aruco.drawMarker(aruco_dict, marker_id, 200)
        cv2.imwrite(f'marker_{marker_id}.png', marker_img)
        print(f"Generated marker_{marker_id}.png")
```

#### Detect ArUco Markers with RoboMaster

```python
import cv2
from robomaster import robot, camera
import numpy as np

def detect_aruco_markers():
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    
    ep_camera = ep_robot.camera
    ep_camera.start_video_stream(display=False)
    
    # ArUco setup
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    aruco_params = cv2.aruco.DetectorParameters_create()
    
    while True:
        frame = ep_camera.read_cv2_image(strategy="newest")
        
        # Detect ArUco markers
        corners, ids, _ = cv2.aruco.detectMarkers(
            frame, aruco_dict, parameters=aruco_params
        )
        
        if ids is not None:
            # Draw detected markers
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            for i, marker_id in enumerate(ids.flatten()):
                # Calculate marker center
                corner = corners[i][0]
                center_x = int(corner[:, 0].mean())
                center_y = int(corner[:, 1].mean())
                
                print(f"Marker {marker_id} at ({center_x}, {center_y})")
                
                # Optional: Control robot based on marker position
                # navigate_to_marker(ep_robot, center_x, center_y)
        
        cv2.imshow('ArUco Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    ep_camera.stop_video_stream()
    ep_robot.close()
    cv2.destroyAllWindows()

def navigate_to_marker(ep_robot, marker_x, marker_y):
    """Simple navigation toward detected marker"""
    frame_center_x = 640 // 2  # Assuming 640px width
    
    # Calculate error from center
    error_x = marker_x - frame_center_x
    
    # Simple proportional control
    if abs(error_x) > 50:  # Dead zone
        rotation_speed = error_x * 0.01  # Adjust gain as needed
        ep_robot.chassis.drive_speed(x=0, y=0, z=rotation_speed)
    else:
        # Move forward when centered
        ep_robot.chassis.drive_speed(x=0.3, y=0, z=0)
```

### Comparison: RoboMaster vs ArUco Markers

| Feature | RoboMaster Markers | OpenCV ArUco |
|---------|-------------------|--------------|
| **Setup** | Built-in SDK support | Requires OpenCV setup |
| **Customization** | Limited to official designs | Unlimited custom markers |
| **Detection Speed** | Optimized for RoboMaster | Very fast, optimized |
| **Accuracy** | High | Very high |
| **Pose Estimation** | Basic | Advanced 3D pose |
| **Competition Use** | Required for official events | Personal projects |
| **Learning Curve** | Minimal | Moderate |

**Recommendation**: Use RoboMaster markers for competitions and official events, but consider ArUco for development, learning, and custom applications where you need more flexibility.

---

## 🚀 9. Advanced Development Tips

### Project Structure
```
my_robomaster_project/
├── main.py                 # Main application entry
├── config/
│   ├── robot_config.py     # Robot settings
│   └── camera_config.py    # Camera parameters
├── modules/
│   ├── vision.py           # Computer vision functions
│   ├── sensors.py          # Sensor handling
│   ├── navigation.py       # Movement algorithms
│   └── utils.py           # Helper functions
├── data/
│   ├── markers/           # ArUco marker images
│   └── calibration/       # Camera calibration data
└── tests/
    └── test_connection.py  # Unit tests
```

### Best Practices

1. **Always use try-except blocks** for robot connections
2. **Implement proper cleanup** with `ep_robot.close()`
3. **Use threading** for concurrent sensor monitoring
4. **Calibrate your camera** for accurate vision applications
5. **Test with different lighting conditions**
6. **Keep backup power** - low battery can cause unexpected disconnections

### Debugging Tips

```python
import logging
from robomaster import robot

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG)

def debug_robot_connection():
    try:
        ep_robot = robot.Robot()
        ep_robot.initialize(conn_type="ap")
        
        # Test basic functions
        version = ep_robot.get_version()
        print(f"Robot version: {version}")
        
        battery = ep_robot.battery.get_battery().wait_for_completed()
        print(f"Battery level: {battery}%")
        
        ep_robot.close()
        print("Connection test successful!")
        
    except Exception as e:
        print(f"Connection failed: {e}")
```

---

## 🏆 10. Next Steps

### Beginner Projects
1. **Remote Control**: Create a custom controller using keyboard input
2. **Obstacle Avoidance**: Use distance sensors to navigate around objects
3. **Line Following**: Implement basic line-following using camera vision
4. **Marker Navigation**: Create a waypoint system using ArUco markers

### Intermediate Projects
1. **Autonomous Mapping**: Build a SLAM (Simultaneous Localization and Mapping) system
2. **Object Tracking**: Follow colored objects or people using computer vision
3. **Voice Control**: Integrate speech recognition for voice commands
4. **Multi-Robot Coordination**: Control multiple robots working together

### Advanced Projects
1. **AI Integration**: Implement machine learning models for decision making
2. **Competition Bot**: Build a fully autonomous competition robot
3. **Gesture Recognition**: Control robot with hand gestures
4. **Augmented Reality**: Overlay digital information on robot's camera feed

### Useful Resources

- **Official Documentation**: [RoboMaster Developer Guide](https://robomaster-dev.readthedocs.io/)
- **Community Forum**: [RoboMaster Community](https://www.dji.com/robomaster-s1/community)
- **GitHub Issues**: [SDK Issues & Support](https://github.com/dji-sdk/RoboMaster-SDK/issues)
- **OpenCV Documentation**: [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)

---

✅ **You are now ready to develop advanced applications with your RoboMaster EP!**

Happy coding and building! 🤖
