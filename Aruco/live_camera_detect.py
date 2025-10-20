#!/usr/bin/env python3
"""
ArUco Distance Measurement Script for RoboMaster Camera
Uses solvePnP for accurate 3D pose estimation
Compatible with camera_calibration2.py calibration data

REQUIREMENTS:
- OpenCV 4.x with ArUco support
- RoboMaster SDK
- Camera calibration file (camera_calibration.pkl from camera_calibration2.py)
- Printed ArUco markers

USAGE:
1. First calibrate your camera using camera_calibration2.py
2. Print ArUco markers (DICT_4X4_50 recommended)
3. Measure your printed marker size in meters
4. Run: python aruco_distance_robomaster.py

MARKER SETUP:
- Default dictionary: DICT_4X4_50 (IDs 0-49)
- Measure marker size from outer black border to outer black border
- Common sizes: 50mm (0.05m), 100mm (0.10m), 150mm (0.15m)

CONTROLS:
- 'q': Quit
- 's': Save current detection data
- 'c': Toggle coordinate system display
- 'r': Reset saved data
"""

import cv2
import numpy as np
import pickle
import json
import math
import time
from datetime import datetime
import os
import robomaster
from robomaster import robot


class ArUcoDistanceMeasurer:
    def __init__(self, calibration_file='camera_calibration_20251005_150222.pkl', marker_size=0.05):
        """
        Initialize ArUco distance measurer
        
        Args:
            calibration_file: Path to camera calibration file (.pkl format)
            marker_size: Size of ArUco markers in meters (outer border to outer border)
        """
        self.calibration_file = calibration_file
        self.marker_size = marker_size
        
        # Load camera calibration
        self.camera_matrix, self.dist_coeffs = self.load_calibration()
        
        # ArUco detection setup (OpenCV 4.7+ API)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        # Optimize detection parameters
        self.aruco_params.adaptiveThreshWinSizeMin = 3
        self.aruco_params.adaptiveThreshWinSizeMax = 23
        self.aruco_params.adaptiveThreshWinSizeStep = 10
        self.aruco_params.minMarkerPerimeterRate = 0.03
        self.aruco_params.maxMarkerPerimeterRate = 4.0
        self.aruco_params.polygonalApproxAccuracyRate = 0.05
        self.aruco_params.minCornerDistanceRate = 0.05
        self.aruco_params.minDistanceToBorder = 3
        
        # Create ArUco detector
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # 3D object points for ArUco marker (square centered at origin)
        half_size = marker_size / 2
        self.object_points = np.array([
            [-half_size, -half_size, 0],  # Bottom-left
            [ half_size, -half_size, 0],  # Bottom-right  
            [ half_size,  half_size, 0],  # Top-right
            [-half_size,  half_size, 0]   # Top-left
        ], dtype=np.float32)
        
        # Detection data storage
        self.detections = []
        self.show_coordinate_system = False
        
        print(f"ArUco Distance Measurer initialized")
        print(f"Marker size: {marker_size*1000:.1f}mm ({marker_size:.3f}m)")
        print(f"Dictionary: DICT_4X4_50 (IDs 0-49)")
        print(f"Calibration: {calibration_file}")
    
    def load_calibration(self):
        """Load camera calibration data from pickle file"""
        try:
            with open(self.calibration_file, 'rb') as f:
                calib_data = pickle.load(f)
            
            camera_matrix = calib_data['camera_matrix']
            dist_coeffs = calib_data['dist_coeffs']
            
            print("✓ Camera calibration loaded successfully")
            print(f"Camera matrix shape: {camera_matrix.shape}")
            print(f"Distortion coefficients shape: {dist_coeffs.shape}")
            
            return camera_matrix, dist_coeffs
            
        except FileNotFoundError:
            print(f"❌ Error: Calibration file '{self.calibration_file}' not found!")
            print("Please run camera_calibration2.py first to generate calibration data.")
            raise
        except Exception as e:
            print(f"❌ Error loading calibration: {e}")
            raise
    
    def detect_and_measure(self, frame):
        """
        Detect ArUco markers and measure their distance/pose using solvePnP
        
        Args:
            frame: Input image frame
            
        Returns:
            Tuple of (annotated_frame, detection_data)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect ArUco markers (OpenCV 4.7+ API)
        corners, ids, rejected = self.aruco_detector.detectMarkers(gray)
        
        detection_data = []
        annotated_frame = frame.copy()
        
        if ids is not None and len(corners) > 0:
            # Draw detected markers
            cv2.aruco.drawDetectedMarkers(annotated_frame, corners, ids)
            
            for i, marker_corners in enumerate(corners):
                marker_id = ids[i][0]
                
                # Reshape corners for solvePnP (needs (N, 1, 2) format)
                image_points = marker_corners[0].astype(np.float32)
                
                # Use solvePnP for pose estimation
                success, rvec, tvec = cv2.solvePnP(
                    self.object_points,
                    image_points,
                    self.camera_matrix,
                    self.dist_coeffs
                )
                
                if success:
                    # Calculate distance
                    distance = np.linalg.norm(tvec)
                    
                    # Convert rotation vector to rotation matrix
                    rotation_matrix, _ = cv2.Rodrigues(rvec)
                    
                    # Calculate Euler angles
                    roll, pitch, yaw = self.rotation_matrix_to_euler(rotation_matrix)
                    
                    # Store detection data
                    detection_info = {
                        'id': int(marker_id),
                        'distance': float(distance),
                        'position': {
                            'x': float(tvec[0][0]),
                            'y': float(tvec[1][0]), 
                            'z': float(tvec[2][0])
                        },
                        'rotation': {
                            'roll': float(math.degrees(roll)),
                            'pitch': float(math.degrees(pitch)),
                            'yaw': float(math.degrees(yaw))
                        },
                        'timestamp': time.time(),
                        'marker_size': self.marker_size
                    }
                    detection_data.append(detection_info)
                    
                    # Draw 3D coordinate axes
                    if self.show_coordinate_system:
                        axis_length = self.marker_size * 0.5
                        axis_points = np.float32([
                            [0, 0, 0],           # Origin
                            [axis_length, 0, 0], # X-axis (red)
                            [0, axis_length, 0], # Y-axis (green)
                            [0, 0, -axis_length] # Z-axis (blue)
                        ]).reshape(-1, 3)
                        
                        # Project 3D points to image plane
                        projected_points, _ = cv2.projectPoints(
                            axis_points, rvec, tvec, self.camera_matrix, self.dist_coeffs
                        )
                        projected_points = projected_points.reshape(-1, 2).astype(int)
                        
                        # Draw coordinate axes
                        # Draw coordinate axes - ensure proper integer conversion
                        origin = tuple(map(int, projected_points[0].flatten()))
                        x_point = tuple(map(int, projected_points[1].flatten()))
                        y_point = tuple(map(int, projected_points[2].flatten()))
                        z_point = tuple(map(int, projected_points[3].flatten()))
                        
                        # X-axis (red)
                        cv2.arrowedLine(annotated_frame, origin, x_point, (0, 0, 255), 3, tipLength=0.3)
                        # Y-axis (green)
                        cv2.arrowedLine(annotated_frame, origin, y_point, (0, 255, 0), 3, tipLength=0.3)
                        # Z-axis (blue)  
                        cv2.arrowedLine(annotated_frame, origin, z_point, (255, 0, 0), 3, tipLength=0.3)
                    
                    # Draw marker information
                    self.draw_marker_info(annotated_frame, marker_corners[0], detection_info)
        
        # Draw rejected markers for debugging
        if len(rejected) > 0:
            cv2.aruco.drawDetectedMarkers(annotated_frame, rejected, borderColor=(0, 0, 255))
        
        return annotated_frame, detection_data
    
    def rotation_matrix_to_euler(self, R):
        """Convert rotation matrix to Euler angles (roll, pitch, yaw)"""
        sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
        singular = sy < 1e-6
        
        if not singular:
            roll = math.atan2(R[2,1], R[2,2])
            pitch = math.atan2(-R[2,0], sy)
            yaw = math.atan2(R[1,0], R[0,0])
        else:
            roll = math.atan2(-R[1,2], R[1,1])
            pitch = math.atan2(-R[2,0], sy)
            yaw = 0
        
        return roll, pitch, yaw
    
    def draw_marker_info(self, frame, corners, detection_info):
        """Draw marker information on the frame"""
        # Calculate marker center
        center = np.mean(corners, axis=0).astype(int)
        
        # Prepare text
        marker_id = detection_info['id']
        distance = detection_info['distance']
        position = detection_info['position']
        
        # Text lines
        lines = [
            f"ID: {marker_id}",
            f"Dist: {distance:.3f}m",
            f"X: {position['x']:.3f}m",
            f"Y: {position['y']:.3f}m", 
            f"Z: {position['z']:.3f}m"
        ]
        
        # Draw text with background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        line_height = 20
        
        # Calculate text background size
        text_widths = []
        for line in lines:
            (w, h), _ = cv2.getTextSize(line, font, font_scale, thickness)
            text_widths.append(w)
        
        max_width = max(text_widths)
        bg_height = len(lines) * line_height + 10
        
        # Draw background rectangle
        bg_top_left = (center[0] - max_width//2 - 5, center[1] - bg_height//2)
        bg_bottom_right = (center[0] + max_width//2 + 5, center[1] + bg_height//2)
        cv2.rectangle(frame, bg_top_left, bg_bottom_right, (0, 0, 0), -1)
        cv2.rectangle(frame, bg_top_left, bg_bottom_right, (255, 255, 255), 1)
        
        # Draw text lines
        for i, line in enumerate(lines):
            text_x = center[0] - text_widths[i]//2
            text_y = center[1] - bg_height//2 + 20 + i * line_height
            cv2.putText(frame, line, (text_x, text_y), font, font_scale, (0, 255, 255), thickness)
    
    def save_detection_data(self, detections):
        """Save detection data to JSON file"""
        if not detections:
            print("No detection data to save")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"aruco_detections_{timestamp}.json"
        
        # Prepare data for JSON serialization
        save_data = {
            'timestamp': timestamp,
            'marker_size': self.marker_size,
            'calibration_file': self.calibration_file,
            'detections': detections
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(save_data, f, indent=2)
            print(f"✓ Detection data saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving data: {e}")
    
    def print_detection_summary(self, detections):
        """Print summary of current detections"""
        if not detections:
            return
            
        print("\n" + "="*60)
        print("CURRENT DETECTIONS")
        print("="*60)
        
        for detection in detections:
            print(f"ArUco ID: {detection['id']}")
            print(f"  Distance: {detection['distance']:.3f} m ({detection['distance']*100:.1f} cm)")
            print(f"  Position: X={detection['position']['x']:.3f}m, "
                  f"Y={detection['position']['y']:.3f}m, Z={detection['position']['z']:.3f}m")
            print(f"  Rotation: Roll={detection['rotation']['roll']:.1f}°, "
                  f"Pitch={detection['rotation']['pitch']:.1f}°, Yaw={detection['rotation']['yaw']:.1f}°")
            print("-" * 40)


def main():
    print("ArUco Distance Measurement for RoboMaster Camera")
    print("="*50)
    
    # Configuration
    CALIBRATION_FILE = input("Input your calibration file path: ")
    MARKER_SIZE = float(input("Input your marker size in meters (e.g., 0.15 for 15cm): "))
    
    # Check if calibration file exists
    if not os.path.exists(CALIBRATION_FILE):
        print(f"❌ Calibration file '{CALIBRATION_FILE}' not found!")
        print("Please run calibrate.py first to generate calibration data.")
        print("Example: python calibrate.py")
        return
    
    try:
        # Initialize ArUco distance measurer
        measurer = ArUcoDistanceMeasurer(CALIBRATION_FILE, MARKER_SIZE)
        
        # Initialize RoboMaster robot
        print("Connecting to RoboMaster...")
        ep_robot = robot.Robot()
        ep_robot.initialize(conn_type="ap")
        ep_camera = ep_robot.camera
        
        # Start video stream
        ep_camera.start_video_stream(False)
        print("✓ RoboMaster camera started")
        
        print("\nControls:")
        print("  'q' - Quit")
        print("  's' - Save current detection data")
        print("  'c' - Toggle coordinate system display")
        print("  'r' - Reset detection history")
        print("\nStarting ArUco detection...")
        
        frame_count = 0
        all_detections = []
        
        while True:
            # Get frame from RoboMaster camera
            frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
            
            if frame is None:
                print("Warning: Failed to read frame from camera")
                continue
            
            frame_count += 1
            
            # Detect and measure ArUco markers
            annotated_frame, detections = measurer.detect_and_measure(frame)
            
            # Store detections
            all_detections.extend(detections)
            
            # Print detection info to console
            if detections:
                measurer.print_detection_summary(detections)
            
            # Add UI elements to frame
            cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Markers detected: {len(detections)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Marker size: {MARKER_SIZE*1000:.0f}mm", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Controls help
            cv2.putText(annotated_frame, "Controls: 'q'=quit, 's'=save, 'c'=axes, 'r'=reset", 
                       (10, annotated_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow("ArUco Distance Measurement - RoboMaster", annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('s'):
                print("Saving current detection data...")
                measurer.save_detection_data(all_detections)
            elif key == ord('c'):
                measurer.show_coordinate_system = not measurer.show_coordinate_system
                status = "ON" if measurer.show_coordinate_system else "OFF"
                print(f"Coordinate system display: {status}")
            elif key == ord('r'):
                all_detections.clear()
                print("Detection history reset")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("Cleaning up...")
        try:
            cv2.destroyAllWindows()
            ep_camera.stop_video_stream()
            ep_robot.close()
        except:
            pass
        print("✓ Cleanup completed")


if __name__ == "__main__":
    main()
