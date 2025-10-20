#!/usr/bin/env python3
"""
Unified Camera Calibration System
Supports: Regular webcams, RoboMaster cameras
Features: Folder processing, auto-capture
Compatible with Python 3.8 and OpenCV 4.12.0
"""

import cv2
import numpy as np
import pickle
import os
import argparse
import glob
import time
from datetime import datetime

# Try to import RoboMaster SDK (optional)
try:
    from robomaster import robot
    ROBOMASTER_AVAILABLE = True
except ImportError:
    ROBOMASTER_AVAILABLE = False
    print("Note: RoboMaster SDK not available. Using standard webcam only.")


class UnifiedCalibrator:
    def __init__(self, chessboard_size=(9, 6), save_dir="calibration_images"):
        """Initialize the unified calibrator"""
        self.chessboard_size = chessboard_size
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        print(f"Chessboard size: {chessboard_size[0]}x{chessboard_size[1]} internal corners")
        print(f"Save directory: {save_dir}")
        
        # OpenCV criteria for corner refinement
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Prepare object points
        self.objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        
        # Calibration data storage
        self.objpoints = []  # 3D points
        self.imgpoints = []  # 2D points
        self.images_captured = 0
        
        # Auto-capture settings
        self.auto_capture_enabled = False
        self.capture_interval = 2.0
        self.last_capture_time = 0
        
        # Camera objects
        self.camera_type = None
        self.cap = None
        self.ep_robot = None
        self.ep_camera = None
        
    def display_main_menu(self):
        """Display main menu"""
        print("\n" + "="*60)
        print("        UNIFIED CAMERA CALIBRATION SYSTEM")
        print("="*60)
        print("\nCalibration Modes:")
        print("  1 - Use existing folder of images")
        print("  2 - Capture new images (manual/auto capture)")
        print("  3 - Settings (chessboard size, intervals)")
        print("  4 - Exit")
        print("="*60)
        
    def display_camera_menu(self):
        """Display camera selection menu"""
        print("\n" + "="*60)
        print("Select Camera Type:")
        print("="*60)
        if ROBOMASTER_AVAILABLE:
            print("  1 - Standard Webcam")
            print("  2 - RoboMaster Camera")
            print("  3 - Back to main menu")
        else:
            print("  1 - Standard Webcam (RoboMaster not available)")
            print("  2 - Back to main menu")
        print("="*60)
        
    def display_capture_menu(self):
        """Display capture controls"""
        print("\n" + "="*60)
        print("CAPTURE MODE - Controls:")
        print("="*60)
        print("  'c' - Capture single image")
        print("  'a' - Toggle auto-capture ON/OFF")
        print("  's' - Set capture interval")
        print("  'i' - Show status")
        print("  'd' - Done capturing (proceed to calibration)")
        print("  'q' - Quit without calibrating")
        print("="*60)
        print(f"Auto-capture: {'ON' if self.auto_capture_enabled else 'OFF'}")
        print(f"Interval: {self.capture_interval}s | Captured: {self.images_captured}")
        print("="*60)
        
    def display_settings_menu(self):
        """Display and handle settings"""
        while True:
            print("\n" + "="*60)
            print("SETTINGS")
            print("="*60)
            print(f"Current chessboard size: {self.chessboard_size[0]}x{self.chessboard_size[1]}")
            print(f"Save directory: {self.save_dir}")
            print(f"Auto-capture interval: {self.capture_interval}s")
            print("\n  1 - Change chessboard size")
            print("  2 - Change save directory")
            print("  3 - Change auto-capture interval")
            print("  4 - Back to main menu")
            print("="*60)
            
            choice = input("\nEnter choice: ").strip()
            
            if choice == '1':
                try:
                    width = int(input("Enter width (internal corners, e.g., 9): "))
                    height = int(input("Enter height (internal corners, e.g., 6): "))
                    if width >= 3 and height >= 3:
                        self.chessboard_size = (width, height)
                        # Recalculate objp
                        self.objp = np.zeros((width * height, 3), np.float32)
                        self.objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
                        print(f"Chessboard size set to {width}x{height}")
                    else:
                        print("Width and height must be at least 3")
                except ValueError:
                    print("Invalid input")
                    
            elif choice == '2':
                new_dir = input("Enter new save directory: ").strip()
                if new_dir:
                    self.save_dir = new_dir
                    os.makedirs(self.save_dir, exist_ok=True)
                    print(f"Save directory set to {new_dir}")
                    
            elif choice == '3':
                try:
                    interval = float(input("Enter interval in seconds (0.1-300): "))
                    if 0.1 <= interval <= 300:
                        self.capture_interval = interval
                        print(f"Interval set to {interval}s")
                    else:
                        print("Interval must be between 0.1 and 300 seconds")
                except ValueError:
                    print("Invalid input")
                    
            elif choice == '4':
                break
                
    def initialize_camera(self, camera_type):
        """Initialize the selected camera"""
        self.camera_type = camera_type
        
        if camera_type == "webcam":
            print("Initializing webcam...")
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("ERROR: Could not open webcam")
                return False
            print("Webcam initialized successfully")
            return True
            
        elif camera_type == "robomaster":
            if not ROBOMASTER_AVAILABLE:
                print("ERROR: RoboMaster SDK not available")
                return False
            print("Initializing RoboMaster...")
            try:
                self.ep_robot = robot.Robot()
                self.ep_robot.initialize(conn_type="sta")
                self.ep_camera = self.ep_robot.camera
                self.ep_camera.start_video_stream(display=False)
                print("RoboMaster initialized successfully")
                return True
            except Exception as e:
                print(f"ERROR initializing RoboMaster: {e}")
                return False
        
        return False
        
    def read_frame(self):
        """Read a frame from the active camera"""
        if self.camera_type == "webcam" and self.cap:
            ret, frame = self.cap.read()
            return frame if ret else None
        elif self.camera_type == "robomaster" and self.ep_camera:
            return self.ep_camera.read_cv2_image(strategy="newest")
        return None
        
    def cleanup_camera(self):
        """Clean up camera resources"""
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.ep_camera:
            self.ep_camera.stop_video_stream()
            self.ep_camera = None
        if self.ep_robot:
            self.ep_robot.close()
            self.ep_robot = None
        cv2.destroyAllWindows()
        
    def save_image(self, img, auto=False):
        """Save captured image"""
        timestamp = int(time.time() * 1000)
        self.images_captured += 1
        
        prefix = "auto" if auto else "manual"
        filename = os.path.join(self.save_dir, 
                               f"{prefix}_{self.images_captured:04d}_{timestamp}.jpg")
        cv2.imwrite(filename, img)
        
        print(f"{'Auto' if auto else 'Manual'} captured: {os.path.basename(filename)}")
        return filename
        
    def should_auto_capture(self):
        """Check if auto-capture should trigger"""
        if not self.auto_capture_enabled:
            return False
        current_time = time.time()
        if current_time - self.last_capture_time >= self.capture_interval:
            self.last_capture_time = current_time
            return True
        return False
        
    def capture_mode(self):
        """Capture images with manual/auto control"""
        self.display_capture_menu()
        
        cv2.namedWindow("Camera Calibration - Capture Mode", cv2.WINDOW_NORMAL)
        
        print("\nCamera feed started. Show chessboard from different angles.")
        
        while True:
            frame = self.read_frame()
            if frame is None:
                print("Warning: Could not read frame")
                time.sleep(0.1)
                continue
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect chessboard
            ret_corners, corners = cv2.findChessboardCorners(
                gray, self.chessboard_size, None)
            
            # Create display frame
            display_frame = frame.copy()
            
            # Draw corners if detected
            if ret_corners:
                corners_refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), self.criteria)
                cv2.drawChessboardCorners(
                    display_frame, self.chessboard_size, corners_refined, ret_corners)
                status_color = (0, 255, 0)
                status_text = "Chessboard DETECTED - Ready to capture"
            else:
                status_color = (0, 0, 255)
                status_text = "Chessboard NOT detected"
            
            # Add status overlay
            cv2.putText(display_frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            info_text = f"Auto: {'ON' if self.auto_capture_enabled else 'OFF'} | "
            info_text += f"Interval: {self.capture_interval}s | Count: {self.images_captured}"
            cv2.putText(display_frame, info_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if self.auto_capture_enabled:
                time_since = time.time() - self.last_capture_time
                time_until = max(0, self.capture_interval - time_since)
                countdown = f"Next capture: {time_until:.1f}s"
                cv2.putText(display_frame, countdown, (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Auto-capture check
            if ret_corners and self.should_auto_capture():
                self.save_image(frame, auto=True)
            
            cv2.imshow("Camera Calibration - Capture Mode", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c') and ret_corners:  # Manual capture
                self.save_image(frame, auto=False)
                
            elif key == ord('a'):  # Toggle auto-capture
                self.auto_capture_enabled = not self.auto_capture_enabled
                status = "ENABLED" if self.auto_capture_enabled else "DISABLED"
                print(f"\nAuto-capture {status}")
                if self.auto_capture_enabled:
                    self.last_capture_time = time.time()
                    
            elif key == ord('s'):  # Set interval
                print(f"\nCurrent interval: {self.capture_interval}s")
                try:
                    new_interval = float(input("Enter new interval (0.1-300): "))
                    if 0.1 <= new_interval <= 300:
                        self.capture_interval = new_interval
                        print(f"Interval set to {new_interval}s")
                    else:
                        print("Invalid interval")
                except ValueError:
                    print("Invalid input")
                    
            elif key == ord('i'):  # Show info
                self.display_capture_menu()
                
            elif key == ord('d'):  # Done capturing
                if self.images_captured >= 10:
                    print(f"\nDone capturing. Total images: {self.images_captured}")
                    break
                else:
                    print(f"\nNeed at least 10 images. Currently have {self.images_captured}")
                    
            elif key == ord('q'):  # Quit
                print("\nQuitting without calibration")
                return False
                
        return True
        
    def process_folder(self, folder_path):
        """Process images from existing folder"""
        if not os.path.exists(folder_path):
            print(f"ERROR: Folder '{folder_path}' does not exist")
            return False
            
        # Find images
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
            image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
        
        if not image_files:
            print(f"ERROR: No images found in '{folder_path}'")
            return False
        
        print(f"\nFound {len(image_files)} images")
        print("Processing for chessboard detection...")
        
        # Reset calibration data
        self.objpoints = []
        self.imgpoints = []
        processed = 0
        skipped = 0
        image_size = None
        
        for i, img_path in enumerate(sorted(image_files)):
            print(f"[{i+1}/{len(image_files)}] {os.path.basename(img_path)}", end=" ... ")
            
            img = cv2.imread(img_path)
            if img is None:
                print("FAILED (could not load)")
                skipped += 1
                continue
            
            if image_size is None:
                h, w = img.shape[:2]
                image_size = (w, h)
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret_corners, corners = cv2.findChessboardCorners(
                gray, self.chessboard_size, None)
            
            if ret_corners:
                corners_refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), self.criteria)
                self.objpoints.append(self.objp)
                self.imgpoints.append(corners_refined)
                processed += 1
                print("SUCCESS")
            else:
                print("FAILED (no chessboard)")
                skipped += 1
        
        self.images_captured = processed
        print(f"\nResults: {processed} successful, {skipped} skipped")
        
        if processed >= 10:
            print("Sufficient images for calibration")
            return self.perform_calibration(image_size)
        else:
            print(f"Insufficient images ({processed} < 10)")
            return False
            
    def perform_calibration(self, image_size):
        """Perform camera calibration"""
        print("\n" + "="*60)
        print("PERFORMING CALIBRATION...")
        print("="*60)
        
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, image_size, None, None)
        
        if not ret:
            print("Calibration FAILED")
            return False
        
        # Calculate reprojection error
        total_error = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(
                self.objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        
        mean_error = total_error / len(self.objpoints)
        
        # Save calibration
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"camera_calibration_{timestamp}.pkl"
        
        calibration_data = {
            'camera_matrix': mtx,
            'dist_coeffs': dist,
            'image_size': image_size,
            'reprojection_error': mean_error,
            'chessboard_size': self.chessboard_size,
            'num_images': len(self.objpoints)
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(calibration_data, f)
        
        # Display results
        print("\n" + "="*60)
        print("✓ CALIBRATION SUCCESSFUL!")
        print("="*60)
        print(f"Images used: {len(self.objpoints)}")
        print(f"Chessboard size: {self.chessboard_size[0]}x{self.chessboard_size[1]}")
        print(f"Reprojection error: {mean_error:.4f} pixels")
        print(f"\nCamera Matrix:")
        print(mtx)
        print(f"\nDistortion Coefficients:")
        print(dist)
        print(f"\nSaved to: {filename}")
        print("="*60)
        
        return True
        
    def run(self):
        """Main application loop"""
        while True:
            self.display_main_menu()
            choice = input("\nEnter choice: ").strip()
            
            if choice == '1':  # Existing folder
                folder = input("Enter folder path: ").strip()
                if folder:
                    self.process_folder(folder)
                    
            elif choice == '2':  # Capture new images
                self.display_camera_menu()
                cam_choice = input("\nEnter choice: ").strip()
                
                camera_type = None
                if cam_choice == '1':
                    camera_type = "webcam"
                elif cam_choice == '2' and ROBOMASTER_AVAILABLE:
                    camera_type = "robomaster"
                    
                if camera_type and self.initialize_camera(camera_type):
                    success = self.capture_mode()
                    self.cleanup_camera()
                    
                    if success:
                        # Ask if user wants to calibrate now
                        cal = input("\nCalibrate now with captured images? (y/n): ").strip().lower()
                        if cal == 'y':
                            self.process_folder(self.save_dir)
                            
            elif choice == '3':  # Settings
                self.display_settings_menu()
                
            elif choice == '4':  # Exit
                print("\nExiting. Goodbye!")
                break
                
            else:
                print("Invalid choice")


def main():
    """Main entry point"""
    print("\n" + "="*60)
    print("  UNIFIED CAMERA CALIBRATION SYSTEM")
    print("  Supports: Webcam, RoboMaster, Folder processing")
    print("="*60)
    
    calibrator = UnifiedCalibrator()
    
    try:
        calibrator.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        calibrator.cleanup_camera()


if __name__ == "__main__":
    main()
