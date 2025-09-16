import cv2
import os
import time
from robomaster import robot

# --- Configuration ---
PHOTO_LIMIT = 200  # The total number of photos to take
CAPTURE_INTERVAL = 1.0  # Time to wait between captures, in seconds
SAVE_DIR = "robomaster_dataset"  # Folder to save the images

# --- Movement Speeds ---
# You can adjust these values to make the robot move faster or slower
FWD_SPEED = 0.5  # Forward/Backward speed (m/s)
SIDE_SPEED = 0.5  # Left/Right speed (m/s)
ROT_SPEED = 30    # Rotation speed (degrees/s)


# --- Main Script ---

# Create the folder if it doesn't already exist
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Images will be saved in the '{SAVE_DIR}' folder.")

# Initialize the RoboMaster
ep_robot = robot.Robot()
try:
    ep_robot.initialize(conn_type="sta")
    print("RoboMaster connected.")
except Exception as e:
    print(f"Failed to initialize robot: {e}")
    print("Please ensure the robot is on and connected to the same Wi-Fi network.")
    exit()

# Get the camera and chassis objects
ep_camera = ep_robot.camera
ep_chassis = ep_robot.chassis

# Start the video stream
ep_camera.start_video_stream(display=False)
print("Camera stream started.")
print("-" * 20)
print("Controls:")
print(" W, A, S, D = Move Robot")
print(" Q, E = Turn Robot")
print(" ESC = Quit Program")
print("-" * 20)


# Create an OpenCV window
cv2.namedWindow("RoboMaster Control", cv2.WINDOW_NORMAL)

# Variables for capture logic
last_capture_time = time.time()
image_count = 0
capture_complete = False

try:
    while True:
        # Get the latest image from the camera
        img = ep_camera.read_cv2_image(strategy="newest")

        if img is None:
            continue

        # --- Automatic Photo Capture Logic ---
        if not capture_complete:
            current_time = time.time()
            if current_time - last_capture_time >= CAPTURE_INTERVAL:
                last_capture_time = current_time
                
                # Create a unique filename and save the image
                filename = os.path.join(SAVE_DIR, f"image_{int(time.time() * 1000)}.jpg")
                cv2.imwrite(filename, img)
                image_count += 1
                print(f"Saved image {image_count}/{PHOTO_LIMIT}: {filename}")

                if image_count >= PHOTO_LIMIT:
                    capture_complete = True
                    print("\nPhoto capture limit reached! You can continue driving.")

        # --- Display Information on Screen ---
        # Add a status text overlay on the image
        status_text = ""
        if not capture_complete:
            status_text = f"Capturing: {image_count}/{PHOTO_LIMIT}"
            color = (0, 255, 0) # Green
        else:
            status_text = "Capture Complete"
            color = (0, 0, 255) # Red

        cv2.putText(img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("RoboMaster Control", img)

        # --- Keyboard Control Logic ---
        key = cv2.waitKey(1) & 0xFF

        x_speed, y_speed, z_speed = 0, 0, 0

        if key == ord('w'):
            x_speed = FWD_SPEED
        elif key == ord('s'):
            x_speed = -FWD_SPEED
        elif key == ord('a'):
            y_speed = -SIDE_SPEED
        elif key == ord('d'):
            y_speed = SIDE_SPEED
        elif key == ord('q'):
            z_speed = -ROT_SPEED
        elif key == ord('e'):
            z_speed = ROT_SPEED
        elif key == 27:  # Escape key
            print("Escape key pressed. Shutting down.")
            break
        
        # Send the movement command to the robot's chassis
        ep_chassis.drive_speed(x=x_speed, y=y_speed, z=z_speed, timeout=0.5)

finally:
    # This block ensures the robot stops and cleans up resources
    print("Cleaning up resources...")
    # Send one last command to stop all movement
    ep_chassis.drive_speed(x=0, y=0, z=0, timeout=1)
    ep_camera.stop_video_stream()
    ep_robot.close()
    cv2.destroyAllWindows()
    print("Cleanup complete. Exiting.")
