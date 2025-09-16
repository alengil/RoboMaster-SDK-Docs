import cv2
import time
from ultralytics import YOLO
from robomaster import robot

# --- Configuration ---
CONF_THRESHOLD = 0.5  # YOLO confidence threshold
MODEL_PATH = "<MODEL_PATH>"  # Path to your trained YOLOv8 model

# --- Movement Speeds ---
FWD_SPEED = 0.5   # Forward/Backward speed (m/s)
SIDE_SPEED = 0.5  # Left/Right speed (m/s)
ROT_SPEED = 30    # Rotation speed (degrees/s)


def main():
    # Load YOLO model
    model = YOLO(MODEL_PATH)

    # Initialize RoboMaster
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="sta")
    ep_camera = ep_robot.camera
    ep_chassis = ep_robot.chassis

    ep_camera.start_video_stream(display=False)
    print("RoboMaster connected. Camera stream started.")
    print("-" * 20)
    print("Controls:")
    print(" W, A, S, D = Move Robot")
    print(" Q, E = Turn Robot")
    print(" ESC = Quit Program")
    print("-" * 20)

    try:
        while True:
            img = ep_camera.read_cv2_image(strategy="newest")
            if img is None:
                continue

            # Run YOLO detection
            results = model.predict(img, conf=CONF_THRESHOLD, verbose=False)

            # Draw detections
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]

                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("RoboMaster YOLO Detection", img)

            # --- Keyboard Control ---
            key = cv2.waitKey(1) & 0xFF

            x_speed, y_speed, z_speed = 0, 0, 0

            if key == ord("w"):
                x_speed = FWD_SPEED
            elif key == ord("s"):
                x_speed = -FWD_SPEED
            elif key == ord("a"):
                y_speed = -SIDE_SPEED
            elif key == ord("d"):
                y_speed = SIDE_SPEED
            elif key == ord("q"):
                z_speed = -ROT_SPEED
            elif key == ord("e"):
                z_speed = ROT_SPEED
            elif key == 27:  # ESC
                print("Escape key pressed. Shutting down.")
                break

            ep_chassis.drive_speed(x=x_speed, y=y_speed, z=z_speed, timeout=0.5)

    finally:
        # Cleanup
        print("Cleaning up resources...")
        ep_chassis.drive_speed(x=0, y=0, z=0, timeout=1)
        ep_camera.stop_video_stream()
        ep_robot.close()
        cv2.destroyAllWindows()
        print("Cleanup complete. Exiting.")


if __name__ == "__main__":
    main()
