import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time object detection with YOLOv8"
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index (default: 0)",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum detection confidence (default: 0.5)",
    )

    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLO model to use (default: yolov8n.pt)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading YOLO model: {args.model}")
    model = YOLO(args.model)
    print("Model loaded.")

    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {args.camera}. "
            "Try --camera 1 or --camera 2."
        )

    time.sleep(1)

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    previous_time = time.time()
    screenshot_count = 0

    print("\nControls:")
    print("  q - quit")
    print("  s - save current detection frame\n")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Warning: could not read frame.")
            continue

        results = model(
            frame,
            conf=args.confidence,
            verbose=False,
        )

        annotated_frame = results[0].plot()

        # Calculate FPS
        current_time = time.time()
        elapsed = current_time - previous_time

        fps = 1 / elapsed if elapsed > 0 else 0
        previous_time = current_time

        # Count detected objects
        detection_count = len(results[0].boxes)

        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            annotated_frame,
            f"Objects: {detection_count}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.imshow(
            "YOLOv8 Real-Time Object Detection",
            annotated_frame,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("s"):
            screenshot_count += 1
            output_path = (
                output_dir
                / f"detection_{screenshot_count:03d}.jpg"
            )

            cv2.imwrite(
                str(output_path),
                annotated_frame,
            )

            print(f"Saved: {output_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
