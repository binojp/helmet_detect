import cv2
import os
import time
from src.inference import HelmetPredictor

MODEL_PATH = "artifacts/best_helmet_model.pt"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

predictor = HelmetPredictor(MODEL_PATH)


def run_video(source=0, output_name="output.mp4", conf_threshold=0.5):

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Could not open video source")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(cap.get(cv2.CAP_PROP_FPS)) or 25

    output_path = os.path.join(OUTPUT_DIR, output_name)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps_input, (width, height))

    print("Starting detection... Press 'q' to quit")

    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Start time
        start_time = time.time()

        annotated_frame, helmet_count, no_helmet_count = predictor.predict_frame(
            frame, conf_threshold
        )

        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
        prev_time = current_time

        # Overlay information
        cv2.putText(
            annotated_frame,
            f"FPS: {int(fps)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            annotated_frame,
            f"Helmet: {helmet_count}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
        )

        cv2.putText(
            annotated_frame,
            f"No Helmet: {no_helmet_count}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

        out.write(annotated_frame)
        cv2.imshow("Helmet Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Saved output to: {output_path}")


if __name__ == "__main__":

    print("Choose mode:")
    print("1 - Live Webcam")
    print("2 - Video File")

    choice = input("Enter choice: ")

    conf_threshold = float(input("Enter confidence threshold (0.0 - 1.0): "))

    if choice == "1":
        run_video(source=0, output_name="webcam_output.mp4", conf_threshold=conf_threshold)
    elif choice == "2":
        video_path = input("Enter video file path: ")
        run_video(source=video_path, output_name="video_output.mp4", conf_threshold=conf_threshold)
    else:
        print("Invalid choice")