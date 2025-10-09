import os
import json
import cv2
from ultralytics import YOLO
from tqdm import tqdm

def extract_objects_from_videos(
    video_dir,
    output_dir,
    model_path='yolov8x.pt',
    target_classes=['backpack', 'knife', 'cell phone'],  # YOLOv8 class names
    conf_threshold=0.5
):
    """
    Extracts specific objects from each frame of videos using YOLOv8,
    and saves results to JSON files (one per video).

    Args:
        video_dir (str): Folder containing videos.
        output_dir (str): Folder to save JSON results.
        model_path (str): Path or name of YOLOv8 detection model.
        target_classes (list): List of object names to detect.
        conf_threshold (float): Minimum detection confidence.
    """
    os.makedirs(output_dir, exist_ok=True)
    model = YOLO(model_path)

    # Map target class names to class indices
    class_names = model.names  # YOLOv8 class names
    target_class_ids = [i for i, name in class_names.items() if name in target_classes]

    video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    if not video_files:
        print(f"No video files found in {video_dir}.")
        return

    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        output_path = os.path.join(output_dir, os.path.splitext(video_file)[0] + '_objects.json')

        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Processing {video_file} ({frame_count} frames)...")

        video_data = {"video_name": video_file, "frames": []}

        for frame_idx in tqdm(range(frame_count), desc=f"Detecting objects in {video_file}"):
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame, conf=conf_threshold, verbose=False)

            frame_objects = []
            for r in results:
                if r.boxes is not None:
                    boxes = r.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                    class_ids = r.boxes.cls.cpu().numpy().astype(int)
                    confs = r.boxes.conf.cpu().numpy()

                    for box, cls_id, conf in zip(boxes, class_ids, confs):
                        if cls_id in target_class_ids:
                            x1, y1, x2, y2 = box
                            cx = (x1 + x2) / 2
                            cy = (y1 + y2) / 2
                            frame_objects.append({
                                "object_name": class_names[cls_id],
                                "confidence": float(conf),
                                "center": [float(cx), float(cy)]
                            })

            video_data["frames"].append({
                "frame_index": frame_idx,
                "objects": frame_objects
            })

        cap.release()

        with open(output_path, "w") as f:
            json.dump(video_data, f)

        print(f"Saved objects to {output_path}")

if __name__ == "__main__":
    video_dir = "../data/27_03_2025-vid" # Folder containing videos
    output_dir = "../objects"        # Folder to save JSON results
    model_path = "yolov8x.pt"     # YOLOv8 detection model (auto-downloads if missing)

    extract_objects_from_videos(video_dir, output_dir, model_path)
