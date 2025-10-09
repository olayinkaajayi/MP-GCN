import os
import cv2
import json
from ultralytics import YOLO
from tqdm import tqdm

def extract_pose_from_videos(
    video_dir,
    output_dir,
    model_path='yolov8x-pose.pt',
    conf_threshold=0.5
):
    """
    Extracts human poses from each frame of videos in video_dir using YOLOv8x-pose,
    and saves results to JSON files (one per video).

    Args:
        video_dir (str): Path to folder containing video files.
        output_dir (str): Folder to save extracted pose files.
        model_path (str): Path or name of YOLOv8 pose model.
        conf_threshold (float): Detection confidence threshold.
    """
    os.makedirs(output_dir, exist_ok=True)
    model = YOLO(model_path)

    video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    if not video_files:
        print(f"No video files found in {video_dir}.")
        return

    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        output_path = os.path.join(output_dir, os.path.splitext(video_file)[0] + '_poses.json')

        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Processing {video_file} ({frame_count} frames)...")

        video_data = {"video_name": video_file, "frames": []}

        for frame_idx in tqdm(range(frame_count), desc=f"Extracting {video_file}"):
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame, conf=conf_threshold, verbose=False)

            # Collect poses for all detected persons
            frame_poses = []
            for r in results:
                if hasattr(r, "keypoints") and r.keypoints is not None:
                    kpts = r.keypoints.xy.cpu().numpy()  # shape: (num_persons, num_keypoints, 2)
                    conf = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None

                    for pid, person_kpts in enumerate(kpts):
                        frame_poses.append({
                            "person_id": pid,
                            "keypoints": person_kpts.tolist(),
                            "confidence": conf[pid].tolist() if conf is not None else None
                        })

            video_data["frames"].append({
                "frame_index": frame_idx,
                "poses": frame_poses
            })

        cap.release()

        # Save poses to JSON file
        with open(output_path, "w") as f:
            json.dump(video_data, f)

        print(f"✅ Saved poses to {output_path}")

if __name__ == "__main__":
    # Example usage:
    video_dir = "../data/27_03_2025-vid" # folder containing .mp4 etc.
    output_dir = "../poses"          # folder to save results
    model_path = "yolov8x-pose.pt"  # pre-trained model (auto-downloads if missing)

    extract_pose_from_videos(video_dir, output_dir, model_path)


# COCO 17-keypoint skeleton (edges between keypoints)
# skeleton = [
#     (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
#     (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
# ]
