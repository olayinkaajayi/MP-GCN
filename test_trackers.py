import cv2
import numpy as np
from ultralytics import YOLO
"""
Sample code for viewing how well ultralytics trackers perform on a video.

"""

model = YOLO("yolo11x-seg.pt")  # Or yolo11x-pose.pt / yolo11x-seg.pt for higher accuracy
cap = cv2.VideoCapture("/its/home/tafj0/datasets/Cambridge1/BLUEB4P5F.mp4")
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
output_path = "BLUEB4P5F_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
print(f"💾 Output will be saved to: {output_path}")
# ID Re-mapping memory for 6 subjects max
max_ids = 6
id_map = {}  # Maps raw tracker ID -> Fixed ID (1 to 6)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, tracker="config/tracker/deepocsort.yaml", persist=True, classes=[0], verbose=False)  # 0 = person

    if results[0].boxes and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        raw_ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, raw_id in zip(boxes, raw_ids):
            # Assign first 6 unique raw IDs to fixed IDs 1..6
            if raw_id not in id_map and len(id_map) < max_ids:
                id_map[raw_id] = len(id_map) + 1

            # Retrieve fixed ID if mapped
            assigned_id = id_map.get(raw_id, "Unassigned")

            # Render Bounding Box and Fixed ID
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.putText(frame, f"Person {assigned_id}", (int(box[0]), int(box[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    writer.write(frame)

    #cv2.imshow("Tracked Scene", frame)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break
writer.release()
cap.release()
cv2.destroyAllWindows()