import cv2
import numpy as np
from pathlib import Path
# 1. Define configuration constants

"""
Code to detect blue circles in a video. Some videos have been corrupted by displaying pose detection points
"""
lower_bound = np.array([240, 0, 0])
upper_bound = np.array([255, 10, 20])
def detect_corrupted_circles(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_count = 0



    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        frame_count += 1
        #if frame_count == 10:
        #    cv2.imwrite('first_frame.png',frame)
        # Create a mask matching the exact corrupt pixel color
        # Since compression can sometimes shift values slightly,
        # we check for an exact match first.
        color_mask = cv2.inRange(frame, lower_bound, upper_bound)
        #if frame_count == 10:
        #    cv2.imwrite('first_mask.png',color_mask)
        # Find isolated groups of pixels (contours) matching the color
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #print(len(contours))
        counter=0
        for i, contour in enumerate(contours):
            # Calculate the area (in pixels) of the detected anomaly
            area = cv2.contourArea(contour)

            # Filter for small circle artifacts (e.g., between 1 and 50 pixels total area)
            if 1 <= area <= 50:
                counter+=1
        if counter>0:
            print(f'Found {counter} dots')
        if counter>100:
            cap.release()
            return True
        if frame_count>15:
            cap.release()
            return False


    #print("Scan complete.")

if __name__ == "__main__":
    video_path = Path('/its/home/tafj0/datasets/Cambridge1/')
    file_list =[]
    for path in video_path.glob("*.mp4"):
        print(f'Testing {path.stem}')
        res= detect_corrupted_circles(path)
        if res:
            print(f"Dots: {res} in {path.stem}")
            file_list.append(path.stem)

    with open('iffy_dots.txt','wt') as f:
        f.write("\n".join(str(item) for item in file_list))

