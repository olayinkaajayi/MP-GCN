
from pathlib import Path
import json
import ffmpeg
"""
Code to reproduce videos that have been marked as corrupted
"""

root_path = Path('/its/home/tafj0/mnt/16TB/IntentMAPS/real_proc2')
output_path = Path('corrrected_videos')
output_path.mkdir(parents=True,exist_ok=True)
with open('iffy_dots.txt','rt') as f:
    filenames = [line.strip() for line in f]

print(filenames)
for file_stem  in filenames:
    full_path = list(root_path.rglob(file_stem))
    if len(full_path) != 1:  #check theres only one
        print(f'error with {full_path}')
    full_path = full_path[0]
    with open(full_path/'config.json', 'rt') as j:
        conf = json.load(j)
    fps = conf['fps']
    print(fps)
    print(full_path)

    (
        ffmpeg
        .input(str(full_path/'left/%06d.png'), framerate=fps)
        .output(str(output_path/f'{file_stem}.mp4'),vcodec='libx264', pix_fmt='yuv420p')
        .run()
    )
