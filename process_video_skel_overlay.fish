#! /bin/fish

set dataset_dir ~/datasets/Cambridge1
set pose_dir data/camD/poses_tracked
set output_dir skel_vids
for p in $dataset_dir/*B1*F.mp4
    echo $p
    set file_basename (path basename -E $p)
    echo $file_basename
    uv run visualize_yolo_skeleton.py --video $p --conf 0.01 -j $pose_dir/$file_basename"_poses.json" --output $output_dir/$file_basename"_annoated.mp4" --no-display --draw-objects

end

