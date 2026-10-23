#! /bin/fish

set dataset_dir ~/datasets/Cambridge1
set pose_dir data/camD/poses_tracked_OC
set output_dir skel_vids_OC
set action_list "B1" "B2"
#set action_list "B2" "B4" "F1" "F2" "F3" "F4"
for action in $action_list
    echo "Action "$action
    for p in $dataset_dir/*$action*F.mp4
        echo $p
        set file_basename (path basename -E $p)
        echo $file_basename
        mkdir -p $output_dir/$action
        uv run visualize_yolo_skeleton.py --video $p --conf 0.01 -j $pose_dir/$file_basename"_poses.json" --output $output_dir/$action/$file_basename"_annoated.mp4" --no-display --draw-objects --box-target

    end
end
