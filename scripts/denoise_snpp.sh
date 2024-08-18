for scene in $(ls $1)
do
    echo "Processing $scene"
    python denoise_room.py --room_path $1/$scene/scans/iphone.ply --model_path ./pretrained/PVDL_SNPP_XYZ/step_100000.pth --steps 10 --k 4
done