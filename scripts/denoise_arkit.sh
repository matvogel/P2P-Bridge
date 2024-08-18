for scene in $(ls $1)
do
    echo "Processing $scene"
    python denoise_room.py --room_path $1/$scene/scans/iphone.ply --model_path ./pretrained/PVDL_ARK_XYZ/step_100000.pth --steps 5 --k 4 --use_ema --average_predictions
done