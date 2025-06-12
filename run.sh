clear
SELECTED_GPU=$(python select_gpu.py)
echo "Selected GPU: $SELECTED_GPU"
CUDA_VISIBLE_DEVICES=$SELECTED_GPU python main.py