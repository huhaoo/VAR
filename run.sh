clear
SELECTED_GPU=$(python select_gpu.py)
echo "Selected GPU: $SELECTED_GPU"
mkdir -p data/train/class0
mkdir -p data/val
rm data/val/class0 -r
cp data/train/class0 data/val/class0 -r
# rm local_output -r
rm nohup.out
# CUDA_VISIBLE_DEVICES=$SELECTED_GPU nohup torchrun train.py --depth=16 --bs=32 --ep=2000 --fp16=1 --alng=1e-3 --wpe=0.1 --data_path=data/ --mid_reso=1 & #--tblr=4.167e-06 & #--master_addr=127.0.0.1 --master_port=29500 --nnodes=1 --node_rank=0 --nproc_per_node=8
# CUDA_VISIBLE_DEVICES=$SELECTED_GPU torchrun train.py --depth=16 --bs=32 --ep=2000 --fp16=1 --alng=1e-3 --wpe=0.1 --data_path=data/  #--tblr=4.167e-06 & #--master_addr=127.0.0.1 --master_port=29500 --nnodes=1 --node_rank=0 --nproc_per_node=8
# CUDA_VISIBLE_DEVICES=$SELECTED_GPU python demo_zero_shot_edit.py
# CUDA_VISIBLE_DEVICES=$SELECTED_GPU python paint.py
# python load_ckpt.py
CUDA_VISIBLE_DEVICES=$SELECTED_GPU python main.py