source /home/your_path/miniconda3/etc/profile.d/conda.sh
conda activate dflow_grpo

CKPT_PATH=/home/your_path/DiscreteFlowRL/FUDOKI/checkpoints

torchrun --nproc_per_node 1 inference_t2i_local.py \
    --batch_size 1 \
    --checkpoint_path $CKPT_PATH \
    --text_embedding_path $CKPT_PATH/text_embedding.pt \
    --image_embedding_path  $CKPT_PATH/image_embedding.pt \
    --discrete_fm_steps 50 \
    --output_dir ./fudoki_output