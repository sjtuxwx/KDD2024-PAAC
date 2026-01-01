#!/bin/bash

# PAAC 运行脚本
# 基于 KDD2024-PAAC 项目

echo "开始运行 PAAC 实验..."

# 激活conda环境
echo "激活 PAAC conda 环境..."
# conda activate PAAC

# 检查GPU是否可用
echo "检查GPU状态..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# 创建结果目录
# mkdir -p OOD_result

echo "=========================================="
echo "运行 Yelp2018 数据集实验"
echo "=========================================="

python PAAC_main_origin.py \
    --dataset_name yelp2018 \
    --layers_list '[5]' \
    --cl_rate_list '[10]' \
    --align_reg_list '[1e3]' \
    --lambada_list '[0.8]' \
    --gama_list '[0.8]' \
    --device 0 \
    --batch_size 2048 \
    --lr 0.001 \
    --decay 0.0001 \
    --emb_size 64 \
    --num_epoch 1000 \
    --EarlyStop 10 \
    --topks '[20]' \
    --seed 12345

python PAAC_maingexingdu_dro.py \
    --dataset_name yelp2018 \
    --layers_list '[5]' \
    --cl_rate_list '[10]' \
    --align_reg_list '[1e3]' \
    --lambada_list '[0.8]' \
    --gama_list '[0.8]' \
    --device 0 \
    --batch_size 2048 \
    --lr 0.001 \
    --decay 0.0001 \
    --emb_size 64 \
    --num_epoch 1000 \
    --EarlyStop 10 \
    --topks '[20]' \
    --seed 12345

echo "=========================================="
echo "运行 Gowalla 数据集实验"
echo "=========================================="

python PAAC_main_gexingdu.py \
    --dataset_name gowalla \
    --layers_list '[6]' \
    --cl_rate_list '[5]' \
    --align_reg_list '[50]' \
    --lambada_list '[0.2]' \
    --gama_list '[0.2]' \
    --device 0 \
    --batch_size 2048 \
    --lr 0.001 \
    --decay 0.0001 \
    --emb_size 64 \
    --num_epoch 1000 \
    --EarlyStop 10 \
    --topks '[20]' \
    --seed 12345

echo "=========================================="
echo "所有实验完成！"
echo "结果保存在 OOD_result 目录中"
echo "=========================================="



# == 基础实验
CUDA_VISIBLE_DEVICES=3 python PAAC_main_gexingdu.py \
    --dataset_name epinions \
    --layers_list '[5]' \
    --cl_rate_list '[0.0]' \
    --align_reg_list '[0.0]' \
    --lambada_list '[0.0]' \
    --gama_list '[0.0]' \
    --device 0 \
    --batch_size 2048 \
    --lr 0.001 \
    --decay 0.0001 \
    --emb_size 64 \
    --num_epoch 1000 \
    --EarlyStop 10 \
    --topks '[20]' \
    --seed 12345 \
    --tau_list '[0.2]' \
    --pop_gamma_list '[0.2]'

python PAAC_main_gexingdu.py     --dataset_name epinions  --layers_list '[5]'     --cl_rate_list '[5]'     --align_reg_list '[50]'    --lambada_list '[0.2]'  --gama_list '[0.2]'     --device 1     --batch_size 2048     --lr 0.001     --decay 0.0001     --emb_size 64  --num_epoch 1000     --EarlyStop 10     --topks '[20]'     --seed 1234 --tau_list '[0.2]' --pop_gamma_list '[0.2]'
 python PAAC_main_origin.py     --dataset_name epinions     --layers_list '[5]'     --cl_rate_list '[2]'     --align_reg_list '[10]'    --lambada_list '[0.2]'     --gama_list '[0.2]'     --device 1     --batch_size 2048     --lr 0.001     --decay 0.0001     --emb_size 64     --num_epoch 1000     --EarlyStop 10     --topks '[20]'     --seed 1234


 CUDA_VISIBLE_DEVICES=0 python PAAC_main_XSimGCL.py \
    --dataset_name epinions \
    --layers_list '[5]' \
    --cl_rate_list '[0.0]' \
    --align_reg_list '[0.0]' \
    --lambada_list '[0.0]' \
    --gama_list '[0.0]' \
    --device 0 \
    --batch_size 2048 \
    --lr 0.001 \
    --decay 0.0001 \
    --emb_size 64 \
    --num_epoch 1000 \
    --EarlyStop 10 \
    --topks '[20]' \
    --seed 12345 \
    --keep_layers_list '[1]'


python PAAC_main_gexingdu_dro.py \
    --dataset_name ml-1m \
    --layers_list '[2]' \
    --cl_rate_list '[0.1]' \
    --align_reg_list '[0.5]' \
    --lambada_list '[0.8]' \
    --gama_list '[0.8]' \
    --device 2 \
    --batch_size 2048 \
    --lr 0.001 \
    --decay 0.0001 \
    --emb_size 64 \
    --num_epoch 1000 \
    --EarlyStop 10 \
    --topks '[20]' \
    --seed 12345 \
    --tau_list '[0.5]' \
    --pop_gamma_list '[0.2]' \
    --rou_list '[0.2]'