python PAAC_main_gexingdu_twoloss_dro.py \
     --dataset_name ml-1m \
    --layers_list '[2]' \
    --cl_rate_list '[0.1]' \
    --align_reg_list '[0.5]' \
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
    --seed 12345 \
    --tau_list '[0.5, 0.2]' \
    --pop_gamma_list '[0.2]' \
    --inter_rate_list '[0.3, 0.5, 1.0, 2.0, 5.0]' \
    --origin_bpr_rate_list '[1.0]' \
    --dro_rate_list '[0.1, 0.2, 0.5, 0.7, 1.0]'


python PAAC_main_gexingdu_twoloss_dro.py \
    --dataset_name ml-1m \
    --layers_list '[2]' \
    --cl_rate_list '[0.1]' \
    --align_reg_list '[0.5]' \
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
    --seed 12345 \
    --tau_list '[0.5, 0.2]' \
    --pop_gamma_list '[0.2]' \
    --inter_rate_list '[0.0]' \
    --origin_bpr_rate_list '[1.0]' \
    --dro_rate_list '[0.0]'



python PAAC_main_gexingdu_twoloss_dro.py \
   --dataset_name ml-1m \
    --layers_list '[2]' \
    --cl_rate_list '[0.1]' \
    --align_reg_list '[0.5]' \
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
    --seed 12345 \
    --tau_list '[0.5, 0.2]' \
    --pop_gamma_list '[0.2]' \
    --inter_rate_list '[1.0]' \
    --origin_bpr_rate_list '[0.0]' \
    --dro_rate_list '[0.0]'

