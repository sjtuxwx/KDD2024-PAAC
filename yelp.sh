# python PAAC_main_gexingdu_dro.py \
#     --dataset_name ml-1m \
#     --layers_list '[2]' \
#     --cl_rate_list '[0.2]' \
#     --align_reg_list '[1]' \
#     --lambada_list '[0.8]' \
#     --gama_list '[0.8]' \
#     --device 2 \
#     --batch_size 2048 \
#     --lr 0.001 \
#     --decay 0.0001 \
#     --emb_size 64 \
#     --num_epoch 1000 \
#     --EarlyStop 10 \
#     --topks '[20]' \
#     --seed 12345 \
#     --tau_list '[0.2]' \
#     --pop_gamma_list '[0.2]' \
    # --rou_list '[0.2]'


python PAAC_main_origin.py \
    --dataset_name yelp2018 \
    --layers_list '[5]' \
    --cl_rate_list '[10]' \
    --align_reg_list '[1e3]' \
    --lambada_list '[0.8]' \
    --gama_list '[0.8]' \
    --device 1 \
    --batch_size 2048 \
    --lr 0.001 \
    --decay 0.0001 \
    --emb_size 64 \
    --num_epoch 1000 \
    --EarlyStop 10 \
    --topks '[20]' \
    --seed 12345

python PAAC_main_gexingdu.py \
    --dataset_name yelp2018 \
    --layers_list '[5]' \
    --cl_rate_list '[10]' \
    --align_reg_list '[1e3]' \
    --lambada_list '[0.8]' \
    --gama_list '[0.8]' \
    --device 1 \
    --batch_size 2048 \
    --lr 0.001 \
    --decay 0.0001 \
    --emb_size 64 \
    --num_epoch 1000 \
    --EarlyStop 10 \
    --topks '[20]' \
    --seed 12345 \
    --tau_list '[0.2]' \
    --pop_gamma_list '[0.2]' \
    --inference_mode complex 

python PAAC_main_gexingdu_dro.py \
    --dataset_name yelp2018 \
    --layers_list '[5]' \
    --cl_rate_list '[10]' \
    --align_reg_list '[1e3]' \
    --lambada_list '[0.8]' \
    --gama_list '[0.8]' \
    --device 1 \
    --batch_size 2048 \
    --lr 0.001 \
    --decay 0.0001 \
    --emb_size 64 \
    --num_epoch 1000 \
    --EarlyStop 10 \
    --topks '[20]' \
    --seed 12345 \
    --tau_list '[0.2]' \
    --pop_gamma_list '[0.2]' \
    --rou_list '[0.2]'

python PAAC_main_XSimGCL.py \
    --dataset_name yelp2018 \
    --layers_list '[5]' \
    --cl_rate_list '[10]' \
    --align_reg_list '[1e3]' \
    --lambada_list '[0.8]' \
    --gama_list '[0.8]' \
    --device 1 \
    --batch_size 2048 \
    --lr 0.001 \
    --decay 0.0001 \
    --emb_size 64 \
    --num_epoch 1000 \
    --EarlyStop 10 \
    --topks '[20]' \
    --seed 12345 \
    --keep_layers_list '[4]'


python PAAC_main_gexingdu_caiyang.py \
    --dataset_name yelp2018 \
    --layers_list '[5]' \
    --cl_rate_list '[10]' \
    --align_reg_list '[1e3]' \
    --lambada_list '[0.8]' \
    --gama_list '[0.8]' \
    --device 1 \
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



python PAAC_main_gexingdu_norm_rating.py \
    --dataset_name yelp2018 \
    --layers_list '[5]' \
    --cl_rate_list '[10]' \
    --align_reg_list '[1e3]' \
    --lambada_list '[0.8]' \
    --gama_list '[0.8]' \
    --device 1 \
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

python PAAC_main_pgsm.py \
    --dataset_name yelp2018 \
    --layers_list '[5]' \
    --cl_rate_list '[10]' \
    --align_reg_list '[1e3]' \
    --lambada_list '[0.8]' \
    --gama_list '[0.8]' \
    --device 1 \
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


python PAAC_main_paae.py \
    --dataset_name yelp2018 \
    --layers_list '[5]' \
    --cl_rate_list '[10]' \
    --align_reg_list '[1e3]' \
    --lambada_list '[0.8]' \
    --gama_list '[0.8]' \
    --device 1 \
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

python PAAC_main_gexingdu_margin.py \
    --dataset_name yelp2018 \
    --layers_list '[5]' \
    --cl_rate_list '[10]' \
    --align_reg_list '[1e3]' \
    --lambada_list '[0.8]' \
    --gama_list '[0.8]' \
    --device 1 \
    --batch_size 2048 \
    --lr 0.001 \
    --decay 0.0001 \
    --emb_size 64 \
    --num_epoch 1000 \
    --EarlyStop 10 \
    --topks '[20]' \
    --seed 12345 \
    --tau_list '[0.2]' \
    --pop_gamma_list '[0.2]' \
    --margin_coeff_list '[0.2, 0.3, 0.4, 0.1]' \
    --margin_gamma_list '[0.2]'

python PAAC_main_gexingdu_pred.py \
    --dataset_name yelp2018 \
    --layers_list '[5]' \
    --cl_rate_list '[10]' \
    --align_reg_list '[1e3]' \
    --lambada_list '[0.8]' \
    --gama_list '[0.8]' \
    --device 1 \
    --batch_size 2048 \
    --lr 0.001 \
    --decay 0.0001 \
    --emb_size 64 \
    --num_epoch 1000 \
    --EarlyStop 10 \
    --topks '[20]' \
    --seed 12345 \
    --tau_list '[0.2]' \
    --pop_gamma_list '[0.2]' \
    --inference_mode complex 

python PAAC_main_gexingdu_residual.py \
    --dataset_name yelp2018 \
    --layers_list '[5]' \
    --cl_rate_list '[10]' \
    --align_reg_list '[1e3]' \
    --lambada_list '[0.8]' \
    --gama_list '[0.8]' \
    --device 1 \
    --batch_size 2048 \
    --lr 0.001 \
    --decay 0.0001 \
    --emb_size 64 \
    --num_epoch 1000 \
    --EarlyStop 10 \
    --topks '[20]' \
    --seed 12345 \
    --tau_list '[0.2]' \
    --pop_gamma_list '[0.2]' \
    --pop_scale_list '[0.5]'