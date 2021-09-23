method=0 # lsq : 0 pact : 1
gradient_scaling=None
NBITS_W=2

# RTE
init_scaling=0.25
lr_scale=500
# CUDA_VISIBLE_DEVICES=1 python run.py --arch roberta_base --task RTE --lr_scale ${lr_scale} --clip_wd 0 --base_model rte --teacher self --kd output --kd_num none \
# --senqnn_config "{'quantize':True, 'nbits_w':${NBITS_W}, 'ffn_quantize': True, 'emb_quantize': True, 'qkv_quantize': True, \
# 'method':${method}, 'gradient_scaling':${gradient_scaling}, 'init_scaling':${init_scaling}}"

CUDA_VISIBLE_DEVICES=1 python run.py --arch roberta_base --task RTE --lr_scale ${lr_scale} --clip_wd 0 --base_model rte --teacher self --kd output --kd_num 0 \
--senqnn_config "{'quantize':True, 'nbits_w':${NBITS_W}, 'ffn_quantize': True, 'emb_quantize': True, 'qkv_quantize': True, \
'method':${method}, 'gradient_scaling':${gradient_scaling}, 'init_scaling':${init_scaling}}"

CUDA_VISIBLE_DEVICES=1 python run.py --arch roberta_base --task RTE --lr_scale ${lr_scale} --clip_wd 0 --base_model rte --teacher self --kd output --kd_num 1 \
--senqnn_config "{'quantize':True, 'nbits_w':${NBITS_W}, 'ffn_quantize': True, 'emb_quantize': True, 'qkv_quantize': True, \
'method':${method}, 'gradient_scaling':${gradient_scaling}, 'init_scaling':${init_scaling}}"

CUDA_VISIBLE_DEVICES=1 python run.py --arch roberta_base --task RTE --lr_scale ${lr_scale} --clip_wd 0 --base_model rte --teacher self --kd output --kd_num 2 \
--senqnn_config "{'quantize':True, 'nbits_w':${NBITS_W}, 'ffn_quantize': True, 'emb_quantize': True, 'qkv_quantize': True, \
'method':${method}, 'gradient_scaling':${gradient_scaling}, 'init_scaling':${init_scaling}}"

CUDA_VISIBLE_DEVICES=1 python run.py --arch roberta_base --task RTE --lr_scale ${lr_scale} --clip_wd 0 --base_model rte --teacher self --kd output --kd_num 3 \
--senqnn_config "{'quantize':True, 'nbits_w':${NBITS_W}, 'ffn_quantize': True, 'emb_quantize': True, 'qkv_quantize': True, \
'method':${method}, 'gradient_scaling':${gradient_scaling}, 'init_scaling':${init_scaling}}"

CUDA_VISIBLE_DEVICES=1 python run.py --arch roberta_base --task RTE --lr_scale ${lr_scale} --clip_wd 0 --base_model rte --teacher self --kd output --kd_num 4 \
--senqnn_config "{'quantize':True, 'nbits_w':${NBITS_W}, 'ffn_quantize': True, 'emb_quantize': True, 'qkv_quantize': True, \
'method':${method}, 'gradient_scaling':${gradient_scaling}, 'init_scaling':${init_scaling}}"

CUDA_VISIBLE_DEVICES=1 python run.py --arch roberta_base --task RTE --lr_scale ${lr_scale} --clip_wd 0 --base_model rte --teacher self --kd output --kd_num 5 \
--senqnn_config "{'quantize':True, 'nbits_w':${NBITS_W}, 'ffn_quantize': True, 'emb_quantize': True, 'qkv_quantize': True, \
'method':${method}, 'gradient_scaling':${gradient_scaling}, 'init_scaling':${init_scaling}}"

CUDA_VISIBLE_DEVICES=1 python run.py --arch roberta_base --task RTE --lr_scale ${lr_scale} --clip_wd 0 --base_model rte --teacher self --kd output --kd_num 6 \
--senqnn_config "{'quantize':True, 'nbits_w':${NBITS_W}, 'ffn_quantize': True, 'emb_quantize': True, 'qkv_quantize': True, \
'method':${method}, 'gradient_scaling':${gradient_scaling}, 'init_scaling':${init_scaling}}"

CUDA_VISIBLE_DEVICES=1 python run.py --arch roberta_base --task RTE --lr_scale ${lr_scale} --clip_wd 0 --base_model rte --teacher self --kd output --kd_num 7 \
--senqnn_config "{'quantize':True, 'nbits_w':${NBITS_W}, 'ffn_quantize': True, 'emb_quantize': True, 'qkv_quantize': True, \
'method':${method}, 'gradient_scaling':${gradient_scaling}, 'init_scaling':${init_scaling}}"

CUDA_VISIBLE_DEVICES=1 python run.py --arch roberta_base --task RTE --lr_scale ${lr_scale} --clip_wd 0 --base_model rte --teacher self --kd output --kd_num 8 \
--senqnn_config "{'quantize':True, 'nbits_w':${NBITS_W}, 'ffn_quantize': True, 'emb_quantize': True, 'qkv_quantize': True, \
'method':${method}, 'gradient_scaling':${gradient_scaling}, 'init_scaling':${init_scaling}}"

CUDA_VISIBLE_DEVICES=1 python run.py --arch roberta_base --task RTE --lr_scale ${lr_scale} --clip_wd 0 --base_model rte --teacher self --kd output --kd_num 8 \
--senqnn_config "{'quantize':True, 'nbits_w':${NBITS_W}, 'ffn_quantize': True, 'emb_quantize': True, 'qkv_quantize': True, \
'method':${method}, 'gradient_scaling':${gradient_scaling}, 'init_scaling':${init_scaling}}"

CUDA_VISIBLE_DEVICES=1 python run.py --arch roberta_base --task RTE --lr_scale ${lr_scale} --clip_wd 0 --base_model rte --teacher self --kd output --kd_num 9 \
--senqnn_config "{'quantize':True, 'nbits_w':${NBITS_W}, 'ffn_quantize': True, 'emb_quantize': True, 'qkv_quantize': True, \
'method':${method}, 'gradient_scaling':${gradient_scaling}, 'init_scaling':${init_scaling}}"

CUDA_VISIBLE_DEVICES=1 python run.py --arch roberta_base --task RTE --lr_scale ${lr_scale} --clip_wd 0 --base_model rte --teacher self --kd output --kd_num 10 \
--senqnn_config "{'quantize':True, 'nbits_w':${NBITS_W}, 'ffn_quantize': True, 'emb_quantize': True, 'qkv_quantize': True, \
'method':${method}, 'gradient_scaling':${gradient_scaling}, 'init_scaling':${init_scaling}}"

CUDA_VISIBLE_DEVICES=1 python run.py --arch roberta_base --task RTE --lr_scale ${lr_scale} --clip_wd 0 --base_model rte --teacher self --kd output --kd_num 11 \
--senqnn_config "{'quantize':True, 'nbits_w':${NBITS_W}, 'ffn_quantize': True, 'emb_quantize': True, 'qkv_quantize': True, \
'method':${method}, 'gradient_scaling':${gradient_scaling}, 'init_scaling':${init_scaling}}"

CUDA_VISIBLE_DEVICES=1 python run.py --arch roberta_base --task RTE --lr_scale ${lr_scale} --clip_wd 0 --base_model rte --teacher self --kd output --kd_num 12 \
--senqnn_config "{'quantize':True, 'nbits_w':${NBITS_W}, 'ffn_quantize': True, 'emb_quantize': True, 'qkv_quantize': True, \
'method':${method}, 'gradient_scaling':${gradient_scaling}, 'init_scaling':${init_scaling}}"

CUDA_VISIBLE_DEVICES=1 python run.py --arch roberta_base --task RTE --lr_scale ${lr_scale} --clip_wd 0 --base_model rte --teacher self --kd att --kd_num none \
--senqnn_config "{'quantize':True, 'nbits_w':${NBITS_W}, 'ffn_quantize': True, 'emb_quantize': True, 'qkv_quantize': True, \
'method':${method}, 'gradient_scaling':${gradient_scaling}, 'init_scaling':${init_scaling}}"

CUDA_VISIBLE_DEVICES=1 python run.py --arch roberta_base --task RTE --lr_scale ${lr_scale} --clip_wd 0 --base_model rte --teacher self --kd att --kd_num 0 \
--senqnn_config "{'quantize':True, 'nbits_w':${NBITS_W}, 'ffn_quantize': True, 'emb_quantize': True, 'qkv_quantize': True, \
'method':${method}, 'gradient_scaling':${gradient_scaling}, 'init_scaling':${init_scaling}}"

CUDA_VISIBLE_DEVICES=1 python run.py --arch roberta_base --task RTE --lr_scale ${lr_scale} --clip_wd 0 --base_model rte --teacher self --kd att --kd_num 1 \
--senqnn_config "{'quantize':True, 'nbits_w':${NBITS_W}, 'ffn_quantize': True, 'emb_quantize': True, 'qkv_quantize': True, \
'method':${method}, 'gradient_scaling':${gradient_scaling}, 'init_scaling':${init_scaling}}"

CUDA_VISIBLE_DEVICES=1 python run.py --arch roberta_base --task RTE --lr_scale ${lr_scale} --clip_wd 0 --base_model rte --teacher self --kd att --kd_num 2 \
--senqnn_config "{'quantize':True, 'nbits_w':${NBITS_W}, 'ffn_quantize': True, 'emb_quantize': True, 'qkv_quantize': True, \
'method':${method}, 'gradient_scaling':${gradient_scaling}, 'init_scaling':${init_scaling}}"

CUDA_VISIBLE_DEVICES=1 python run.py --arch roberta_base --task RTE --lr_scale ${lr_scale} --clip_wd 0 --base_model rte --teacher self --kd att --kd_num 3 \
--senqnn_config "{'quantize':True, 'nbits_w':${NBITS_W}, 'ffn_quantize': True, 'emb_quantize': True, 'qkv_quantize': True, \
'method':${method}, 'gradient_scaling':${gradient_scaling}, 'init_scaling':${init_scaling}}"

CUDA_VISIBLE_DEVICES=1 python run.py --arch roberta_base --task RTE --lr_scale ${lr_scale} --clip_wd 0 --base_model rte --teacher self --kd att --kd_num 4 \
--senqnn_config "{'quantize':True, 'nbits_w':${NBITS_W}, 'ffn_quantize': True, 'emb_quantize': True, 'qkv_quantize': True, \
'method':${method}, 'gradient_scaling':${gradient_scaling}, 'init_scaling':${init_scaling}}"

CUDA_VISIBLE_DEVICES=1 python run.py --arch roberta_base --task RTE --lr_scale ${lr_scale} --clip_wd 0 --base_model rte --teacher self --kd att --kd_num 5 \
--senqnn_config "{'quantize':True, 'nbits_w':${NBITS_W}, 'ffn_quantize': True, 'emb_quantize': True, 'qkv_quantize': True, \
'method':${method}, 'gradient_scaling':${gradient_scaling}, 'init_scaling':${init_scaling}}"

CUDA_VISIBLE_DEVICES=1 python run.py --arch roberta_base --task RTE --lr_scale ${lr_scale} --clip_wd 0 --base_model rte --teacher self --kd att --kd_num 6 \
--senqnn_config "{'quantize':True, 'nbits_w':${NBITS_W}, 'ffn_quantize': True, 'emb_quantize': True, 'qkv_quantize': True, \
'method':${method}, 'gradient_scaling':${gradient_scaling}, 'init_scaling':${init_scaling}}"

CUDA_VISIBLE_DEVICES=1 python run.py --arch roberta_base --task RTE --lr_scale ${lr_scale} --clip_wd 0 --base_model rte --teacher self --kd att --kd_num 7 \
--senqnn_config "{'quantize':True, 'nbits_w':${NBITS_W}, 'ffn_quantize': True, 'emb_quantize': True, 'qkv_quantize': True, \
'method':${method}, 'gradient_scaling':${gradient_scaling}, 'init_scaling':${init_scaling}}"

CUDA_VISIBLE_DEVICES=1 python run.py --arch roberta_base --task RTE --lr_scale ${lr_scale} --clip_wd 0 --base_model rte --teacher self --kd att --kd_num 8 \
--senqnn_config "{'quantize':True, 'nbits_w':${NBITS_W}, 'ffn_quantize': True, 'emb_quantize': True, 'qkv_quantize': True, \
'method':${method}, 'gradient_scaling':${gradient_scaling}, 'init_scaling':${init_scaling}}"

CUDA_VISIBLE_DEVICES=1 python run.py --arch roberta_base --task RTE --lr_scale ${lr_scale} --clip_wd 0 --base_model rte --teacher self --kd att --kd_num 8 \
--senqnn_config "{'quantize':True, 'nbits_w':${NBITS_W}, 'ffn_quantize': True, 'emb_quantize': True, 'qkv_quantize': True, \
'method':${method}, 'gradient_scaling':${gradient_scaling}, 'init_scaling':${init_scaling}}"

CUDA_VISIBLE_DEVICES=1 python run.py --arch roberta_base --task RTE --lr_scale ${lr_scale} --clip_wd 0 --base_model rte --teacher self --kd att --kd_num 9 \
--senqnn_config "{'quantize':True, 'nbits_w':${NBITS_W}, 'ffn_quantize': True, 'emb_quantize': True, 'qkv_quantize': True, \
'method':${method}, 'gradient_scaling':${gradient_scaling}, 'init_scaling':${init_scaling}}"

CUDA_VISIBLE_DEVICES=1 python run.py --arch roberta_base --task RTE --lr_scale ${lr_scale} --clip_wd 0 --base_model rte --teacher self --kd att --kd_num 10 \
--senqnn_config "{'quantize':True, 'nbits_w':${NBITS_W}, 'ffn_quantize': True, 'emb_quantize': True, 'qkv_quantize': True, \
'method':${method}, 'gradient_scaling':${gradient_scaling}, 'init_scaling':${init_scaling}}"

CUDA_VISIBLE_DEVICES=1 python run.py --arch roberta_base --task RTE --lr_scale ${lr_scale} --clip_wd 0 --base_model rte --teacher self --kd att --kd_num 11 \
--senqnn_config "{'quantize':True, 'nbits_w':${NBITS_W}, 'ffn_quantize': True, 'emb_quantize': True, 'qkv_quantize': True, \
'method':${method}, 'gradient_scaling':${gradient_scaling}, 'init_scaling':${init_scaling}}"
