method=2 # lsq : 0 pact : 1 Ternary : 2
gradient_scaling=None
NBITS_W=2

init_scaling=0.25
lr_scale=500

CUDA_VISIBLE_DEVICES=$1 python run.py --arch roberta_base --task $2 --lr_scale ${lr_scale} --clip_wd 0 --base_model $3 --teacher self --kd all --kd_num none \
--senqnn_config "{'quantize':True, 'nbits_w':${NBITS_W}, 'ffn_quantize': True, 'emb_quantize': True, 'qkv_quantize': True, 'classifier_dense': True, \
'method':${method}, 'gradient_scaling':${gradient_scaling}, 'init_scaling':${init_scaling}}"
