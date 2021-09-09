NBITS_A=4
NBITS_W=2
act_clip_init_val=8.0
act_clip_init_valn=-8.0
weight_clip_init_val=1
ACT_CLIP_DECAY=0.001
WEIGHT_CLIP_DECAY=0.001
method=0 # lsq : 0 pact : 1
gradient_scaling=True

CUDA_VISIBLE_DEVICES=$1 python run.py --arch roberta_base --task CoLA --lr_scale 1000 --clip_wd 0.5 --base_model cola --teacher self --kd $2 \
--senqnn_config "{'quantize':True, 'nbits_w':${NBITS_W}, 'nbits_a':${NBITS_A}, \
'act_clip_init_val': ${act_clip_init_val},'act_clip_init_valn': ${act_clip_init_valn}, \
'weight_clip_init_val': ${weight_clip_init_val}, \
'pact_decay_a':${ACT_CLIP_DECAY}, 'pact_decay_w':${WEIGHT_CLIP_DECAY}, 'ffn_quantize': True, 'emb_quantize':False, 'qkv_quantize': False, \
'method':${method}, 'gradient_scaling':${gradient_scaling}}"
