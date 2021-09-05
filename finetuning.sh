NBITS_A=4
NBITS_W=4
act_clip_init_val=8.0
act_clip_init_valn=-8.0
weight_clip_init_val=1
ACT_CLIP_DECAY=0.001
WEIGHT_CLIP_DECAY=0.001

CUDA_VISIBLE_DEVICES=$2,$3 python run.py --arch roberta_base --task $1 --lr_scale 1000 --clip_wd 0.1 \
--senqnn_config "{'quantize': False, 'nbits_w':${NBITS_W}, 'nbits_a':${NBITS_A}, \
'act_clip_init_val': ${act_clip_init_val},'act_clip_init_valn': ${act_clip_init_valn}, \
'weight_clip_init_val': ${weight_clip_init_val}, \
'pact_decay_a':${ACT_CLIP_DECAY}, 'pact_decay_w':${WEIGHT_CLIP_DECAY}, 'emb_quantize':False}"
