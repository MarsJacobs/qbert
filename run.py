import os
import subprocess
import argparse
from time import gmtime, strftime

def make_dir(args, is_large, lr):
    root = args.output_dir
    no_save = args.no_save
    task_dir = args.task + '-' + ('large' if is_large else 'base')
    hyperparam_dir = 'wd%s_ad%s_d%s_lr%s' % (str(args.weight_decay), 
        str(args.attn_dropout), str(args.dropout),  str(lr))

    time = strftime("%m%d-%H%M%S", gmtime())
    log_name = '%s.log' % time
    ckpt_name = '%s_ckpt' % time

    log_dir = os.path.join(root, 'none')
    log_dir = os.path.join(log_dir, task_dir)
    log_dir = os.path.join(log_dir, hyperparam_dir)

    log_file = os.path.join(log_dir, log_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not no_save:
        ckpt_dir = os.path.join(log_dir, ckpt_name)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
    else:
        ckpt_dir = log_dir # dummy directory

    return log_file, ckpt_dir


def arg_parse():
    parser = argparse.ArgumentParser(
        description='This repository contains the PyTorch implementation for the paper ZeroQ: A Novel Zero-Shot Quantization Framework.')

    # hyperparameters
    parser.add_argument('--attn-dropout', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--max-epochs', type=int, default=None)
    parser.add_argument('--bs', type=float, default=None, help='batch size')

    parser.add_argument('--arch', type=str, default='roberta_base',
                        choices=['roberta_base', 'roberta_large', ],
                        help='model architecture')
    parser.add_argument('--task', type=str,
                        choices=['RTE', 'SST-2', 'MNLI', 'QNLI',
                                 'CoLA', 'QQP', 'MRPC', 'STS-B',],
                        help='finetuning task')

    parser.add_argument('--model-dir', type=str, default='models',
                        help='model directory')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='folder name to store logs and checkpoints')
    parser.add_argument('--restore-file', type=str, default=None,
                        help='finetuning from the given checkpoint')
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--base_model', type=str, default='roberta')
    parser.add_argument('--senqnn_config', default='', help='SENQNN: Quantization Config')
    parser.add_argument('--lr_scale', type=int, default=1)
    parser.add_argument('--clip_wd', type=float, default=0.1)
    parser.add_argument('--teacher', type=str, default="none")
    parser.add_argument('--kd', type=str, default="all")
    parser.add_argument('--kd_num', default="none")
    
    args = parser.parse_args()
    return args

args = arg_parse()
task = args.task
senqnn_config = args.senqnn_config
lr_scale = args.lr_scale
clip_wd = args.clip_wd
teacher = args.teacher
kd = args.kd
kd_num = args.kd_num

######################## Task specs ##########################

task_specs = {
    'RTE' : {
        'dataset': 'RTE-bin',
        'num_classes': '2',
        'lr': '2e-5', # 2e-5
        'max_sentences': '16', # 4 # 32
        'total_num_updates': '2036',
        'warm_updates': '122',
    },
    'SST-2' : {
        'dataset': 'SST-2-bin',
        'num_classes': '2',
        'lr': '1e-5', # 1e-6 -> 1e-05
        'max_sentences': '32',
        'total_num_updates': '20935',
        'warm_updates': '1256'
    },
    'MNLI' : {
        'dataset': 'MNLI-bin',
        'num_classes': '3',
        'lr': '1e-5',
        'max_sentences': '32',
        'total_num_updates': '123873',
        'warm_updates': '7432',
        'valid_interval_sentences': '100000',
    },
    'QNLI' : {
        'dataset': 'QNLI-bin',
        'num_classes': '2',
        'lr': '1e-5',
        'max_sentences': '32',
        'total_num_updates': '33112',
        'warm_updates': '1986',
        'valid_interval_sentences': '55000',
    },
    'CoLA' : {
        'dataset': 'CoLA-bin',
        'num_classes': '2',
        'lr': '2e-5', # 1e-05 -> 2e-05 Reproduced
        'max_sentences': '16',
        'total_num_updates': '5336',
        'warm_updates': '320'
    },
    'QQP' : {
        'dataset': 'QQP-bin',
        'num_classes': '2',
        'lr': '1.5e-5',
        'max_sentences': '32',
        'total_num_updates': '113272',
        'warm_updates': '28318',
        'valid_interval_sentences': '950000',
    },
    'MRPC' : {
        'dataset': 'MRPC-bin',
        'num_classes': '2',
        'lr': '1e-5', # 1e-05
        'max_sentences': '16', # 16 
        'total_num_updates': '2296',
        'warm_updates': '137'
    },
    'STS-B' : {
        'dataset': 'STS-B-bin',
        'num_classes': '1',
        'lr': '2e-5',
        'max_sentences': '16',
        'total_num_updates': '3598',
        'warm_updates': '214'
    },
}


is_large = 'large' in args.arch
spec = task_specs[task]
dataset = '%s-bin' % task
num_classes = spec['num_classes']
total_num_updates = spec['total_num_updates']
warm_updates = spec['warm_updates']
max_epochs = '6' if task in ['MNLI', 'QQP'] else '12'

lr = str(args.lr) if args.lr else spec['lr'] 
bs = str(args.bs) if args.bs else spec['max_sentences']

#mskim checkpoint path
log_file, ckpt_dir = make_dir(args, is_large, lr)
#model_path = args.model_dir  + '/roberta.large/model.pt' if is_large \
#        else args.model_dir + '/roberta.base/model.pt'

if args.base_model == "roberta":
    model_path = args.model_dir + '/roberta.base/model.pt'    
elif args.base_model == "sst":
    model_path = args.model_dir + '/SST_FP.pt'
elif args.base_model == "cola":
    model_path = args.model_dir + '/CoLA_FP.pt'
elif args.base_model == "rte":
    model_path = args.model_dir + '/RTE_FP.pt'
else:
    print("No Pretrained File Exists!")

valid_subset = 'valid' if task != 'MNLI' else 'valid,valid1'

print('valid_subset:',valid_subset)

#============================ metric  ============================#
if args.task in ["SST-2", "RTE", "QNLI", "MNLI"]:
    best_metric = "accuracy"
if args.task in ["MRPC", "QQP"]:
    best_metric = "f1"
if args.task in ["STS-B"]:
    best_metric = "corr"
if args.task in ["CoLA"]:
    best_metric = "mcc"

###############################################################

subprocess_args = [
    'fairseq-train', dataset,
    '--restore-file', model_path,
    '--valid-subset', valid_subset,
    '--max-positions', '512',
    '--max-sentences', bs,
    '--max-tokens', '4400',
    '--task', 'sentence_prediction',
    '--criterion', 'sentence_prediction',
    '--reset-optimizer',  '--reset-dataloader', '--reset-meters',
    '--required-batch-size-multiple',  '1',
    '--init-token', '0', '--separator-token', '2',
    '--arch',  args.arch,
    '--num-classes', num_classes,
    '--weight-decay', str(args.weight_decay), 
    '--optimizer', 'adam', '--adam-betas', '(0.9, 0.98)', '--adam-eps', '1e-06',
    '--clip-norm',  '0.0',
    '--lr-scheduler',  'polynomial_decay', '--lr', lr,
    '--total-num-update', total_num_updates, '--warmup-updates', warm_updates,
    '--max-epoch',  max_epochs,
    '--find-unused-parameters',  
    '--best-checkpoint-metric', best_metric, 
    '--save-dir', ckpt_dir, 
    '--log-file', log_file,
    '--dropout', str(args.dropout), '--attention-dropout', str(args.attn_dropout),
    '--senqnn_config', senqnn_config,
    '--lr_scale', str(lr_scale),
    '--clip_wd', str(clip_wd),
    '--teacher', teacher,
    '--kd', kd,
    '--kd_num', str(kd_num),
    '--task_glue', task
]

if args.no_save:
    subprocess_args += ['--no-save']

if args.task == 'STS-B':
    subprocess_args += ['--regression-target', '--best-checkpoint-metric', 'corr']
else:
    subprocess_args.append('--maximize-best-checkpoint-metric')

subprocess.call(subprocess_args)
