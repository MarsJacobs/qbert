''' PROFIT Implementation '''
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import os
import datetime
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.backends.cudnn as cudnn
import argparse
import torchvision
import torchvision.transforms as transforms
'''
from my_lib.train_test import (
    resume_checkpoint,
    create_checkpoint,
    test,
    train_ts,
    CosineWithWarmup
)
'''
import pickle


from ast import literal_eval
parser = argparse.ArgumentParser()
parser.add_argument("--data", default="/SSD/ILSVRC2012")
parser.add_argument("--ckpt", required=True, help="checkpoint directory")

parser.add_argument("--quant_op", choices=["duq", "pact"])
parser.add_argument("--model", choices=["mobilenetv2", "mobilenetv3", "efficientnet", "cifar10_efficientnet", "cifar10_mobilenetv2"])
parser.add_argument("--teacher", choices=["none", "self", "resnet101"])

parser.add_argument("--lr", default=0.04, type=float)
parser.add_argument("--decay", default=2e-5, type=float) # 5e-4

parser.add_argument("--warmup", default=3, type=int)
parser.add_argument("--bn_epoch", default=5, type=int)
parser.add_argument("--ft_epoch", default=15, type=int)
parser.add_argument("--sample_epoch", default=5, type=int)

parser.add_argument("--use_ema", action="store_true", default=False)
parser.add_argument("--stabilize", action="store_true", default=False)

parser.add_argument("--w_bit", required=True, type=int, nargs="+")
parser.add_argument("--a_bit", required=True, type=int, nargs="+")
parser.add_argument("--w_profit", required=True, type=int, nargs="+")

parser.add_argument('--CGGrad', default=False, type=lambda x: (str(x).lower() == 'true'), help='using calibrated grad for clip_val')
parser.add_argument('--senqnn_config', default='',help='SENQNN: Quantiation config')
parser.add_argument('--folder', type=str, default='dir_not_specified', help='Folder to save checkpoints and log.')
parser.add_argument('--batch', required=True, type=int)
parser.add_argument('--tensorboard', action="store_true", default=False)
parser.add_argument('--neptune', action="store_true")
parser.add_argument('--is_test', action="store_true")
parser.add_argument('--dataset', default='imagenet', type=str)

args = parser.parse_args()
print(args)

# ================================================================================  # 
# Neptune & Tensorboard Setting
# ================================================================================  # 
import neptune.new as neptune

run = neptune.init(project='niceball0827/hwsw-2',
                    api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0YjM0ZTYwMi1kNjQwLTQ4NGYtOTYxMy03Mjc5ZmVkMzY2YTgifQ==')

#  import neptune

# neptune.init('niceball0827/hwsw')
# neptune.create_experiment(name='CIFAR10_PROFIT')
#neptune.create_experiment(name='EfficientNet')

event_root = args.ckpt + '/log_tensor'
if args.tensorboard:
    from torch.utils.tensorboard import SummaryWriter
    writer_param = SummaryWriter(log_dir = event_root, filename_suffix = '_param')
 
# ================================================================================  # 
# DATA LOADING
# ================================================================================  # 

ckpt_root = args.ckpt   
data_root = args.data 
use_cuda = torch.cuda.is_available()

print("==> Prepare data..")
if args.dataset == 'imagenet':
    print("==> IMAGENET")
    from my_lib.imagenet import get_loader
    testloader, trainloader, _ = get_loader(data_root, test_batch=args.batch, train_batch=args.batch)

if args.dataset == 'cifar10':
    data_root = 'cifar10_data'
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
    testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# ================================================================================  # 
# MODEL CREATING
# ================================================================================  # 


senqnn_setting = False
if args.senqnn_config is not '':
    senqnn_config = dict(**literal_eval(args.senqnn_config))
    senqnn_setting = True
assert senqnn_setting == True, "senqnn_config is not properly set. Possibly has to do with hypen or minus in the middle of seqnn and config "

from my_lib.common_funcs import *
from my_lib.khs_utils import load_model

if args.quant_op == "duq":
    from quant_op.duq import QuantOps
    from my_lib.duq_funcs import categorize_param, get_optimizer, set_n_lv, train_ts
    print("==> differentiable and unified quantization method is selected..")
elif args.quant_op == "pact":
    from quant_op.pact import QuantOps
    from my_lib.pact_funcs import categorize_param, get_optimizer, set_n_lv, train_ts
    print("==> Pact method is selected..")
else:
    raise NotImplementedError

print("==> Student model: %s" % args.model)
if args.model == "mobilenetv2":
    from conv_model.ilsvrc.MobileNetV2_quant import mobilenet_v2
    model = mobilenet_v2(QuantOps)
    model.load_state_dict(torch.load("./pretrained/mobilenet_v2-b0353104.pth"), False)
elif args.model == "mobilenetv3":
    from conv_model.ilsvrc.MobileNetV3Large_pad_quant import MobileNetV3Large
    model = MobileNetV3Large(QuantOps)
    model.load_state_dict(torch.load("./pretrained/mobilenet_v3_pad.pth"), False)
elif args.model == "efficientnet":
    from conv_model.ilsvrc.model import EfficientNet
    model = EfficientNet.from_name(model_name="efficientnet-b0", ops=QuantOps)
    pretrained = torch.load("./pretrained/efficientnet.pth")
    print("==> Load pth file")
    load_model(model, pretrained, False)
elif args.model == "cifar10_efficientnet":
    from conv_model.ilsvrc.EfficientNet_quant import QEfficientNetB0
    pretrained = torch.load("./pretrained/efficientnet_fp.pth")
    model = QEfficientNetB0(QuantOps)
    print("==> Load pth file")
    print(model.load_state_dict(pretrained, strict=False))
elif args.model == "cifar10_mobilenetv2":
    from conv_model.ilsvrc.MobileNetV2 import MobileNetV2
    pretrained = torch.load("./pretrained/mobilenetv2_fp.pth")
    model = MobileNetV2(QuantOps)
    print("==> Load pth file")
    print(model.load_state_dict(pretrained, strict=False))
else:
    raise NotImplementedError


print("==> Teacher model: %s" % args.teacher)
if args.teacher == "none":
    model_t = None
elif args.teacher == "self":    
    model_t = copy.deepcopy(model)
elif args.teacher == "resnet101":
    from torchvision.models.resnet import resnet101
    model_t = resnet101(True)
else:
    raise NotImplementedError

if model_t is not None:
    for params in model_t.parameters():
        params.requires_grad = False

modle_ema = copy.deepcopy(model)
# ================================================================================  # 
# CUDA
# ================================================================================  # 

if use_cuda:
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    
    if model_t is not None:
        model_t.cuda()
        model_t = torch.nn.DataParallel(model_t, device_ids=range(torch.cuda.device_count()))

    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

# ================================================================================  # 
# Logging
# ================================================================================  # 

def make_log(model, abs_epoch, train_acc, acc_base, acc_ema, optimizer):

    if args.tensorboard:
        # writer_acc.add_scalar('train acc', train_acc, abs_epoch)
        # writer_acc.add_scalar('test acc', acc_base[0], abs_epoch)
        # writer_acc.add_scalar('lr', optimizer.param_groups[-1]['lr'], abs_epoch)
        
        # if args.quant_op == 'pact':
        #     for name, param in model.module.named_parameters():
        #         if 'clip_val' in name:
        #             writer_param.add_scalar(name, param, abs_epoch)
        # if args.quant_op == 'duq':
        #         if '.a' in name or '.c' in name and not 'conv' in name:
        #             writer_param.add_scalar(name, param, abs_epoch)
        
        if abs_epoch % 1 == 0:
            for name , param in model.module.named_parameters():
               if 'weight' in name and not 'clip_val' in name:
                   writer_param.add_histogram(name, param.clone().cpu().data.numpy(), abs_epoch)
                # if args.quant_op == 'pact':
                #    if('clip_val') in name:
                #     writer_param.add_scalar(name, param, abs_epoch)
                # if args.quant_op == 'duq':
                #     if '.a' in name or '.c' in name and not 'conv' in name:
                #         writer_param.add_scalar(name, param, abs_epoch)
                 
            # i=0
            # for module in model.modules():
            #     if 'Q_Conv2d' in module.__class__.__name__:
            #         writer_param.add_histogram(str(module.__class__.__name__)+"_"+str(i)+'_conv_input', module.conv_input, abs_epoch)
            #         #writer_param.add_histogram(str(module.__class__.__name__)+"_"+str(i)+'_qweight', module.qweight, abs_epoch)
            #     if 'Q_ReLU6' in module.__class__.__name__:
            #         writer_param.add_histogram(str(module)+str(i)+'input', module.relu_input, abs_epoch)
            #         writer_param.add_histogram(str(module)+str(i)+'output', module.relu_output, abs_epoch)
            #     if 'Q_Swish' in module.__class__.__name__:
            #         writer_param.add_histogram(str(module.__class__.__name__)+"_"+str(i)+'_input', module.swish_input, abs_epoch)
            #         writer_param.add_histogram(str(module.__class__.__name__)+"_"+str(i)+'_output', module.swish_output, abs_epoch)
                # i += 1
            

                    
    if args.neptune:
        

        # for name, param in model.module.named_parameters():
        #     if 'conv' in name and not 'clip' in name:
        #         print("metrics/model/" + name)
        #         run["metrics/model/" + name].log(param)
                   
        # neptune.log_metric('epoch', abs_epoch)
        # neptune.log_metric('train acc', train_acc)
        # neptune.log_metric('test acc', acc_base[0])
        # neptune.log_metric('ema test acc', acc_ema[0])
        # neptune.log_metric('loss', acc_base[1])
        # neptune.log_metric('lr', optimizer.param_groups[-1]['lr'])
        
        # if args.quant_op == 'pact':
        #         for name, param in model.module.named_parameters():
        #             if 'clip_val' in name:
        #                 neptune.log_metric(name, param)
        run["train"] = {'lr' : optimizer.param_groups[-1]['lr'],
                         'train_acc' : train_acc,
                         'test_acc' : acc_base[0],
                         'loss' : acc_base[1],
                         }
        run["metrics/train/accuracy"].log(train_acc)
        run["metrics/train/loss"].log(acc_base[1])
        run["metrics/test/accuracy"].log(acc_base[0])
        run["metrics/train/lr"].log(optimizer.param_groups[-1]['lr'])

        for name, param in model.module.named_parameters():
            if 'clip_val' in name:
                run["metrics/model/" + name].log(param)

    

            
    
# ================================================================================  # 
# Train per Epoch
# ================================================================================  # 

# print("==> Pretrained Accuracy is...")
# test(testloader, model, criterion, 0)
 
abs_epoch = 0
def train_epochs(optimizer, warmup_len, max_epochs, prefix):
    model_ema = copy.deepcopy(model)

    last_epoch, best_acc = resume_checkpoint(model, model_ema, optimizer, args.ckpt, prefix)
    
    scheduler = CosineWithWarmup(optimizer, 
                        warmup_len=warmup_len, warmup_start_multiplier=0.1,
                        max_epochs=max_epochs, eta_min=1e-3, last_epoch=last_epoch)
    
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    global abs_epoch
    for epoch in range(last_epoch+1, max_epochs):
        abs_epoch += 1
        train_acc = train_ts(trainloader, model, model_ema, model_t, criterion, optimizer, epoch, test=args.is_test)
        acc_base = test(testloader, model, criterion, epoch, test=args.is_test)
        
        acc_ema = (0,0)        
    
        is_best = False
        if acc_base[0] > best_acc:
           is_best = True
     
        is_ema_best = False
        if acc_ema[0] > best_acc:
           is_ema_best =  True


        make_log(model, abs_epoch, train_acc, acc_base, acc_ema, optimizer) # logging!
        

        best_acc = max(best_acc, acc_base[0], acc_ema[0])        
        # create_checkpoint(model, model_ema, optimizer,
        #                   is_best, is_ema_best, best_acc, epoch, ckpt_root, 1, prefix)    
        scheduler.step()  
    return best_acc

# ================================================================================  # 
# TRAINING START
# ================================================================================  # 
a_bit, w_bit = 32, 32
if args.teacher != "none": # full-precision fine-tuning with teacher-student
    
    prefix =  phase_prefix(args.model, args.teacher, args.quant_op, args.use_ema, a_bit, w_bit)

    print("==> Full precision fine-tuning")
    params = categorize_param(model)
    optimizer = get_optimizer(params, train_quant=True, train_weight=True, train_bnbias=True, w_decay = args.decay, lr = args.lr,  w_clip_decay = senqnn_config['w_clip_decay'], a_clip_decay =  senqnn_config['act_clip_decay'])    
    train_epochs(optimizer, args.warmup, args.ft_epoch, prefix)

# ================================================================================  # 
# Quant Initialize
# ================================================================================  # 

print("==> Quantizaion PACT initialize...")
QuantOps.initialize(model, args.a_bit[0], args.w_bit[0], act=True, weight = True, # Weight = True only non scheduled QAT
                    clip_init_val = senqnn_config['act_clip_init_val'],
                    clip_init_valn = senqnn_config['act_clip_init_valn'],
                    weight_init_val = senqnn_config['weight_clip_init_val'])

# ================================================================================  # 
# ACTIVATION QUANTIZATION
# ================================================================================  # 
print("==> Activation Quantization")
for a_bit in args.a_bit:
    if a_bit == 32:
        break
    prefix =  phase_prefix(args.model, args.teacher, args.quant_op, args.use_ema, a_bit, w_bit)
    print("==> Activation quantization, bit %d" % a_bit)
    if args.quant_op == 'pact':
        for name, module in model.named_modules():
            if isinstance(module, (QuantOps.ReLU, QuantOps.ReLU6, QuantOps.Swish)):
                module.num_bit = a_bit
            # if isinstance(module,(QuantOps.Conv2d, QuantOps.Linear)):  # temp
            #     module.num_bit = a_bit
    
    if args.stabilize:
        print("==> BN stabilize")
        params = categorize_param(model)
        optimizer = get_optimizer(params, train_quant=True, train_weight=False, train_bnbias=True, w_decay = args.decay, lr = args.lr,  w_clip_decay = senqnn_config['w_clip_decay'], a_clip_decay =  senqnn_config['act_clip_decay'])    
        train_epochs(optimizer, 0, args.bn_epoch, prefix + "_bn")

    print("==> Fine-tuning")
    params = categorize_param(model)
    optimizer = get_optimizer(params, train_quant=True, train_weight=True, train_bnbias=True, w_decay = args.decay, lr = args.lr,  w_clip_decay = senqnn_config['w_clip_decay'], a_clip_decay =  senqnn_config['act_clip_decay'])    
    train_epochs(optimizer, args.warmup, args.ft_epoch, prefix)

    if args.stabilize:
        print("==> BN stabilize 2")
        optimizer = get_optimizer(params, train_quant=True, train_weight=False, train_bnbias=True, w_decay = args.decay, lr = args.lr,  w_clip_decay = senqnn_config['w_clip_decay'], a_clip_decay =  senqnn_config['act_clip_decay'])    
        train_epochs(optimizer, 0, args.bn_epoch, prefix + "_bn2")

# ================================================================================  # 
# WEIGHT QUANTIZATION
# ================================================================================  # 
print("==> Weight Quantization")
for w_bit in args.w_bit:
    if w_bit == 32:
        break
    prefix =phase_prefix(args.model, args.teacher, args.quant_op, args.use_ema, a_bit, w_bit)
    print("==> Weight quantization, bit %d" % w_bit)
    for name, module in model.named_modules():
        if isinstance(module,(QuantOps.Conv2d, QuantOps.Linear)): 
                module.num_bit = w_bit

    if args.stabilize and not w_bit == 4:
        print("==> BN stabilize")
        params = categorize_param(model)
        optimizer = get_optimizer(params, train_quant=True, train_weight=False, train_bnbias=True, w_decay = args.decay, lr = args.lr,  w_clip_decay = senqnn_config['w_clip_decay'], a_clip_decay =  senqnn_config['act_clip_decay'])    
        train_epochs(optimizer, 0, args.bn_epoch, prefix + "_bn")
    # ================================================================================  # 
    # PROFIT TRAINING
    # ================================================================================  # 
    if w_bit in args.w_profit:
        print("==> Sampling")
        metric_map = {}
        for name, module in model.module.named_modules():
            if hasattr(module, "_weight_quant") and isinstance(module, nn.Conv2d):
                metric_map[name] = 0

        if os.path.exists(os.path.join(args.ckpt, prefix + ".pkl")):
            print("==> Load existed sampled map")
            with open(os.path.join(args.ckpt, prefix + ".pkl"), "rb") as f:
                metric_map = pickle.load(f)

        else:
            optimizer = get_optimizer(params, train_quant=True, train_weight=True, train_bnbias=True, w_decay = args.decay, lr = args.lr,  w_clip_decay = senqnn_config['w_clip_decay'], a_clip_decay =  senqnn_config['act_clip_decay']) 
            for epoch in range(args.sample_epoch):      
                model_ema = copy.deepcopy(model)  
                train_acc = train_ts(trainloader, model, model_ema, model_t, criterion, optimizer, epoch, metric_map, test=args.is_test)  
                acc_base = test(testloader, model, criterion, epoch,train = False ,test=args.is_test)
                abs_epoch += 1
                acc_ema = (0,0)
                
                make_log(model, abs_epoch, train_acc, acc_base, acc_ema, optimizer) # logging!
                
            with open(os.path.join(args.ckpt, prefix + ".pkl"), "wb") as f:
                pickle.dump(metric_map, f)  
        
        skip_list = []
        import operator
        sort = sorted(metric_map.items(), key=operator.itemgetter(1), reverse=True)

        print("sorted AIWQ metrics")
        for i in range(len(sort)):
            print('%d.  %s : %.10f' %(i, sort[i][0], sort[i][1]) )
        
        for s in sort[0:int(len(sort) * 1/3)]:
            #if 'se1' in s[0] or 'se2' in s[0] or 'conv2' in s[0]:
            skip_list.append(s[0])
        
        print(len(skip_list))
        print(skip_list)

        skip_list_next = []
        for s in sort[int(len(sort) * 1/3):int(len(sort) * 2/3)]:
            #if 'se1' in s[0] or 'se2' in s[0] or 'conv2' in s[0]:
            skip_list_next.append(s[0])
       
        print("==> PROFIT 1")
        params = categorize_param(model)
        optimizer = get_optimizer(params, train_quant=True, train_weight=True, train_bnbias=True, w_decay = args.decay, lr = args.lr,  w_clip_decay = senqnn_config['w_clip_decay'], a_clip_decay =  senqnn_config['act_clip_decay'])
        train_epochs(optimizer, args.warmup, args.ft_epoch, prefix + "_ft1")

        print("==> PROFIT 2")
        params = categorize_param(model, skip_list)
        optimizer = get_optimizer(params, train_quant=True, train_weight=True, train_bnbias=True, w_decay = args.decay, lr = args.lr,  w_clip_decay = senqnn_config['w_clip_decay'], a_clip_decay =  senqnn_config['act_clip_decay'])
        train_epochs(optimizer, args.warmup, args.ft_epoch, prefix + "_ft2")

        # ================================================================================  # 
        # RESAMPLING
        # ================================================================================  # 
        print("==> ReSampling")
        metric_map = {}
        for name, module in model.module.named_modules():
            if hasattr(module, "_weight_quant") and isinstance(module, nn.Conv2d):
                metric_map[name] = 0

        optimizer = get_optimizer(params, train_quant=True, train_weight=True, train_bnbias=True, w_decay = args.decay, lr = args.lr,  w_clip_decay = senqnn_config['w_clip_decay'], a_clip_decay =  senqnn_config['act_clip_decay']) 
        for epoch in range(args.sample_epoch):      
            model_ema = copy.deepcopy(model)  
            train_acc = train_ts(trainloader, model, model_ema, model_t, criterion, optimizer, epoch, metric_map, test=args.is_test)  
            acc_base = test(testloader, model, criterion, epoch,train = False ,test=args.is_test)
            abs_epoch += 1
            acc_ema = (0,0)
        
        skip_list = []
        import operator
        sort = sorted(metric_map.items(), key=operator.itemgetter(1), reverse=True)

        print("sorted AIWQ metrics")
        for i in range(len(sort)):
            print('%d.  %s : %.10f' %(i, sort[i][0], sort[i][1]) )

        # ================================================================================  # 
        # RESAMPLING
        # ================================================================================  # 

        print("==> PROFIT 3") 
        params = categorize_param(model, skip_list)
        optimizer = get_optimizer(params, train_quant=True, train_weight=True, train_bnbias=True, w_decay = args.decay, lr = args.lr,  w_clip_decay = senqnn_config['w_clip_decay'], a_clip_decay =  senqnn_config['act_clip_decay'])
        train_epochs(optimizer, args.warmup, args.ft_epoch, prefix + "_ft3")
        
        # params = categorize_param(model)
        # print("==> PROFIT 1")
        # optimizer = get_optimizer(params, train_quant=True, train_weight=True, train_bnbias=True, w_decay = args.decay, lr = args.lr,  w_clip_decay = senqnn_config['w_clip_decay'], a_clip_decay =  senqnn_config['act_clip_decay'])
        # train_epochs(optimizer, args.warmup, args.ft_epoch, prefix + "_ft1")

        # print("==> PROFIT 2")
        # params = categorize_param(model, skip_list)
        # optimizer = get_optimizer(params, train_quant=True, train_weight=True, train_bnbias=True, w_decay = args.decay, lr = args.lr,  w_clip_decay = senqnn_config['w_clip_decay'], a_clip_decay =  senqnn_config['act_clip_decay'])
        # train_epochs(optimizer, args.warmup, args.ft_epoch, prefix + "_ft2")

        # print("==> PROFIT 3")
        # params = categorize_param(model, skip_list)
        # optimizer = get_optimizer(params, train_quant=True, train_weight=True, train_bnbias=True, w_decay = args.decay, lr = args.lr,  w_clip_decay = senqnn_config['w_clip_decay'], a_clip_decay =  senqnn_config['act_clip_decay'])
        # train_epochs(optimizer, args.warmup, args.ft_epoch, prefix + "_ft3")

    # ================================================================================  # 
    # PROFIT Training
    # ================================================================================  # 

    else:                     
        print("==> Fine-tuning")
        params = categorize_param(model)
        optimizer = get_optimizer(params, train_quant=True, train_weight=True, train_bnbias=True, w_decay = args.decay, lr = args.lr,  w_clip_decay = senqnn_config['w_clip_decay'], a_clip_decay =  senqnn_config['act_clip_decay'])
        train_epochs(optimizer, args.warmup, args.ft_epoch, prefix)

    if args.stabilize and not w_bit == 4:
        print("==> BN stabilize 2")
        optimizer = get_optimizer(params, train_quant=True, train_weight=False, train_bnbias=True, w_decay = args.decay, lr = args.lr,  w_clip_decay = senqnn_config['w_clip_decay'], a_clip_decay =  senqnn_config['act_clip_decay'])
        best_acc = train_epochs(optimizer, 0, args.bn_epoch, prefix + "_bn2")

print("==> Finish training.. last best accuracy is {}".format(best_acc))
