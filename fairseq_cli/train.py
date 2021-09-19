#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import argparse
import logging
import math
import random
import sys

import numpy as np
import torch
from fairseq import (
    checkpoint_utils,
    distributed_utils,
    options,
    quantization_utils,
    tasks,
    utils,
) 
from fairseq.data import iterators
from fairseq.logging import meters, metrics, progress_bar
from fairseq.model_parallel.megatron_trainer import MegatronTrainer
from fairseq.trainer import Trainer

from pact import QuantOps # MSKIM Import PACT Quantizer
from ast import literal_eval

from scipy.stats import pearsonr, spearmanr

from ast import literal_eval

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.train")

# ========================================================================= #
# MSKIM Logging
# ========================================================================= #


def make_log(run, model, epoch, loss_func, accuracy, writer_param, loss):

    for module in model.named_modules():
        if 'Q_Linear' in module[1].__class__.__name__ or 'Q_Embedding' in module[1].__class__.__name__ or 'custom_linear' in module[1].__class__.__name__ or 'custom_embedding' in module[1].__class__.__name__:
            writer_param.add_histogram(module[0][25:] + '_weight', module[1].weight_buffer.avg, epoch)
            writer_param.add_histogram(module[0][25:] + '_qweight', module[1].qweight_buffer.avg, epoch)
            #import pdb; pdb.set_trace()
            if epoch > 0:
                run["metrics/Weight_max/" + module[0][25:]].log(module[1].weight_buffer.avg.max())
                run["metrics/Weight_min/" + module[0][25:]].log(module[1].weight_buffer.avg.min())
                run["metrics/Weight_mean/" + module[0][25:]].log(module[1].weight_buffer.avg.abs().mean())

                run["metrics/Weight_max/" + module[0][25:] + '_Q'].log(module[1].qweight_buffer.avg.max())
                run["metrics/Weight_min/" + module[0][25:] + '_Q'].log(module[1].qweight_buffer.avg.min())
                run["metrics/Weight_mean/" + module[0][25:] + '_Q'].log(module[1].qweight_buffer.avg.abs().mean())
                
                run["metrics/Weight_Loss/" + module[0][25:]].log(loss_func(module[1].weight_buffer.avg, module[1].qweight_buffer.avg).item())
            
            module[1].weight_buffer.reset()
            module[1].qweight_buffer.reset()
            

    # for name, param in model.named_parameters():
    #     # if 'weight' in name and not 'clip_val' in name:
    #     #     writer_param.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
    #     if 'clip_val' in name:        
    #         run["metrics/ClipVal/" + name[17:]].log(param)
        
        #run["metrics/lr"].log(trainer._optimizer.param_groups[0]['lr'])
        #run["metrics/Quant/lr"].log(trainer._optimizer.param_groups[-1]['lr'])
    run["metrics/ACC"].log(accuracy)
    loss = loss
    run["metrics/loss"].log(loss)

def make_filename(quant_options, run):

    if quant_options['quantize'] and quant_options['method'] == 0:
        bits = quant_options['nbits_w']
        ffn = str(quant_options['ffn_quantize'])
        qkv = str(quant_options['qkv_quantize'])
        emb = str(quant_options['emb_quantize'])

        file_name = run.get_run_url()[47:] + '_' + str(bits) + '_' + 'F_' + ffn + '_' + 'Q_' + qkv + '_' + 'E_' + emb 
    else:
        file_name = run.get_run_url()[47:] + '_' + "Full_Precision"
    return file_name



def main(args):
    loss_func = torch.nn.L1Loss()
    import neptune.new as neptune
    run = neptune.init(project='niceball0827/QRoberta',
                   api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0YjM0ZTYwMi1kNjQwLTQ4NGYtOTYxMy03Mjc5ZmVkMzY2YTgifQ==')
    
    utils.import_user_module(args)
    assert (
        args.max_tokens is not None or args.max_sentences is not None
    ), "Must specify batch size either with --max-tokens or --max-sentences"

    from torch.utils.tensorboard import SummaryWriter
    
    quant_options = dict(**literal_eval(args.senqnn_config))
    suffix = make_filename(quant_options, run)
    
    writer_param = SummaryWriter(log_dir = "output_tensorboard/" + args.data + "/", filename_suffix = suffix)
    
    metrics.reset()

    np.random.seed(args.seed)
    utils.set_torch_seed(args.seed)

    if distributed_utils.is_master(args):
        checkpoint_utils.verify_checkpoint_directory(args.save_dir)
    
    # Print args
    logger.info(args)
    
    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)
    
    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in args.valid_subset.split(","):
        task.load_dataset(valid_sub_split, combine=False, epoch=1)

    # Build model and criterion
    dict_senqnn_config= dict(**literal_eval(args.senqnn_config))

    if args.teacher == "none":
        senqnn_config = dict_senqnn_config
        model = task.build_model(args, QuantOps)
        model_t = None

    elif args.teacher == "self":
        senqnn_config = dict_senqnn_config
        model = task.build_model(args, QuantOps)
        
        dict_senqnn_config['quantize'] = False
        args.senqnn_config = str(dict_senqnn_config)
        model_t = task.build_model(args, QuantOps) # Teacher Model 
    
    criterion = task.build_criterion(args)

    # PACT Quantization Initialization
    if senqnn_config['method'] == 1:
        for name, module in model.named_modules():
            if isinstance(module, (QuantOps.QLinear, QuantOps.QEmbedding)):
                module.initialize(senqnn_config['nbits_w'], senqnn_config['weight_clip_init_val'], -1*senqnn_config['weight_clip_init_val'])

    # Init Value Logging 
    make_log(run, model, 0, loss_func, 0, writer_param,  0)

    #logger.info(model)
    logger.info("task: {} ({})".format(args.task, task.__class__.__name__))
    logger.info("model: {} ({})".format(args.arch, model.__class__.__name__))
    logger.info(
        "criterion: {} ({})".format(args.criterion, criterion.__class__.__name__)
    )
    logger.info(
        "num. model params: {} (num. trained: {})".format(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
    )

    #args.quantization_config_path = "examples/quant_noise/transformer_quantization_config.yaml"
    # (optionally) Configure quantization

    if args.quantization_config_path is not None:
        quantizer = quantization_utils.Quantizer(
            config_path=args.quantization_config_path,
            max_epoch=args.max_epoch,
            max_update=args.max_update,
        )
    else:
        quantizer = None

    # Build trainer
    if args.model_parallel_size == 1:
        trainer = Trainer(args, task, model, model_t, criterion, quantizer)
    else:
        trainer = MegatronTrainer(args, task, model, criterion)

    logger.info(
        "training on {} devices (GPUs/TPUs)".format(args.distributed_world_size)
    )
    logger.info(
        "max tokens per GPU = {} and max sentences per GPU = {}".format(
            args.max_tokens, args.max_sentences
        )
    )

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)

    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    lr = trainer.get_lr()
    train_meter = meters.StopwatchMeter()
    train_meter.start()

    # ========================================================================= #
    # Clip Value LR Scheduler
    # ========================================================================= #
    
    while lr > args.min_lr and epoch_itr.next_epoch_idx <= max_epoch:
        
        valid_losses, should_stop, epoch_losses = train(args, trainer, task, epoch_itr, run)

        # Make Log MSKIM
        accuracy = valid_losses
        
        make_log(run, model, epoch_itr.next_epoch_idx -1, loss_func, accuracy, writer_param, epoch_losses)

        if should_stop:
            break
        
        # only use first validation loss to update the learning rate
        # MSKIM lr step
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        epoch_itr = trainer.get_train_iterator(
            epoch_itr.next_epoch_idx,
            # sharded data: get train iterator for next epoch
            load_dataset=task.has_sharded_data("train"),
        )
    train_meter.stop()
    logger.info("done training in {:.1f} seconds".format(train_meter.sum))


def should_stop_early(args, valid_loss):
    # skip check if no validation was done in the current epoch
    if valid_loss is None:
        return False
    if args.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if args.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, "best", None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= args.patience:
            logger.info(
                "early stop since valid performance hasn't improved for last {} runs".format(
                    args.patience
                )
            )
            return True
        else:
            return False


@metrics.aggregate("train")
def train(args, trainer, task, epoch_itr, run):
    """Train the model for one epoch and return validation losses."""
    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=(epoch_itr.next_epoch_idx > args.curriculum),
    )
    update_freq = (
        args.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(args.update_freq)
        else args.update_freq[-1]
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    if getattr(args, "tpu", False):
        itr = utils.tpu_data_loader(itr)
    progress = progress_bar.progress_bar(
        itr,
        log_format=args.log_format,
        log_interval=args.log_interval,
        epoch=epoch_itr.epoch,
        tensorboard_logdir=(
            args.tensorboard_logdir if distributed_utils.is_master(args) else None
        ),
        default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
    )

    trainer.begin_epoch(epoch_itr.epoch)
    
    valid_subsets = args.valid_subset.split(",")
    should_stop = False
    num_updates = trainer.get_num_updates()
    for i, samples in enumerate(progress):
        
        with metrics.aggregate("train_inner"), torch.autograd.profiler.record_function(
            "train_step-%d" % i
        ):
            log_output = trainer.train_step(samples, run = run)

            for name, param in trainer.model.named_parameters():
                if i % 10 == 0: # MSKIM Clip Value Step Wise Logging
                    if 'clip_val' in name:
                        run["metrics/ClipVal/grad/" + name[17:]].log(param.grad)
                        run["metrics/ClipVal/" + name[17:]].log(param)
                         
        if log_output is not None:  # not OOM, overflow, ...
            # log mid-epoch stats
            num_updates = trainer.get_num_updates()
            if num_updates % args.log_interval == 0:
                stats = get_training_stats(metrics.get_smoothed_values("train_inner"))
                progress.log(stats, tag="train_inner", step=num_updates)

                # reset mid-epoch stats after each log interval
                # the end-of-epoch stats will still be preserved
                metrics.reset_meters("train_inner")

        end_of_epoch = not itr.has_next()
        valid_losses, should_stop, epoch_losses = validate_and_save(
            args, trainer, task, epoch_itr, valid_subsets, end_of_epoch
        )

        if should_stop:
            break

    # log end-of-epoch stats
    logger.info("end of epoch {} (average epoch stats below)".format(epoch_itr.epoch))
    stats = get_training_stats(metrics.get_smoothed_values("train"))
    progress.print(stats, tag="train", step=num_updates)

    # reset epoch-level meters
    metrics.reset_meters("train")
    return valid_losses, should_stop, epoch_losses # MSKIM add loss


def validate_and_save(args, trainer, task, epoch_itr, valid_subsets, end_of_epoch):
    num_updates = trainer.get_num_updates()
    do_save = (
        args.save_interval_updates > 0
        and num_updates > 0
        and num_updates % args.save_interval_updates == 0
        and num_updates >= args.validate_after_updates
    ) or (end_of_epoch and epoch_itr.epoch % args.save_interval == 0)
    do_validate = (
        (not end_of_epoch and do_save)  # validate during mid-epoch saves
        or (end_of_epoch and epoch_itr.epoch % args.validate_interval == 0)
        or (args.validate_interval_updates > 0 and num_updates % args.validate_interval_updates == 0)
    ) and not args.disable_validation

    # Validate
    valid_losses = [None]
    epoch_losses = [None]

    if do_validate:
        valid_losses, epoch_losses = validate(args, trainer, task, epoch_itr, valid_subsets)

    # Stopping conditions
    max_update = args.max_update or math.inf
    should_stop = (
        should_stop_early(args, valid_losses[0])
        or trainer.get_num_updates() >= max_update
        or (
            args.stop_time_hours > 0
            and trainer.cumulative_training_time() / (60 * 60) > args.stop_time_hours
        )
    )

    # Save checkpoint
    if do_save or should_stop:
        logger.info("begin save checkpoint")
        #checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0]) #MSKIM CheckPoint Save Option

    return valid_losses, should_stop, epoch_losses


def get_training_stats(stats):
    stats["wall"] = round(metrics.get_meter("default", "wall").elapsed_time, 0)
    return stats


def validate(args, trainer, task, epoch_itr, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""

    if args.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(args.fixed_validation_seed)

    valid_losses = []
    epoch_losses = []
    for subset in subsets:
        logger.info('begin validation on "{}" subset'.format(subset))

        # Initialize data iterator
        itr = trainer.get_valid_iterator(subset).next_epoch_itr(shuffle=False)
        if getattr(args, "tpu", False):
            itr = utils.tpu_data_loader(itr)
        progress = progress_bar.progress_bar(
            itr,
            log_format=args.log_format,
            log_interval=args.log_interval,
            epoch=epoch_itr.epoch,
            prefix=f"valid on '{subset}' subset",
            tensorboard_logdir=(
                args.tensorboard_logdir if distributed_utils.is_master(args) else None
            ),
            default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
        )

        # create a new root metrics aggregator so validation metrics
        # don't pollute other aggregators (e.g., train meters)
        logit_list, target_list = [], []
        with metrics.aggregate(new_root=True) as agg:
            for sample in progress:
                if task.args.best_checkpoint_metric == 'corr':
                    _, logits, targets = trainer.valid_step(sample)
                    
                    logit_list.append(logits)
                    target_list.append(targets)
                else:
                    trainer.valid_step(sample)

        # log validation stats
        stats = get_valid_stats(args, trainer, agg.get_smoothed_values())
        # MSKIM Add Other Stats
        if task.args.best_checkpoint_metric == 'f1':
            tn = stats['tn']; tp = stats['tp']; fp = stats['fp']; fn = stats['false'] - stats['fp']
            f1 = 100 * tp/(tp+0.5*(fp+fn))
            stats['f1'] = f1

        if task.args.best_checkpoint_metric == 'mcc':
            tn = stats['tn']; tp = stats['tp']; fp = stats['fp']; fn = stats['false'] - stats['fp']
            n = tn+tp+fn+fp
            s = (tp+fn) / float(n)
            p = (tp+fp)/float(n)
            if s == 0 or p == 0 or tp ==0 :
                stats['mcc'] = 0
            else:
                mcc = 100 * (tp/float(n)-s*p)/math.sqrt(p*s*(1-s)*(1-p))
                stats["mcc"] = mcc

        if task.args.best_checkpoint_metric == 'corr':
            init_logit_tensor = torch.zeros(len(logit_list[0]), device='cuda:0') 
            init_target_tensor = torch.zeros(len(target_list[0]), device='cuda:0')
            logit_tensor = np.array(init_logit_tensor.cpu())
            target_tensor = np.array(init_target_tensor.cpu())
#            init_logit_tensor, init_target_tensor = [], []
            for t in logit_list:
                if len(t) != len(logit_list[0]):
                    sup_zeros = torch.zeros(len(logit_list[0])-len(t), device='cuda:0')
                    t = torch.cat([t,sup_zeros], dim=0)
                    t = np.array(t.cpu())
                    logit_tensor = np.concatenate((logit_tensor, t), axis = None) 
                else:
                    t = np.array(t.cpu())
                    logit_tensor = np.concatenate((logit_tensor, t), axis = None) 
            for s in target_list:
                if len(s) != len(target_list[0]):
                    sup_zeros = torch.zeros([len(target_list[0])-len(s)], device='cuda:0')
                    s = torch.cat([s, sup_zeros], dim=0)
                    s = np.array(s.cpu())
                    target_tensor = np.concatenate((target_tensor, s), axis = None)
                else:
                    s = np.array(s.cpu())
                    target_tensor = np.concatenate((target_tensor, s), axis = None)

            logit_array = logit_tensor[len(logit_list[0]):,]
            target_array = target_tensor[len(target_list[0]):,]
            pearson_corr = pearsonr(logit_array, target_array)
            spearman_corr = spearmanr(logit_array, target_array)
            corr = (pearson_corr[0] + spearman_corr[0]) / 2
            print("pearson corr :", pearson_corr[0])
            print("spearman corr :", spearman_corr[0])
            print("corr :",corr)


            stats['corr'] = corr


        progress.print(stats, tag=subset, step=trainer.get_num_updates())
        if args.log_file is not None:
            with open(args.log_file, "a") as logfile:
                logfile.write("Epoch %s: %s\n" % (str(epoch_itr.epoch), str(stats)))
                
                if task.args.best_checkpoint_metric == 'corr':
                    logfile.write("pearson corr: %s\n" % pearson_corr[0])
                    logfile.write("spearman corr: %s\n" % spearman_corr[0])
                    logfile.write("corr: %s\n" % corr)
                if task.args.best_checkpoint_metric == 'mcc':
                    logfile.write("mcc: %s\n" % mcc)

        valid_losses.append(stats[args.best_checkpoint_metric])
        epoch_losses.append(stats['loss'])

    return valid_losses, epoch_losses


def get_valid_stats(args, trainer, stats):
    stats["num_updates"] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, "best"):
        key = "best_{0}".format(args.best_checkpoint_metric)
        best_function = max if args.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best, stats[args.best_checkpoint_metric]
        )
    return stats


def cli_main(modify_parser=None): # First
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)
    if args.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                distributed_utils.call_main(args, main)
    else:
        distributed_utils.call_main(args, main)
    


if __name__ == "__main__":
    cli_main()
