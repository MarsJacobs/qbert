# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion

def soft_cross_entropy(predicts, targets):
    
    student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
    targets_prob = torch.nn.functional.softmax(targets, dim=-1)
    #cls_loss = (torch.sum((- targets_prob * student_likelihood), dim=-1).mean()) # MSKIM 
    cls_loss_2 = (- targets_prob * student_likelihood).mean()
    #cls_loss_3 = -(targets_prob * student_likelihood).sum(dim=1).mean()
    
    return cls_loss_2

@register_criterion('sentence_prediction')
class SentencePredictionCriterion(FairseqCriterion):

    def __init__(self, task, classification_head_name, regression_target):
        super().__init__(task)
        self.classification_head_name = classification_head_name
        self.regression_target = regression_target

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--classification-head-name',
                            default='sentence_classification_head',
                            help='name of the classification head to use')
        # fmt: on

    def forward(self, model, sample, model_t=None, reduce=True, args=None):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert (
            hasattr(model, 'classification_heads')
            and self.classification_head_name in model.classification_heads
        ), 'model must provide sentence classification head for --criterion=sentence_prediction'
        import pdb; pdb.set_trace()
        student_logits, student_reps, student_atts = model(
            **sample['net_input'],
            features_only=True,
            return_all_hiddens=True, # MSKIM Make Return all inner states 
            classification_head_name=self.classification_head_name,
        )
        
        loss_kd = 0
        pred_loss = 0
        att_loss = 0
        rep_loss = 0
       
        tr_loss_kd = 0
        tr_pred_loss = 0
        tr_att_loss = 0
        tr_rep_loss = 0

        if model_t is not None:
            # MSKIM Teacher Inference 
            with torch.no_grad():
                teacher_logits, teacher_reps, teacher_atts = model_t(
                    **sample['net_input'],
                    features_only=True,
                    return_all_hiddens=True,
                    classification_head_name=self.classification_head_name,
                )

            # Prediction Distill
            if args.kd == "all" or args.kd == "pred":
                if not self.regression_target: # False
                    pred_loss = soft_cross_entropy(student_logits, teacher_logits)
                else:
                    pred_loss = F.mse_loss(student_logits, teacher_logits)
                # if not self.regression_target: # False
                #     pred_loss = F.mse_loss(student_logits, teacher_logits)
                # else:
                #     pred_loss = soft_cross_entropy(student_logits, teacher_logits)

                tr_pred_loss = pred_loss.item()
            

            
            # Attention Score Distill
            if args.kd == "all" or args.kd == "att" or args.kd == "trm":
                for i, (student_att, teacher_att) in enumerate(zip(student_atts, teacher_atts)):
                    student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to("cuda"),
                                                student_att)
                    teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to("cuda"),
                                                teacher_att)

                    if args.kd_num != "none":
                        if str(i) == args.kd_num:
                        #if i >= 10:
                            tmp_loss = F.mse_loss(student_att, teacher_att)
                        else:
                            tmp_loss = F.mse_loss(student_att, teacher_att)
                            tmp_loss = tmp_loss * 0
                            
                    else:    
                        tmp_loss = F.mse_loss(student_att, teacher_att)
                    att_loss += tmp_loss
            
            
                tr_att_loss = att_loss.item()
        
            # Transformer Layer Output Distill
            if args.kd == "all" or args.kd == "output" or args.kd == "trm":
                for i, (student_rep, teacher_rep) in enumerate(zip(student_reps['inner_states'], teacher_reps['inner_states'])):
                    if args.kd_num != "none":
                        if str(i) == args.kd_num:
                        #if i >= 11:
                            tmp_loss = F.mse_loss(student_rep, teacher_rep)
                        else :
                            tmp_loss = F.mse_loss(student_rep, teacher_rep)
                            tmp_loss = tmp_loss * 0
                    else:
                        tmp_loss = F.mse_loss(student_rep, teacher_rep)

                    rep_loss += tmp_loss
                
                tr_rep_loss = rep_loss.item()
            
            if args.kd == "all":
                loss_kd = pred_loss + att_loss + rep_loss
            elif args.kd == "pred":
                loss_kd = pred_loss
            elif args.kd == "trm":
                loss_kd = att_loss + rep_loss
            elif args.kd == "output":
                loss_kd = rep_loss
            elif args.kd == "att":
                loss_kd = att_loss
            else:
                print("KD setting is unspecified!!")
                loss_kd = 0
            
            tr_loss_kd = loss_kd.item()
                
        targets = model.get_targets(sample, [student_logits]).view(-1)
        sample_size = targets.numel()
        
        # # MSKIM KD Only LOSS Setting
        # if model_t is not None:
        #     #if args.kd == 'kd_only':
        #     loss_class = 0 # MSKIM TernaryBERT does not use GT Label objective Loss
        #     tr_loss_class = 0
        # else:
        #     # MSKIM Loss Calculation 
        #     if not self.regression_target:
        #         lprobs = F.log_softmax(student_logits, dim=-1, dtype=torch.float32)
        #         loss_class = F.nll_loss(lprobs, targets, reduction='sum')
        #     else:
        #         student_logits = student_logits.view(-1).float()
        #         targets = targets.float()
        #         loss_class = F.mse_loss(student_logits, targets, reduction='sum')

        #     tr_loss_class = loss_class.item()

        
        if not self.regression_target:
            lprobs = F.log_softmax(student_logits, dim=-1, dtype=torch.float32)
            loss_class = F.nll_loss(lprobs, targets, reduction='sum')
        else:
            student_logits = student_logits.view(-1).float()
            targets = targets.float()
            loss_class = F.mse_loss(student_logits, targets, reduction='sum')

        tr_loss_class = loss_class.item()
        
        # Loss    
        loss = loss_class + loss_kd 
        tr_loss = loss.item()

        logging_output = {
            'loss': tr_loss,
            'GT_class_loss' : tr_loss_class,
            'kd_loss' : tr_loss_kd,
            'att_loss' : tr_att_loss,
            'rep_loss' : tr_rep_loss,
            'pred_loss' : tr_pred_loss,
            'ntokens': sample['ntokens'],
            'nsentences': sample_size,
            'sample_size': sample_size,
        }
        
        if not self.regression_target:
            preds = student_logits.argmax(dim=1)
            logging_output['ncorrect'] = (preds == targets).sum()
            logging_output['false'] = (preds != targets).sum()
            logging_output['tp'] = ((preds == 1) & (targets == 1)).sum()
            logging_output['tn'] = ((preds == 0) & (targets == 0)).sum()
            logging_output['fp'] = ((preds == 1) & (targets == 0)).sum()
            logging_output['fn'] = ((preds == 0) & (targets == 1)).sum()



        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        #MSKIM reduce metrics
        gt_loss_sum = sum(log.get('GT_class_loss', 0) for log in logging_outputs)
        loss_kd_sum = sum(log.get('kd_loss', 0) for log in logging_outputs)

        att_loss_sum = sum(log.get('att_loss', 0) for log in logging_outputs)
        rep_loss_sum = sum(log.get('rep_loss', 0) for log in logging_outputs)
        pred_loss_sum = sum(log.get('pred_loss', 0) for log in logging_outputs)

        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
    
        metrics.log_scalar('gt_loss', gt_loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('kd_loss', loss_kd_sum / sample_size / math.log(2), sample_size, round=3)

        metrics.log_scalar('pred_loss', pred_loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('att_loss', att_loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('rep_loss', rep_loss_sum / sample_size / math.log(2), sample_size, round=3)



        if sample_size != ntokens:
            metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=3)

        if len(logging_outputs) > 0 and 'ncorrect' in logging_outputs[0]:
            ncorrect = sum(log.get('ncorrect', 0) for log in logging_outputs)
            false = sum(log.get('false', 0) for log in logging_outputs)
            tp = sum(log.get('tp', 0) for log in logging_outputs)
            tn = sum(log.get('tn', 0) for log in logging_outputs)
            fp = sum(log.get('fp', 0) for log in logging_outputs)
            fn = sum(log.get('fn', 0) for log in logging_outputs)
            metrics.log_scalar('accuracy', 100.0 * ncorrect / nsentences, nsentences, round=1)
            #metrics.log_scalar('f1', tp * 100.0 / (tp + 0.5 * false), tp + 0.5 * false , round=1)
            metrics.log_scalar('tp', 1. * tp, 1., round=1, sum_meter=True)
            metrics.log_scalar('tn', 1. * tn, 1., round=1, sum_meter=True)
            metrics.log_scalar('fp', 1. * fp, 1., round=1, sum_meter=True)
            metrics.log_scalar('fp', 1. * fp, 1., round=1, sum_meter=True)
            metrics.log_scalar('false', 1. * false, 1., round=1, sum_meter=True)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
