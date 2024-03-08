import pdb
import time
import os
import torch
import sys

sys.path.append("..")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import evaluation
import torch.nn as nn
import torch
import random
import numpy as np
import torch.utils.data as torch_data
from torch.utils.data.distributed import DistributedSampler
import logging
from torch.autograd import Variable
from tqdm import tqdm
import copy
# from transformers.modeling_bert import BertLayerNorm
# from transformers. transformers.models.bert.modeling_bert import BertLayerNorm
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers.modeling_utils import PreTrainedModel
import Attention


logger = logging.getLogger(__name__)


def l2norm(X, eps=1e-13, dim=-1):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps + 1e-14
    X = torch.div(X, norm)
    return X


class MarginRankingLossWithScore(nn.Module):
    """
    Compute margin ranking loss
    arg input: (batchsize, subspace) and (batchsize, subspace)
    """
    def __init__(self, margin=0.0, max_violation=True,
                 cost_style='sum', direction='bidir', device=torch.device('cpu')):
        """
        :param margin:
        :param measure: cosine 余弦相似度， hist_sim 扩展 jaccard 相似度
        :param max_violation: Use max instead of sum in the rank loss
        :param cost_style: 把所有误差相加 sum，还是取平均值 mean
        :param direction: compare every diagonal score to scores in its column and row
        """
        super().__init__()
        self.margin = margin
        self.cost_style = cost_style
        self.direction = direction

        self.max_violation = max_violation
        self.device = device

    def forward(self, score):
        # num_text, num_video

        score = score.t()
        device = score.device

        diagonal = score.diag().view(score.size(0), 1)
        d1 = diagonal.expand_as(score)  # 扩展维度
        d2 = diagonal.t().expand_as(score)

        # clear diagonals
        I = torch.eye(score.size(0)) > .5
        I = I.to(device)

        cost_s = None
        cost_im = None
        # compare every diagonal score to scores in its column
        if self.direction in ['i2t', 'bidir']:
            # caption retrieval
            cost_s = (self.margin + score - d1).clamp(min=0)  # clamp 最大最小裁剪
            cost_s = cost_s.masked_fill_(I, 0)
        # compare every diagonal score to scores in its row
        if self.direction in ['t2i', 'bidir']:
            # image retrieval
            cost_im = (self.margin + score - d2).clamp(min=0)
            cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            if cost_s is not None:
                cost_s = cost_s.max(1)[0]
            if cost_im is not None:
                cost_im = cost_im.max(0)[0]

        if cost_s is None:
            cost_s = torch.zeros(1).to(device)
        if cost_im is None:
            cost_im = torch.zeros(1).to(device)

        if self.cost_style == 'sum':
            return cost_s.sum() + cost_im.sum()
        else:
            return cost_s.mean() + cost_im.mean()


class ModelBinary(PreTrainedModel):
    def __init__(self, encoder, config, tokenizer, args):
        super(ModelBinary, self).__init__(config)
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.mlp = nn.Sequential(nn.Linear(768*4, 768),
                                 nn.Tanh(),
                                 nn.Linear(768, 1),
                                 nn.Sigmoid())
        self.loss_func = nn.BCELoss()
        self.args = args

    def forward(self, code_inputs, nl_inputs, labels, return_vec=False):
        """
        label: 1 is matched. 0 is unmatched.
        """
        bs = code_inputs.shape[0]
        inputs = torch.cat((code_inputs, nl_inputs), 0)
        outputs = self.encoder(inputs, attention_mask=inputs.ne(1))[1]
        code_vec = outputs[:bs]
        nl_vec = outputs[bs:]
        if return_vec:
            return code_vec, nl_vec

        # 在这里随机构建一些负 examples，组成一个 (nl_vec, code_vec) 输入 mlp 得到 loss.
        logits = self.mlp(torch.cat((nl_vec, code_vec, nl_vec-code_vec, nl_vec*code_vec), 1))
        loss = self.loss_func(logits, labels.float())
        predictions = (logits > 0.5).int()  # (Batch, )
        return loss, predictions


class ModelBinaryCreateNeg(PreTrainedModel):
    """
    在 ModelBinary 基础上增加 随机构建一些负 examples，组成一个 (nl_vec, code_vec) 输入 mlp 得到 loss.
    """
    def __init__(self, encoder, config, tokenizer, args):
        super().__init__(config)
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.mlp = nn.Sequential(nn.Linear(768*4, 768),
                                 nn.Tanh(),
                                 nn.Linear(768, 1),
                                 nn.Sigmoid())
        self.loss_func = nn.BCELoss()
        self.args = args

    def forward(self, code_inputs, nl_inputs, labels, return_vec=False):
        """
        label: 1 is matched. 0 is unmatched.
        """
        if not (labels == 1).all():
            print("labels are not all 1. It does not match ModelBinaryCreateNeg class.")

        labels = labels.unsqueeze(1)
        code_vec = self.encoder(code_inputs, attention_mask=code_inputs.ne(1))[1]
        nl_vec = self.encoder(nl_inputs, attention_mask=nl_inputs.ne(1))[1]
        if return_vec:
            return code_vec, nl_vec


        logits = self.mlp(torch.cat((nl_vec, code_vec, nl_vec-code_vec, nl_vec*code_vec), 1))
        loss_positive = self.loss_func(logits, labels.float())
        predictions = (logits > 0.5).int()  # (Batch, )

        # 在这里随机构建一些负 examples，组成一个 (nl_vec, code_vec) 输入 mlp 得到 loss.
        # shuffle_index =[(each+1) % len(code_vec) for each in list(range(len(code_vec)))]
        shuffle_index=list(range(len(code_vec))); random.shuffle(shuffle_index)

        shuffle_index1 = list(range(len(code_vec)))
        # random.shuffle(shuffle_index1)

        # nl_vec = nl_vec[shuffle_index1]
        code_vec = code_vec[shuffle_index]
        logits_negative = self.mlp(
            torch.cat((nl_vec, code_vec,
                       nl_vec-code_vec, nl_vec*code_vec), 1))
        loss_negative = self.loss_func(logits_negative, torch.zeros_like(labels).float())

        loss = loss_positive+loss_negative
        loss_dict = {
            "all": loss, "loss_positive": loss_positive, "loss_negative": loss_negative
        }

        return loss_dict, predictions

    def predict_old(self, code_inputs, nl_inputs, device=torch.device("cpu"),
                num_workers=10, batch_size=256, induce=[]):
        """
        返回相似度矩阵. 必然是在 single GPU
        """

        from tqdm import tqdm
        score = None
        time_dict = {
            "nl_time": 0, "code_time": 0, "score_time": 0
        }

        with torch.no_grad():
            nl_vec = None
            start_time = time.time()
            logger.info("Calculate nl_inputs, code_inputs")
            for each in tqdm(torch_data.dataloader.DataLoader(dataset=nl_inputs, batch_size=batch_size, shuffle=False,
                                                   num_workers=num_workers)):
                each = each.to(device)
                each = self.encoder(each, attention_mask=each.ne(1))[1].cpu()
                nl_vec = each if nl_vec is None else torch.cat((nl_vec, each), dim=0)
            end_time = time.time()
            time_dict["nl_time"] = end_time - start_time

            code_vec = None
            start_time = time.time()
            for each in tqdm(torch_data.dataloader.DataLoader(dataset=code_inputs, batch_size=batch_size, shuffle=False,
                                                   num_workers=num_workers)):
                each = each.to(device)
                each = self.encoder(each, attention_mask=each.ne(1))[1].cpu()
                code_vec = each if code_vec is None else torch.cat((code_vec, each), dim=0)
            end_time = time.time()
            time_dict["code_time"] = end_time - start_time

            start_time = time.time()
            batch_size = 8
            old_each_nl_num = 0
            logger.info("Calculate Sim matrix")
            for each_nl in tqdm(torch_data.dataloader.DataLoader(dataset=nl_vec, batch_size=batch_size, shuffle=False,
                                                   num_workers=num_workers)):
                if each_nl.shape[0] != old_each_nl_num:
                    old_each_nl_num = each_nl.shape[0]
                    code_vec_repeat = code_vec.unsqueeze(0).repeat(each_nl.shape[0], 1, 1).to(device)  # (batch_size, num_code, 768)
                num_code = code_vec_repeat.shape[1]
                each_nl = each_nl.to(device).unsqueeze(1).repeat(1, num_code, 1)  # (batch_size, num_code, 768)
                with torch.no_grad():
                    score_each = self.mlp(torch.cat((each_nl, code_vec_repeat, each_nl - code_vec_repeat, each_nl * code_vec_repeat), -1)).cpu().squeeze(-1)


                # score_each = None
                # for index_code, each_code in enumerate(torch.chunk(code_vec, len(code_vec)//batch_size, dim=0)):
                #     logits = self.mlp(torch.cat((each_nl, each_code, each_nl - each_code, each_nl * each_code), 1))
                #     score_each = logits.cpu() if score_each is None else torch.cat((score_each, logits), dim=1)

                score = score_each if score is None else torch.cat((score, score_each), dim=0)
            end_time = time.time()
            time_dict["score_time"] = end_time - start_time

        return score, time_dict

    def predict(self, input_dict):
        """
        返回相似度矩阵. 必然是在 single GPU
        """
        code_inputs, nl_inputs, device, num_workers, batch_size, induce = \
            input_dict['code_inputs'], input_dict['nl_inputs'], input_dict['device'], \
            input_dict['num_workers'], input_dict['batch_size'], input_dict['induce'],
        from tqdm import tqdm
        score = None
        time_dict = {
            "nl_time": 0, "code_time": 0, "score_time": 0
        }

        with torch.no_grad():
            nl_vec = None
            logger.info("Calculate nl_inputs, code_inputs")
            for index, each in enumerate(tqdm(torch_data.dataloader.DataLoader(dataset=nl_inputs, batch_size=batch_size, shuffle=False,
                                                   num_workers=num_workers))):
                if index == 0:
                    start_time = time.time()
                each = each.to(device)
                each = self.encoder(each, attention_mask=each.ne(1))[1].cpu()
                nl_vec = each if nl_vec is None else torch.cat((nl_vec, each), dim=0)
            end_time = time.time()
            time_dict["nl_time"] = end_time - start_time

            code_vec = None
            for index, each in enumerate(tqdm(torch_data.dataloader.DataLoader(dataset=code_inputs, batch_size=batch_size, shuffle=False,
                                                   num_workers=num_workers))):
                if index == 0:
                    start_time = time.time()
                each = each.to(device)
                each = self.encoder(each, attention_mask=each.ne(1))[1].cpu()
                code_vec = each if code_vec is None else torch.cat((code_vec, each), dim=0)
            end_time = time.time()
            time_dict["code_time"] = end_time - start_time

            start_time = time.time()
            batch_size = 8
            old_each_nl_num = 0
            logger.info("Calculate Sim matrix")
            for each_nl in tqdm(torch_data.dataloader.DataLoader(dataset=nl_vec, batch_size=batch_size, shuffle=False,
                                                   num_workers=num_workers)):
                if each_nl.shape[0] != old_each_nl_num:
                    old_each_nl_num = each_nl.shape[0]
                    code_vec_repeat = code_vec.unsqueeze(0).repeat(each_nl.shape[0], 1, 1).to(device)  # (batch_size, num_code, 768)
                num_code = code_vec_repeat.shape[1]
                each_nl = each_nl.to(device).unsqueeze(1).repeat(1, num_code, 1)  # (batch_size, num_code, 768)
                with torch.no_grad():
                    score_each = self.mlp(torch.cat((each_nl, code_vec_repeat, each_nl - code_vec_repeat, each_nl * code_vec_repeat), -1)).cpu().squeeze(-1)


                # score_each = None
                # for index_code, each_code in enumerate(torch.chunk(code_vec, len(code_vec)//batch_size, dim=0)):
                #     logits = self.mlp(torch.cat((each_nl, each_code, each_nl - each_code, each_nl * each_code), 1))
                #     score_each = logits.cpu() if score_each is None else torch.cat((score_each, logits), dim=1)

                score = score_each if score is None else torch.cat((score, score_each), dim=0)
            end_time = time.time()
            time_dict["score_time"] = end_time - start_time

        return score, time_dict


class ModelCodeBertBiEncoder(PreTrainedModel):
    """
    CodeBert bi-encoder version.
    """
    def __init__(self, encoder, config, tokenizer, args):
        super().__init__(config)
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.loss_func = nn.CrossEntropyLoss()
        self.args = args

    def forward(self, code_inputs, nl_inputs, labels, return_vec=False):
        """
        label: 1 is matched. 0 is unmatched.
        """

        code_vec = self.encoder(code_inputs, attention_mask=code_inputs.ne(1))[1]
        nl_vec = self.encoder(nl_inputs, attention_mask=nl_inputs.ne(1))[1]
        if return_vec:
            return code_vec, nl_vec

        scores = nl_vec.mm(code_vec.T)

        loss = self.loss_func(scores, torch.arange(code_inputs.size(0), device=scores.device))
        predictions = labels
        loss_dict = {
            "all": loss, "loss_positive": torch.Tensor(0).to(scores.device),
            "loss_negative":torch.Tensor(0).to(scores.device)
        }

        return loss_dict, predictions

    def predict_old(self, code_inputs, nl_inputs, device=torch.device("cpu"), code_len_tuple=None,
                num_workers=10, batch_size=256, induce=[], distributed_model=None):
        """
        返回相似度矩阵. 必然是在 single GPU
        """
        from tqdm import tqdm
        score = None
        time_dict = {
            "nl_time": 0, "code_time": 0, "score_time": 0
        }

        with torch.no_grad():
            nl_vec = None
            start_time = time.time()
            logger.info("Calculate nl_inputs, code_inputs")
            for each in tqdm(torch_data.dataloader.DataLoader(dataset=nl_inputs, batch_size=batch_size, shuffle=False,
                                                   num_workers=num_workers)):
                each = each.to(device)
                each = self.encoder(each, attention_mask=each.ne(1))[1].cpu()
                nl_vec = each if nl_vec is None else torch.cat((nl_vec, each), dim=0)
            end_time = time.time()
            time_dict["nl_time"] = end_time - start_time

            code_vec = None
            start_time = time.time()
            for each in tqdm(torch_data.dataloader.DataLoader(dataset=code_inputs, batch_size=batch_size, shuffle=False,
                                                   num_workers=num_workers)):
                each = each.to(device)
                each = self.encoder(each, attention_mask=each.ne(1))[1].cpu()
                code_vec = each if code_vec is None else torch.cat((code_vec, each), dim=0)
            end_time = time.time()
            time_dict["code_time"] = end_time - start_time

            start_time = time.time()
            score = nl_vec.mm(code_vec.T)
            end_time = time.time()
            time_dict["score_time"] = end_time - start_time

        return score, time_dict

    def predict(self, input_dict):
        """
        返回相似度矩阵. 必然是在 single GPU
        """
        code_inputs, nl_inputs, device, num_workers, batch_size, induce, distributed_model = \
            input_dict['code_inputs'], input_dict['nl_inputs'], input_dict['device'], \
            input_dict['num_workers'], input_dict['batch_size'], input_dict['induce'], \
            input_dict["distributed_model"],
        from tqdm import tqdm
        score = None
        time_dict = {
            "nl_time": 0, "code_time": 0, "score_time": 0
        }

        with torch.no_grad():
            nl_vec = None
            start_time = time.time()
            logger.info("Calculate nl_inputs, code_inputs")
            for each in tqdm(torch_data.dataloader.DataLoader(dataset=nl_inputs, batch_size=batch_size, shuffle=False,
                                                   num_workers=num_workers)):
                each = each.to(device)
                each = self.encoder(each, attention_mask=each.ne(1))[1].cpu()
                nl_vec = each if nl_vec is None else torch.cat((nl_vec, each), dim=0)
            end_time = time.time()
            time_dict["nl_time"] = end_time - start_time

            code_vec = None
            start_time = time.time()
            for each in tqdm(torch_data.dataloader.DataLoader(dataset=code_inputs, batch_size=batch_size, shuffle=False,
                                                   num_workers=num_workers)):
                each = each.to(device)
                each = self.encoder(each, attention_mask=each.ne(1))[1].cpu()
                code_vec = each if code_vec is None else torch.cat((code_vec, each), dim=0)
            end_time = time.time()
            time_dict["code_time"] = end_time - start_time

            start_time = time.time()
            score = nl_vec.mm(code_vec.T)
            end_time = time.time()
            time_dict["score_time"] = end_time - start_time

        return score, time_dict


class ModelCodeBertPlus(PreTrainedModel):
    """
    CodeBert version.
    随机构建一些负 examples，CE 得到 loss.
    """
    def __init__(self, encoder, config, tokenizer, args):
        super().__init__(config)
        self.encoder_type = "one_encoder"
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.mlp = nn.Sequential(nn.Linear(768*4, 768),
                                 nn.Tanh(),
                                 nn.Linear(768, 1),
                                 nn.Sigmoid())
        self.loss_bce = nn.BCELoss()
        self.loss_ce = nn.CrossEntropyLoss()
        self.args = args

        self.dense = nn.Linear(self.encoder.config.hidden_size, self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(self.encoder.config.hidden_dropout_prob)
        self.out_proj = nn.Linear(self.encoder.config.hidden_size, 2)

    def _encode(self, bimodal_inputs):
        x = self.encoder(bimodal_inputs, attention_mask=bimodal_inputs.ne(1))[1]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        outputs = self.out_proj(x)
        return outputs

    def _nce_learning(self, code_inputs, nl_inputs, labels):
        # 首先取 dot-product，计算点乘， NCE loss
        code_vec = self.encoder(code_inputs, attention_mask=code_inputs.ne(1))[1]
        nl_vec = self.encoder(nl_inputs, attention_mask=nl_inputs.ne(1))[1]
        # if self.args.encoder_l2norm:
        #     code_vec = l2norm(code_vec, dim=-1)
        #     nl_vec = l2norm(nl_vec, dim=-1)

        nce_scores = nl_vec.mm(code_vec.T)
        nce_loss = self.loss_ce(nce_scores, torch.arange(code_inputs.size(0), device=nce_scores.device))
        return nce_scores, nce_loss

    def _ce_learning(self, nl_inputs, code_inputs, max_index_list=None):
        args = self.args

        # 在这里随机构建一些负 examples，组成一个 (nl_vec, code_vec) 输入 encoder 得到 loss.
        if max_index_list is None:
            base_list = [x for x in range(0, code_inputs.shape[0])]
            list_of_list = list(map(lambda x: [y for y in base_list if y != x], range(code_inputs.shape[0])))
            random_el_list = [random.choice(x) for x in list_of_list]
        else:
            random_el_list = max_index_list

        random_el_array = np.array(random_el_list)
        negative_index = random_el_array != -1
        random_el_array = random_el_array[negative_index]
        negative_codes = code_inputs[random_el_array]
        negative_nls = nl_inputs[negative_index]

        # negative_codes = torch.cat([code_inputs[x].unsqueeze(0) for x in random_el_list], 0)
        # if nl_inputs.shape != negative_nls.shape:
        #     print(nl_inputs.shape, negative_nls.shape)
        #     pdb.set_trace()

        bimodal_inputs = torch.cat([torch.cat([nl_inputs, negative_nls], 0),
                                    torch.cat([code_inputs, negative_codes], 0)
                                    ], 1)

        outputs = self._encode(bimodal_inputs)

        logits = torch.softmax(outputs, -1)[:, 0]
        itm_target = torch.tensor([1] * code_inputs.shape[0] + [0] * negative_codes.shape[0]).to(args.device)
        if args.ClassifyLoss == 'bce':
            loss = self.loss_bce(logits, itm_target.float())
        elif args.ClassifyLoss == 'ce':
            loss = self.loss_ce(outputs, itm_target)

        predictions = (logits[0:code_inputs.shape[0]] > 0.5).int()  # (Batch, )

        return loss, predictions

    def forward(self, code_inputs, nl_inputs, labels=torch.ones(1), return_outputs=False):
        """
        label: 1 is matched. 0 is unmatched. label may be only 1.
        """
        args = self.args
        nce_weight = args.nce_weight
        if hasattr(args, "margin"):
            margin = args.margin
        else:
            margin = 0.2

        if return_outputs:
            bimodal_inputs = torch.cat((code_inputs, nl_inputs), dim=1).to(args.device)
            outputs = self._encode(bimodal_inputs)
            return outputs

        if not (labels == 1).all():
            raise Exception("labels are not all 1. It does not match ModelCodeBert class.")

        if args.max_neg:
            nce_scores, nce_loss = self._nce_learning(code_inputs, nl_inputs, labels)
            nce_scores = evaluation.z_norm(nce_scores, dim=-1)
            # 然后计算 分类 CE loss
            # 找到最大 hard code example, score need to be (num_nl, num_code)
            I = torch.eye(nce_scores.size(0)) > .5
            I = I.to(nce_scores.device)

            diagonal = nce_scores.diag().view(nce_scores.size(0), 1)
            d1 = diagonal.expand_as(nce_scores)
            cost_nl2code = (margin + nce_scores - d1).clamp(min=0)

            max_index_list = cost_nl2code.masked_fill_(I, -10).argmax(dim=1).tolist()  # dim=1 代表找每行 hard
            # 当没有 hard case 时，设为 -1
            for index in range(0 , len(max_index_list)):
                if cost_nl2code[index, max_index_list[index]] == 0:
                    max_index_list[index] = -1
            # pdb.set_trace()
        else:
            max_index_list = None
            nce_loss = torch.zeros(1).to(code_inputs.device)

        ce_loss, predictions = self._ce_learning(code_inputs, nl_inputs, max_index_list=max_index_list)

        nce_loss = nce_weight*nce_loss

        loss_dict = {
            "all": nce_loss+ce_loss, "ce_loss": ce_loss, "nce_loss": nce_loss
        }
        return loss_dict, predictions

    def predict_old(self, code_inputs, nl_inputs, device=torch.device("cpu"), code_len_tuple=None,
                num_workers=10, batch_size=256, induce=[], distributed_model=None):
        """
        返回相似度矩阵. 必然是在 single GPU
        """
        local_rank = self.args.local_rank
        from tqdm import tqdm
        score = torch.zeros((len(nl_inputs), len(code_inputs))) - 50
        time_dict = {
            "nl_time": 0, "code_time": 0, "score_time": 0
        }

        with torch.no_grad():
            start_time = 0
            logger.info("Calculate Sim matrix")
            if local_rank != -1:
                sampler = DistributedSampler(induce, shuffle=False)
            else:
                sampler = None

            for induce_batch in tqdm(torch_data.dataloader.DataLoader(
                    dataset=induce, batch_size=batch_size,
                    num_workers=num_workers, sampler=sampler)):
                if start_time == 0:
                    start_time = time.time()
                induce_batch = induce_batch.tolist()
                query_index = [item // len(code_inputs) for item in induce_batch]
                code_index = [item % len(code_inputs) for item in induce_batch]

                each_code_inputs, each_nl_inputs = code_inputs[code_index].to(device), nl_inputs[query_index].to(device)
                # bimodal_inputs = torch.cat((each_code_inputs, each_nl_inputs), dim=1).to(device)
                # outputs = self._encode(bimodal_inputs)
                # outputs = distributed_model(each_code_inputs, each_nl_inputs, return_outputs=True)
                outputs = distributed_model(code_inputs=each_code_inputs, nl_inputs=each_nl_inputs, return_outputs=True)

                logits = torch.softmax(outputs, -1)[:, 0]


                for logit_index, each_logit in enumerate(logits):
                    score[query_index[logit_index], code_index[logit_index]] = each_logit

            end_time = time.time()
            time_dict["score_time"] = end_time - start_time

        return score, time_dict

    def predict(self, input_dict):
        """
        返回相似度矩阵. 必然是在 single GPU
        """
        code_inputs, nl_inputs, device, num_workers, batch_size, induce, distributed_model = \
            input_dict['code_inputs'], input_dict['nl_inputs'], input_dict['device'], \
            input_dict['num_workers'], input_dict['batch_size'], input_dict['induce'], \
            input_dict["distributed_model"],
        local_rank = self.args.local_rank
        from tqdm import tqdm
        score = torch.zeros((len(nl_inputs), len(code_inputs))) - 50
        time_dict = {
            "nl_time": 0, "code_time": 0, "score_time": 0
        }

        with torch.no_grad():
            start_time = 0
            logger.info("Calculate Sim matrix")
            if local_rank != -1:
                sampler = DistributedSampler(induce, shuffle=False)
            else:
                sampler = None

            for induce_batch in tqdm(torch_data.dataloader.DataLoader(
                    dataset=induce, batch_size=batch_size,
                    num_workers=num_workers, sampler=sampler)):
                if start_time == 0:
                    start_time = time.time()
                induce_batch = induce_batch.tolist()
                query_index = [item // len(code_inputs) for item in induce_batch]
                code_index = [item % len(code_inputs) for item in induce_batch]

                each_code_inputs, each_nl_inputs = code_inputs[code_index].to(device), nl_inputs[query_index].to(device)
                # bimodal_inputs = torch.cat((each_code_inputs, each_nl_inputs), dim=1).to(device)
                # outputs = self._encode(bimodal_inputs)
                # outputs = distributed_model(each_code_inputs, each_nl_inputs, return_outputs=True)
                outputs = distributed_model(code_inputs=each_code_inputs, nl_inputs=each_nl_inputs, return_outputs=True)

                logits = torch.softmax(outputs, -1)[:, 0]


                for logit_index, each_logit in enumerate(logits):
                    score[query_index[logit_index], code_index[logit_index]] = each_logit

            end_time = time.time()
            time_dict["score_time"] = end_time - start_time

        return score, time_dict


class ModelGraphCodeBERT(ModelCodeBertPlus):
    """
    GraphCodeBERTMe.
    """
    def __init__(self, encoder, config, tokenizer, args):
        super().__init__(encoder, config, tokenizer, args)
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.loss_bce = nn.BCELoss()
        self.loss_ce = nn.CrossEntropyLoss()
        self.args = args

    def forward(self, code_inputs, nl_inputs, labels=torch.ones(1), return_outputs=False):
        """
        label: 1 is matched. 0 is unmatched. label may be only 1.
        """
        args = self.args

        if not (labels == 1).all():
            raise Exception("labels are not all 1. It does not match ModelCodeBert class.")

        nce_scores, nce_loss = self._nce_learning(code_inputs, nl_inputs, labels)
        if return_outputs:
            return nce_scores

        ce_loss = torch.zeros(1).to(code_inputs.device)


        loss_dict = {
            "all": nce_loss+ce_loss, "ce_loss": ce_loss, "nce_loss": nce_loss
        }
        return loss_dict, torch.zeros(1).to(code_inputs.device)

    def predict_old(self, code_inputs, nl_inputs, device=torch.device("cpu"), code_len_tuple=None,
                num_workers=10, batch_size=256, induce=[], distributed_model=None):
        """
        返回相似度矩阵. 必然是在 single GPU
        """
        local_rank = self.args.local_rank
        args = self.args
        batchsize = batch_size
        from tqdm import tqdm
        score = torch.zeros((len(nl_inputs), len(code_inputs))) - 50
        time_dict = {
            "nl_time": 0, "code_time": 0, "score_time": 0
        }

        with torch.no_grad():
            start_time = 0
            logger.info("%s Calculate nl_inputs, code_inputs" % type(self).__name__)
            query_embs = None
            start_time = time.time()
            for query_token in tqdm(
                    torch_data.dataloader.DataLoader(dataset=nl_inputs, batch_size=batchsize, shuffle=False,
                                                     num_workers=args.num_workers)):
                query_token = query_token.to(device)
                query_emb = self.encoder(query_token, attention_mask=query_token.ne(1))[1].cpu()
                query_embs = query_emb if query_embs is None else torch.cat((query_embs, query_emb), dim=0)
            end_time = time.time()
            time_dict["nl_time"] = end_time - start_time

            code_embs = None
            start_time = time.time()
            for code_token in tqdm(
                    torch_data.dataloader.DataLoader(dataset=code_inputs, batch_size=batchsize,
                                                     shuffle=False,
                                                     num_workers=args.num_workers)):
                code_token = code_token.to(device)
                code_emb = self.encoder(code_token, attention_mask=code_token.ne(1))[1].cpu()

                # pdb.set_trace()
                code_embs = code_emb if code_embs is None else torch.cat((code_embs, code_emb), dim=0)
            end_time = time.time()
            time_dict["code_time"] = end_time - start_time
            if self.args.encoder_l2norm:
                query_embs = l2norm(query_embs, dim=-1)
                code_embs = l2norm(code_embs, dim=-1)

            start_time = time.time()
            score = query_embs.mm(code_embs.T)
            # score = evaluation.l2norm(query_embs).mm(evaluation.l2norm(code_embs).T)
            end_time = time.time()
            time_dict["score_time"] = end_time - start_time

        return score, time_dict

    def predict(self, input_dict):
        """
        返回相似度矩阵. 必然是在 single GPU
        """
        code_inputs, nl_inputs, device, num_workers, batch_size, induce, distributed_model = \
            input_dict['code_inputs'], input_dict['nl_inputs'], input_dict['device'], \
            input_dict['num_workers'], input_dict['batch_size'], input_dict['induce'], \
            input_dict["distributed_model"],
        local_rank = self.args.local_rank
        args = self.args
        batchsize = batch_size
        from tqdm import tqdm
        score = torch.zeros((len(nl_inputs), len(code_inputs))) - 50
        time_dict = {
            "nl_time": 0, "code_time": 0, "score_time": 0
        }

        with torch.no_grad():
            start_time = 0
            logger.info("%s Calculate nl_inputs, code_inputs" % type(self).__name__)
            query_embs = None
            start_time = time.time()
            for query_token in tqdm(
                    torch_data.dataloader.DataLoader(dataset=nl_inputs, batch_size=batchsize, shuffle=False,
                                                     num_workers=args.num_workers)):
                query_token = query_token.to(device)
                query_emb = self.encoder(query_token, attention_mask=query_token.ne(1))[1].cpu()
                query_embs = query_emb if query_embs is None else torch.cat((query_embs, query_emb), dim=0)
            end_time = time.time()
            time_dict["nl_time"] = end_time - start_time

            code_embs = None
            start_time = time.time()
            for code_token in tqdm(
                    torch_data.dataloader.DataLoader(dataset=code_inputs, batch_size=batchsize,
                                                     shuffle=False,
                                                     num_workers=args.num_workers)):
                code_token = code_token.to(device)
                code_emb = self.encoder(code_token, attention_mask=code_token.ne(1))[1].cpu()

                # pdb.set_trace()
                code_embs = code_emb if code_embs is None else torch.cat((code_embs, code_emb), dim=0)
            end_time = time.time()
            time_dict["code_time"] = end_time - start_time
            if self.args.encoder_l2norm:
                query_embs = l2norm(query_embs, dim=-1)
                code_embs = l2norm(code_embs, dim=-1)

            start_time = time.time()
            score = query_embs.mm(code_embs.T)
            # score = evaluation.l2norm(query_embs).mm(evaluation.l2norm(code_embs).T)
            end_time = time.time()
            time_dict["score_time"] = end_time - start_time

        return score, time_dict


class ModelGraphCodeBERTOriginal(nn.Module):
    """
    GraphCodeBERTOriginal.
    """
    def __init__(self, encoder, config, tokenizer, args):
        super().__init__()
        self.encoder_type = "bi_encoder"
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.loss_bce = nn.BCELoss()
        self.loss_ce = nn.CrossEntropyLoss()
        self.args = args

    def forward(self, code_inputs=None, attn_mask=None, position_idx=None, nl_inputs=None):
        """
        """

        # For code
        if code_inputs is not None:
            code_vec = self.encoder(code_inputs, attention_mask=code_inputs.ne(1))[1]

            # nodes_mask = position_idx.eq(0)
            # token_mask = position_idx.ge(2)
            # inputs_embeddings = self.encoder.embeddings.word_embeddings(code_inputs)
            # nodes_to_token_mask = nodes_mask[:, :, None] & token_mask[:, None, :] & attn_mask
            # nodes_to_token_mask = nodes_to_token_mask / (nodes_to_token_mask.sum(-1) + 1e-10)[:, :, None]
            # avg_embeddings = torch.einsum("abc,acd->abd", nodes_to_token_mask, inputs_embeddings)
            # inputs_embeddings = inputs_embeddings * (~nodes_mask)[:, :, None] + avg_embeddings * nodes_mask[:, :, None]
            # code_vec = self.encoder(inputs_embeds=inputs_embeddings, attention_mask=attn_mask, position_ids=position_idx)[1]

            if nl_inputs is None:
                return code_vec

        # For language
        if nl_inputs is not None:
            nl_vec = self.encoder(nl_inputs, attention_mask=nl_inputs.ne(1))[1]
            if code_inputs is None:
                return nl_vec

        scores = torch.einsum("ab,cb->ac", nl_vec, code_vec)
        nce_loss = self.loss_ce(scores, torch.arange(code_inputs.size(0), device=scores.device))

        ce_loss = torch.zeros(1).to(code_inputs.device)

        loss_dict = {
            "all": nce_loss+ce_loss, "ce_loss": ce_loss, "nce_loss": nce_loss
        }
        return loss_dict, torch.zeros(1).to(code_inputs.device)

    def predict_old(self, code_inputs, nl_inputs, device=torch.device("cpu"), code_len_tuple=None,
                num_workers=10, batch_size=256, induce=[], distributed_model=None):
        """
        返回相似度矩阵. 必然是在 single GPU
        code_len_tuple, num_workers, induce 不需要
        """
        local_rank = self.args.local_rank
        args = self.args
        batchsize = batch_size
        from tqdm import tqdm
        time_dict = {
            "nl_time": 0, "code_time": 0, "score_time": 0
        }

        with torch.no_grad():
            start_time = 0
            logger.info("%s Calculate nl_inputs, code_inputs" % type(self).__name__)
            query_embs = None
            start_time = time.time()
            for query_token in tqdm(
                    torch_data.dataloader.DataLoader(dataset=nl_inputs, batch_size=batchsize, shuffle=False,
                                                     num_workers=args.num_workers)):
                query_token = query_token.to(device)
                # query_emb = self.encoder(query_token, attention_mask=query_token.ne(1))[1].cpu()
                query_emb = distributed_model(nl_inputs=query_token).cpu()
                query_embs = query_emb if query_embs is None else torch.cat((query_embs, query_emb), dim=0)
            end_time = time.time()
            time_dict["nl_time"] = end_time - start_time

            code_embs = None
            start_time = time.time()
            for code_token in tqdm(
                    torch_data.dataloader.DataLoader(dataset=code_inputs, batch_size=batchsize,
                                                     shuffle=False,
                                                     num_workers=args.num_workers)):
                code_token = code_token.to(device)
                # code_emb = self.encoder(code_token, attention_mask=code_token.ne(1))[1].cpu()
                code_emb = distributed_model(code_inputs=code_token).cpu()

                # pdb.set_trace()
                code_embs = code_emb if code_embs is None else torch.cat((code_embs, code_emb), dim=0)
            end_time = time.time()
            time_dict["code_time"] = end_time - start_time
            if self.args.encoder_l2norm:
                query_embs = l2norm(query_embs, dim=-1)
                code_embs = l2norm(code_embs, dim=-1)

            start_time = time.time()
            score = query_embs.mm(code_embs.T)
            # score = evaluation.l2norm(query_embs).mm(evaluation.l2norm(code_embs).T)
            end_time = time.time()
            time_dict["score_time"] = end_time - start_time

        return score, time_dict

    def predict(self, input_dict):
        """
        返回相似度矩阵. 必然是在 single GPU
        """
        code_inputs, nl_inputs, device, num_workers, batch_size, induce, distributed_model = \
            input_dict['code_inputs'], input_dict['nl_inputs'], input_dict['device'], \
            input_dict['num_workers'], input_dict['batch_size'], input_dict['induce'], \
            input_dict["distributed_model"],
        local_rank = self.args.local_rank
        args = self.args
        batchsize = batch_size
        from tqdm import tqdm
        time_dict = {
            "nl_time": 0, "code_time": 0, "score_time": 0
        }

        with torch.no_grad():
            start_time = 0
            logger.info("%s Calculate nl_inputs, code_inputs" % type(self).__name__)
            query_embs = None
            start_time = time.time()
            for query_token in tqdm(
                    torch_data.dataloader.DataLoader(dataset=nl_inputs, batch_size=batchsize, shuffle=False,
                                                     num_workers=args.num_workers)):
                query_token = query_token.to(device)
                # query_emb = self.encoder(query_token, attention_mask=query_token.ne(1))[1].cpu()
                query_emb = distributed_model(nl_inputs=query_token).cpu()
                query_embs = query_emb if query_embs is None else torch.cat((query_embs, query_emb), dim=0)
            end_time = time.time()
            time_dict["nl_time"] = end_time - start_time

            code_embs = None
            start_time = time.time()
            for code_token in tqdm(
                    torch_data.dataloader.DataLoader(dataset=code_inputs, batch_size=batchsize,
                                                     shuffle=False,
                                                     num_workers=args.num_workers)):
                code_token = code_token.to(device)
                # code_emb = self.encoder(code_token, attention_mask=code_token.ne(1))[1].cpu()
                code_emb = distributed_model(code_inputs=code_token).cpu()

                # pdb.set_trace()
                code_embs = code_emb if code_embs is None else torch.cat((code_embs, code_emb), dim=0)
            end_time = time.time()
            time_dict["code_time"] = end_time - start_time
            if self.args.encoder_l2norm:
                query_embs = l2norm(query_embs, dim=-1)
                code_embs = l2norm(code_embs, dim=-1)

            start_time = time.time()
            score = query_embs.mm(code_embs.T)
            # score = evaluation.l2norm(query_embs).mm(evaluation.l2norm(code_embs).T)
            end_time = time.time()
            time_dict["score_time"] = end_time - start_time

        return score, time_dict


class ModelGraphCodeBERTMultiCodeFusion(nn.Module):
    """
    GraphCodeBERTMultiCodeFusion
    Multiple code fusion.
    """
    def __init__(self, encoder, config, tokenizer, args):
        super().__init__()
        self.encoder_type = "bi_encoder"
        self.encoder = encoder
        self.fusion_layer = Attention.Attention_1(
            768, with_ave=args.AttentionWithAve, mul=False, common_type=args.AttentionCommonType)

        self.config = config
        self.tokenizer = tokenizer
        self.loss_bce = nn.BCELoss()
        self.loss_ce = nn.CrossEntropyLoss()
        self.args = args

    def forward(self, code_inputs=None, code_len_tuple=None, nl_inputs=None):
        """
        code_inputs: (batch_size, num_subcode, embedding_size),
        code_len_tuple: [1,2,1, ...]
        """
        # For code
        if code_inputs is not None:
            # code_inputs = code_inputs.squeeze(dim=0)

            code_embs = None
            try:
                for index, code_len in enumerate(code_len_tuple):
                    # start = sum(code_len_tuple[0:index])
                    start = 0
                    end = start + int(code_len)
                    # if end-start > 10:
                    #     start2end = np.linspace(start, end-1, 10, dtype=int)
                    # else:
                    start2end = np.linspace(start, end-1, end-start, dtype=int)
                    code_input = code_inputs[index][start2end]
                    code_encoder_output = self.encoder(code_input, attention_mask=code_input.ne(1))[1]
                    code_emb = self.fusion_layer(code_encoder_output.unsqueeze(0))

                    code_embs = torch.cat(
                        (code_embs, code_emb), dim=0) if code_embs is not None else code_emb
            except Exception as e:
                print(e)
                pdb.set_trace()


            if nl_inputs is None:
                return code_embs

        # For language
        if nl_inputs is not None:
            nl_vec = self.encoder(nl_inputs, attention_mask=nl_inputs.ne(1))[1]
            if code_inputs is None:
                return nl_vec

        scores = torch.einsum("ab,cb->ac", nl_vec, code_embs)
        nce_loss = self.loss_ce(scores, torch.arange(nl_vec.size(0), device=scores.device))

        ce_loss = torch.zeros(1).to(nl_vec.device)

        loss_dict = {
            "all": nce_loss+ce_loss, "ce_loss": ce_loss, "nce_loss": nce_loss
        }
        return loss_dict, torch.zeros(1).to(nl_vec.device)

    def predict_old(self, code_inputs, nl_inputs, code_len_tuple, device=torch.device("cpu"),
                num_workers=10, batch_size=256, induce=[], distributed_model=None):
        """
        返回相似度矩阵. 必然是在 single GPU
        code_len_tuple: ([1], [3])
        """
        local_rank = self.args.local_rank
        args = self.args
        batchsize = batch_size
        from tqdm import tqdm
        time_dict = {
            "nl_time": 0, "code_time": 0, "score_time": 0
        }

        with torch.no_grad():
            start_time = 0
            logger.info("%s Calculate nl_inputs, code_inputs" % type(self).__name__)
            query_embs = None
            start_time = time.time()
            for query_token in tqdm(
                    torch_data.dataloader.DataLoader(dataset=nl_inputs, batch_size=batchsize, shuffle=False,
                                                     num_workers=args.num_workers)):
                query_token = query_token.to(device)
                query_emb = distributed_model(nl_inputs=query_token).cpu()
                query_embs = query_emb if query_embs is None else torch.cat((query_embs, query_emb), dim=0)
            end_time = time.time()
            time_dict["nl_time"] = end_time - start_time

            code_embs = None
            start_time = time.time()
            for code_token, code_len_tuple_small in tqdm(
                    torch_data.dataloader.DataLoader(dataset=zip(code_inputs, code_len_tuple), batch_size=batchsize,
                                                     shuffle=False,
                                                     num_workers=args.num_workers)):
                code_token = code_token.to(device)
                # code_emb = self.encoder(code_token, attention_mask=code_token.ne(1))[1].cpu()
                code_emb = distributed_model(code_inputs=code_token, code_len_tuple=code_len_tuple_small).cpu()

                # pdb.set_trace()
                code_embs = code_emb if code_embs is None else torch.cat((code_embs, code_emb), dim=0)
            end_time = time.time()
            time_dict["code_time"] = end_time - start_time
            if self.args.encoder_l2norm:
                query_embs = l2norm(query_embs, dim=-1)
                code_embs = l2norm(code_embs, dim=-1)

            start_time = time.time()
            score = query_embs.mm(code_embs.T)
            # score = evaluation.l2norm(query_embs).mm(evaluation.l2norm(code_embs).T)
            end_time = time.time()
            time_dict["score_time"] = end_time - start_time

        return score, time_dict

    def predict(self, input_dict):
        """
        返回相似度矩阵. 必然是在 single GPU
        """
        code_inputs, nl_inputs, device, num_workers, batch_size = \
            input_dict['code_inputs'], input_dict['nl_inputs'], input_dict['device'], \
            input_dict['num_workers'], input_dict['batch_size']
        new_query_ids = input_dict["new_query_ids"]
        eval_dataset_class = input_dict["eval_dataset_class"]
        # 得到 code_index 到 其他所有 code index 字典
        if new_query_ids is None:
            new_query_ids = [[each] for each in range(len(code_inputs))]
        code_index_one2other = {}
        for each in new_query_ids:
            code_index_one2other[each[0]] = each

        local_rank = self.args.local_rank
        args = self.args
        batchsize = batch_size
        from tqdm import tqdm
        time_dict = {
            "nl_time": 0, "code_time": 0, "score_time": 0
        }

        with torch.no_grad():
            start_time = 0
            logger.info("%s Calculate nl_inputs, code_inputs" % type(self).__name__)
            query_embs = None
            start_time = time.time()
            for query_token in tqdm(
                    torch_data.dataloader.DataLoader(dataset=nl_inputs, batch_size=batchsize, shuffle=False,
                                                     num_workers=args.num_workers)):
                query_token = query_token.to(device)
                query_emb = self.forward(nl_inputs=query_token).cpu()
                query_embs = query_emb if query_embs is None else torch.cat((query_embs, query_emb), dim=0)
            end_time = time.time()
            time_dict["nl_time"] = end_time - start_time

            code_embs = None
            start_time = 0
            for code_indexs in tqdm(torch_data.dataloader.DataLoader(
                    dataset=np.arange(0, len(code_inputs)), batch_size=int(batchsize/2)+2, shuffle=False,
                                                     num_workers=args.num_workers)):
                if start_time == 0:
                    start_time = time.time()
                # raw_code_token 是没有嵌入其他的
                code_len_list = []
                code_token = None
                for code_index in code_indexs.tolist():
                    code_token = code_inputs[[code_index]] if code_token is None else \
                        torch.cat((code_token, code_inputs[[code_index]]), dim=0)
                    if code_index in code_index_one2other:
                        code_len_list.append(len(code_index_one2other[code_index]) + 1)
                        code_token = torch.cat(
                            (code_token, code_inputs[code_index_one2other[code_index]]))
                    else:
                        code_len_list.append(1)
                code_token = code_token.to(device)  # .unsqueeze(0)

                code_emb = self._get_code_emb(code_concat_inputs=code_token, code_len_tuple=code_len_list).cpu()

                code_embs = code_emb if code_embs is None else torch.cat((code_embs, code_emb), dim=0)
            end_time = time.time()
            time_dict["code_time"] = end_time - start_time
            if self.args.encoder_l2norm:
                query_embs = l2norm(query_embs, dim=-1)
                code_embs = l2norm(code_embs, dim=-1)

            start_time = time.time()
            score = query_embs.mm(code_embs.T)
            # score = evaluation.l2norm(query_embs).mm(evaluation.l2norm(code_embs).T)
            end_time = time.time()
            score[:, len(eval_dataset_class.retrieval_raw_code_base):] = -50

            time_dict["score_time"] = end_time - start_time

        return score, time_dict

    def _get_code_emb(self, code_concat_inputs, code_len_tuple):
        """
        code_concat_inputs: (concat_num, embedding_size)
        """
        code_encoder_output = self.encoder(code_concat_inputs, attention_mask=code_concat_inputs.ne(1))[1]

        code_embs = None
        for index, code_len in enumerate(code_len_tuple):
            start = sum(code_len_tuple[0:index])
            end = start + int(code_len)

            start2end = np.linspace(start, end - 1, end - start, dtype=int)
            code_emb = self.fusion_layer(code_encoder_output[start2end].unsqueeze(0))

            code_embs = torch.cat(
                (code_embs, code_emb), dim=0) if code_embs is not None else code_emb
        return code_embs


class ModelContra(PreTrainedModel):
    def __init__(self, encoder, config, tokenizer, args):
        super(ModelContra, self).__init__(config)
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.mlp = nn.Sequential(nn.Linear(768*4, 768),
                                 nn.Tanh(),
                                 nn.Linear(768, 1),
                                 nn.Sigmoid())
        self.loss_func = nn.BCELoss()
        self.args = args

    def forward(self, code_inputs, nl_inputs, labels, return_vec=False):
        bs = code_inputs.shape[0]
        inputs = torch.cat((code_inputs, nl_inputs), 0)
        outputs = self.encoder(inputs, attention_mask=inputs.ne(1))[1]
        code_vec = outputs[:bs]
        nl_vec = outputs[bs:]
        if return_vec:
            return code_vec, nl_vec

        nl_vec = nl_vec.unsqueeze(1).repeat([1, bs, 1])
        code_vec = code_vec.unsqueeze(0).repeat([bs, 1, 1])
        logits = self.mlp(torch.cat((nl_vec, code_vec, nl_vec-code_vec, nl_vec*code_vec), 2)).squeeze(2)  # (Batch, Batch)
        matrix_labels = torch.diag(labels).float()  # (Batch, Batch)
        poss = logits[matrix_labels==1]
        negs = logits[matrix_labels==0]

        # loss = self.loss_func(logits, matrix_labels)
        # bce equals to -(torch.log(1-logits[matrix_labels==0]).sum() + torch.log(logits[matrix_labels==1]).sum()) / (bs*bs)
        loss = - (torch.log(1 - negs).mean() + torch.log(poss).mean())
        predictions = (logits.gather(0, torch.arange(bs, device=loss.device).unsqueeze(0)).squeeze(0) > 0.5).int()
        return loss, predictions


class ModelContraOnline(PreTrainedModel):
    def __init__(self, encoder, config, tokenizer, args):
        super(ModelContraOnline, self).__init__(config)
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.mlp = nn.Sequential(nn.Linear(768*4, 768),
                                 nn.Tanh(),
                                 nn.Linear(768, 1),
                                 nn.Sigmoid())
        self.loss_func = nn.BCELoss()
        self.args = args

    def forward(self, code_inputs, nl_inputs, labels, return_vec=False):
        bs = code_inputs.shape[0]
        inputs = torch.cat((code_inputs, nl_inputs), 0)
        outputs = self.encoder(inputs, attention_mask=inputs.ne(1))[1]
        code_vec = outputs[:bs]
        nl_vec = outputs[bs:]
        if return_vec:
            return code_vec, nl_vec

        nl_vec = nl_vec.unsqueeze(1).repeat([1, bs, 1])
        code_vec = code_vec.unsqueeze(0).repeat([bs, 1, 1])
        logits = self.mlp(torch.cat((nl_vec, code_vec, nl_vec-code_vec, nl_vec*code_vec), 2)).squeeze(2) # (Batch, Batch)
        matrix_labels = torch.diag(labels).float()  # (Batch, Batch)
        poss = logits[matrix_labels==1]
        negs = logits[matrix_labels==0]

        negative_pairs = negs[negs < (poss.max() if len(poss) > 1 else negs.mean())]
        positive_pairs = poss[poss > (negs.min() if len(negs) > 1 else poss.mean())]

        loss = - (torch.log(1 - negative_pairs).mean() + torch.log(positive_pairs).mean())
        predictions = (logits.gather(0, torch.arange(bs, device=loss.device).unsqueeze(0)).squeeze(0) > 0.5).int()
        return loss, predictions


class CODEnn(nn.Module):

    def __init__(self, model, config, tokenizer, args):
        super().__init__()
        self.args = args
        self.pooling = 'mean'
        self.rnn_size = 1024
        # nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.we = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.rnn = nn.GRU(int(config.hidden_size), int(self.rnn_size),
                          2, batch_first=True, bidirectional=False)
        if args.ClassifyLoss == "MarginRankingLoss":
            self.loss_func = MarginRankingLossWithScore(margin=0.5, device=args.device)
        else:
            self.loss_func = nn.CrossEntropyLoss()

    def encoder(self, input_ids: torch.Tensor, attention_mask=None):
        device = input_ids.device
        batch_size = len(input_ids)

        # caption encoding
        lengths = [int(vec.int().sum()) for vec in attention_mask]

        # caption embedding
        x = self.we(input_ids)
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # Forward propagate RNN
        text_features, _ = self.rnn(packed)
        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(text_features, batch_first=True)

        if self.pooling == 'mean':
            # out = torch.zeros(batch_size, padded[0].shape[-1]).to(device)
            text_features = x.new_zeros((batch_size, padded[0].shape[-1])).to(device)
            for i, ln in enumerate(lengths):
                text_features[i] = torch.mean(padded[0][i][:ln], dim=0)
        elif self.pooling == 'last':
            I = torch.LongTensor(lengths).view(-1, 1, 1)
            I = I.expand(batch_size, 1, self.rnn_size) - 1
            I = I.to(device)
            text_features = torch.gather(padded[0], 1, I).squeeze(1)
        elif self.pooling == 'mean_last':
            # out1 = torch.zeros(batch_size, self.rnn_size).to(device)
            out1 = torch.zeros(batch_size, self.rnn_size).to(device)
            for i, ln in enumerate(lengths):
                out1[i] = torch.mean(padded[0][i][:ln], dim=0)

            I = torch.LongTensor(lengths).view(-1, 1, 1)
            I = I.expand(batch_size, 1, self.rnn_size) - 1
            I = I.to(device)
            out2 = torch.gather(padded[0], 1, I).squeeze(1)
            text_features = torch.cat((out1, out2), dim=1)

        return (None, text_features)

    def forward(self, code_inputs, nl_inputs, labels=torch.Tensor(0), return_vec=False):
        """
        label: 1 is matched. 0 is unmatched.
        """
        args = self.args

        code_vec = self.encoder(code_inputs, attention_mask=code_inputs.ne(1))[1]
        nl_vec = self.encoder(nl_inputs, attention_mask=nl_inputs.ne(1))[1]
        if return_vec:
            return code_vec, nl_vec

        scores = nl_vec.mm(code_vec.T)

        if args.ClassifyLoss == "MarginRankingLoss":
            loss = self.loss_func(scores)
        else:
            loss = self.loss_func(scores, torch.arange(code_inputs.size(0), device=scores.device))
        predictions = labels.to(scores.device)
        loss_dict = {
            "all": loss, "loss_positive": torch.Tensor(0).to(scores.device),
            "loss_negative":torch.Tensor(0).to(scores.device)
        }

        return loss_dict, predictions

    def predict(self, input_dict):
        """
        返回相似度矩阵. 必然是在 single GPU
        """
        code_inputs, nl_inputs, device, num_workers, batch_size, induce, distributed_model = \
            input_dict['code_inputs'], input_dict['nl_inputs'], input_dict['device'], \
            input_dict['num_workers'], input_dict['batch_size'], input_dict['induce'], \
            input_dict["distributed_model"],
        from tqdm import tqdm
        score = None
        time_dict = {
            "nl_time": 0, "code_time": 0, "score_time": 0
        }

        with torch.no_grad():
            nl_vec = None
            start_time = time.time()
            logger.info("Calculate nl_inputs, code_inputs")
            for each in tqdm(torch_data.dataloader.DataLoader(dataset=nl_inputs, batch_size=batch_size, shuffle=False,
                                                   num_workers=num_workers)):
                each = each.to(device)
                each = self.encoder(each, attention_mask=each.ne(1))[1].cpu()
                nl_vec = each if nl_vec is None else torch.cat((nl_vec, each), dim=0)
            end_time = time.time()
            time_dict["nl_time"] = end_time - start_time

            code_vec = None
            start_time = time.time()
            for each in tqdm(torch_data.dataloader.DataLoader(dataset=code_inputs, batch_size=batch_size, shuffle=False,
                                                   num_workers=num_workers)):
                each = each.to(device)
                each = self.encoder(each, attention_mask=each.ne(1))[1].cpu()
                code_vec = each if code_vec is None else torch.cat((code_vec, each), dim=0)
            end_time = time.time()
            time_dict["code_time"] = end_time - start_time

            start_time = time.time()
            score = nl_vec.mm(code_vec.T)
            end_time = time.time()
            time_dict["score_time"] = end_time - start_time

        return score, time_dict



