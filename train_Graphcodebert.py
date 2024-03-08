import torch
import torch.nn as nn
import torch.utils.data as torch_data
from torch.utils.data.distributed import DistributedSampler
import re
import os
import multiprocessing
import numpy as np
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import sys
import pdb

import logging
import pickle
import time
import random
import argparse
import json
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
from spiral import ronin
from StageClass_use_ids import BaseStageCodeSearch, set_seed, GraphCodeBertStageOfficial, get_args
from CoCLR_train.models import ModelBinaryCreateNeg, ModelCodeBertBiEncoder, \
    ModelCodeBertPlus, ModelGraphCodeBERT, ModelGraphCodeBERTOriginal, ModelGraphCodeBERTMultiCodeFusion,\
    CODEnn

from GraphCodeBERT import TextDataset, TextDatasetSplit


logger = logging.getLogger(__name__)
padding_strategy = 'max_length'
pre_process_configs = ['to_snake_case', 'no_punctuation',
                       'no_comment', 'split_with_spiral_for_8', 'lemmatizer']
PreProcessConfig = ['no_comment', "del_redundant_spaces"]
QueryPreProcessConfig = ['None']
# PreProcessConfig = ['no_comment']



def add_setting(args):
    setting_name = "seed_%d-TrainBatch_%d-ClassifyLoss_%s-" \
                   % (args.seed, args.train_batch_size, args.ClassifyLoss)
    if args.TrainModel == "CodeBertP":
        if args.max_neg:
            setting_name += 'MaxNeg-'
            if args.margin != 2:
                setting_name += 'margin_%.2f-' % args.margin
            if args.nce_weight != 1:
                setting_name += 'nce_weight_%.2f-' % args.nce_weight
        if args.encoder_name_or_path != "microsoft/codebert-base":
            setting_name += args.encoder_name_or_path

    elif args.TrainModel in ["GraphCodeBERTMe", 'CodeBert_bi-encoder',
                            "GraphCodeBERTOriginal1", "GraphCodeBERTMultiCodeFusion"]:
        if args.encoder_name_or_path != "microsoft/graphcodebert-base":
            setting_name += args.encoder_name_or_path
        if args.encoder_l2norm:
            setting_name += "encoderL2norm-"

        if args.split_type in ['token', 'word_space', 'line']:
            if args.window_setting != "WindowSize_-1,step_-1":
                setting_name += args.window_setting + "-"
                if args.split_type != "token":
                    setting_name += args.split_type + "-"
        elif args.split_type in ['ast_subtree']:
            if args.window_setting != "WindowSize_-1,step_-1":
                setting_name += args.window_setting + "-"
            setting_name += args.split_type + "-" + "MinLine" + str(args.min_line) + "-"

        if args.TrainModel == "GraphCodeBERTMultiCodeFusion":
            setting_name += "Attention"
            if args.AttentionWithAve:
                setting_name += "WithAve"
            setting_name += "CommonType" + args.AttentionCommonType
    elif args.TrainModel in ['CODEnn']:
        pass
    else:
        raise Exception("This model does not exist")

    return setting_name


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 code_tokens,
                 code_ids,
                 position_idx,
                 dfg_to_code,
                 dfg_to_dfg,
                 nl_tokens,
                 nl_ids,
                 url,
                 ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.position_idx = position_idx
        self.dfg_to_code = dfg_to_code
        self.dfg_to_dfg = dfg_to_dfg
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.url = url


class CodeBertTokenTrainPairDataset(torch_data.Dataset):
    def __init__(self, train_query_token, train_code_token, tokenizer, param=None, new_query_ids=None):
        """

        """
        if param is None:
            param = {
                'max_length': 200,
                'max_query_length': 80
            }
        self.collate_fn = None
        self.tokenizer = tokenizer
        self.train_queries = train_query_token
        self.train_codes = train_code_token

        self.max_length = param['max_length']
        self.max_query_length = param['max_query_length']

        self.new_query_ids = new_query_ids
        self.new_query_induce = []
        if self.new_query_ids is not None:
            for q_idx, each in enumerate(new_query_ids):
                self.new_query_induce.extend([(q_idx, i) for i in each])

    def __len__(self):
        if self.new_query_ids is not None:
            return len(self.new_query_induce)
        else:
            return len(self.train_queries)

    def __getitem__(self, item):
        index = item

        if self.new_query_ids is not None:
            induce = self.new_query_induce
            query = self.train_queries[induce[index][0]]
            code = self.train_codes[induce[index][1]]
        else:
            query = self.train_queries[index]
            code = self.train_codes[index]

        if type(query) != torch.Tensor:
            query = torch.tensor(query)
            code = torch.tensor(code)

        return query, code, 1, index


class CodeBertTokenTrainPairDataset1(torch_data.Dataset):
    def __init__(self, train_query_token, train_code_token, new_query_ids, param=None):
        """

        """
        if param is None:
            param = {
                'max_length': 200,
                'max_query_length': 80,
                "step_num": 6,
            }
        if "step_num" not in param:
            param["step_num"] = 6

        if not hasattr(self, "collate_fn"):
            self.collate_fn = None
        self.train_queries = train_query_token
        self.train_codes = train_code_token

        self.max_length = param['max_length']
        self.max_query_length = param['max_query_length']
        self.step_num = param['step_num']

        self.new_query_ids = new_query_ids
        self.new_query_induce = []

        for q_idx, each in enumerate(new_query_ids):
            self.new_query_induce.extend([(q_idx, i) for i in each])

        pass

    def __len__(self):
        return len(self.train_queries)

    def __getitem__(self, item):
        index = item

        query = self.train_queries[index]

        # codes = self.train_codes[[random.choice(self.new_query_ids[index]) for _ in range(self.step_num)]]
        # if len(self.new_query_ids[index]) > 1:
        #     code_len = torch.tensor(self.step_num)
        # else:
        #     code_len = torch.tensor(1)

        if len(self.new_query_ids[index]) > self.step_num:
            codes = self.train_codes[[random.choice(self.new_query_ids[index]) for _ in range(self.step_num)]]
            code_len = torch.tensor(self.step_num)
        else:
            codes = self.train_codes[self.new_query_ids[index][0]].repeat(self.step_num, 1)
            codes[0:len(self.new_query_ids[index])] = self.train_codes[self.new_query_ids[index]]
            code_len = torch.tensor(len(self.new_query_ids[index]))

        if type(query) != torch.Tensor:
            query = torch.tensor(query)
            codes = torch.tensor(codes)


        return query, codes, code_len, index

    # @staticmethod
    # def collate_fn(data):
    #     query_tuple, codes_tuple, code_len_tuple, index_tuple = list(zip(*data))
    #     querys = torch.stack(query_tuple, dim=0)
    #     codes = torch.cat(codes_tuple, dim=0)
    #     return querys, codes, code_len_tuple, index_tuple


def get_train_args():
    ## Required parameters
    # parser.add_argument("--result_log", default="./result_log", type=str,
    #                     help="the path the store metric result")
    #
    # parser.add_argument("--lang", default=None, type=str,
    #                     help="language.")
    # parser.add_argument("--tokenizer_name", default="", type=str,
    #                     help="Optional pretrained tokenizer name or path if not the same as encoder_name_or_path")
    # parser.add_argument("--encoder_name_or_path", default="microsoft/codebert-base", type=str,
    #                     help="Optional encoder_name_or_path")
    # parser.add_argument("--device", default="cuda", type=str,
    #                     help="torch device")
    # parser.add_argument("--root_path", default="~/VisualSearch", type=str,
    #                     help="root path")
    # parser.add_argument("--num_workers", default=16, type=int,
    #                     help="num_workers of dataloader.")
    # parser.add_argument("--train_batch_size", default=32, type=int,
    #                     help="num_workers of dataloader.")
    # parser.add_argument("--eval_batch_size", default=32, type=int,
    #                     help="num_workers of dataloader.")
    # parser.add_argument("--warmup_steps", default=10, type=int,
    #                     help="Linear warmup over warmup_steps.")
    # parser.add_argument("--num_train_epochs", default=10, type=int,
    #                     help="num_train_epochs")
    # parser.add_argument("--learning_rate", default=5e-5, type=float,
    #                     help="The initial learning rate for Adam.")
    # parser.add_argument("--weight_decay", default=0.0, type=float,
    #                     help="Weight deay if we apply some.")
    # parser.add_argument("--max_grad_norm", default=1.0, type=float,
    #                     help="Max gradient norm.")
    # parser.add_argument("--local_rank", type=int, default=-1,
    #                     help="For distributed training: local_rank")
    #
    # parser.add_argument("--TrainModel", default="CoCLR_model", type=str,
    #                     help="The model need to be trained. ['CoCLR_model', 'CodeBert_bi-encoder']")
    # parser.add_argument("--model_name_or_path", default=None, type=str,
    #                     help="The model checkpoint for weights initialization.")
    # parser.add_argument("--config_name", default="", type=str,
    #                     help="Optional pretrained config name or path if not the same as model_name_or_path")
    # parser.add_argument("--save_temp", action='store_true',
    #                     help="Whether to save temp model file.")
    # parser.add_argument("--overwrite", action='store_true',
    #                     help="Whether to overwrite best model file.")
    #
    # parser.add_argument("--nl_length", default=128, type=int,
    #                     help="Optional NL input sequence length after tokenization.")
    # parser.add_argument("--code_length", default=256, type=int,
    #                     help="Optional Code input sequence length after tokenization.")
    # parser.add_argument("--margin", default=2, type=float,
    #                     help="margin of triplet ranking loss.")
    # parser.add_argument("--nce_weight", default=1, type=float,
    #                     help="weight of nce loss.")
    # parser.add_argument("--max_neg", action='store_true',
    #                     help="Whether to use hard negative miner.")
    # parser.add_argument("--ClassifyLoss", default="ce", type=str,
    #                     help="The Classify Loss. ['ce', 'bce']")
    #
    # parser.add_argument("--do_train", action='store_true',
    #                     help="Whether to run training.")
    # parser.add_argument("--do_eval", action='store_true',
    #                     help="Whether to run eval on the dev set.")
    # parser.add_argument("--do_test", action='store_true',
    #                     help="Whether to run eval on the test set.")
    # parser.add_argument("--do_debug", action='store_true',
    #                     help="Whether to run in debug setting.")
    # parser.add_argument("--online_cal", action='store_true',
    #                     help="Whether to calculate the feature online (Not read the pre-calculate file).")
    #
    #
    # parser.add_argument('--seed', type=int, default=42,
    #                     help="random seed for initialization")


    logging.basicConfig(level=logging.INFO,
                        format='【%(asctime)s】 【%(levelname)s】 >>>  %(message)s', datefmt='%Y-%m-%d %H:%M')

    # print arguments
    args = get_args()

    # set device
    # device = torch.device("cuda" if (torch.cuda.is_available() and args.device == "cuda") else "cpu")
    # args.n_gpu = torch.cuda.device_count() if (torch.cuda.is_available() and args.device == "cuda") else 0
    # args.device = device
    # logger.info("device: %s, n_gpu: %s", args.device, args.n_gpu)

    # if args.do_debug:
    #     args.n_gpu = 1

    # try:
    #     # 1） 初始化
    #     torch.distributed.init_process_group(backend='nccl')
    #     # 2） 配置每个进程的gpu
    #     local_rank = torch.distributed.get_rank()
    #     torch.cuda.set_device(local_rank)
    #     args.device = torch.device("cuda", local_rank)
    #     args.local_rank = local_rank
    # except Exception as e:
    #     # 无法用 torch.distributed
    #     print(e)



    # Set seed
    set_seed(args.seed)

    return args


def eval_coclr_model(args, model, tokenizer, eval_dataset_class, distributed_model,
                     induce=None, eval_when_train=False):
    torch.cuda.empty_cache()
    if induce is None:
        induce = eval_dataset_class.get_induce(topK=10, addGT=False)
    # return time.time()
    # if args.n_gpu > 1 and args.device == torch.device("cuda") and not hasattr(model, 'module'):
    #     distributed_model = torch.nn.DataParallel(model)
    # else:
    #     distributed_model = model

    model = model.module if hasattr(model, 'module') else model
    model.eval()
    distributed_model.eval()

    input_dict = {
        "code_inputs": eval_dataset_class.retrieval_code_base_tokened,
        "nl_inputs": eval_dataset_class.test_query_tokened,
        "device": args.device,
        "new_query_ids": eval_dataset_class.new_query_ids,
        "num_workers": args.num_workers,
        "batch_size": args.eval_batch_size,
        "induce": induce, "distributed_model": distributed_model, "eval_dataset_class":eval_dataset_class,
    }
    score, time_dict = model.predict(
        input_dict
    )
    eval_output_dict = eval_dataset_class.eval_score(score)
    mrr = eval_output_dict["mrr"]

    model.train()
    distributed_model.train()
    torch.cuda.empty_cache()

    if not eval_when_train:
        log_dir = os.path.join(args.result_log, "train", args.TrainModel)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        dataset_name = "GraphCodeBERTtrain_" + args.lang
        setting_name = add_setting(args)
        with open(os.path.join(log_dir, dataset_name+".txt"), "a") as f:
            result_str = "%s \tTopK %d \t %s MRR \t %.4f" % (
                str(time.asctime(time.localtime(time.time()))), 10, setting_name, mrr)
            f.write(result_str)
            f.write("\n")

        try:
            stage1_time = time_dict['nl_time'] + time_dict["score_time"]
            (r1, r5, r10, r100, r1000, medr, mAP) = eval_output_dict["recall_tuple"]
            result_str = str(
                time.asctime(time.localtime(time.time()))) + ("\t%s\t" + "%.4f\t" * 9) % (
                setting_name,
                             eval_output_dict["mrr"], stage1_time, 0, r1, r5, r10, r100, r1000, medr
                         )
            with open(os.path.join(log_dir, dataset_name+"_add_time_recall.txt"), "a") as f:
                f.write(result_str)
                f.write("\n")
        except Exception as e:
            logger.info(str(e))

    return mrr


def train_coclr(args):
    dataset_name = "GraphCodeBERTtrain_" + args.lang
    eval_dataset_name = "GraphCodeBERTvalid_" + args.lang
    test_dataset_name = "GraphCodeBERT_" + args.lang
    # eval_dataset_name = "GraphCodeBERT_" + args.lang
    if args.do_debug:
        dataset_name = eval_dataset_name
    train_model = args.TrainModel
    Model_zoo = {
        'CoCLR_model': ModelBinaryCreateNeg, 'CodeBert_bi-encoder': ModelCodeBertBiEncoder,
        'CodeBertP': ModelCodeBertPlus, "GraphCodeBERTMe": ModelGraphCodeBERT,
        "GraphCodeBERTOriginal1": ModelGraphCodeBERTOriginal,
        "GraphCodeBERTMultiCodeFusion": ModelGraphCodeBERTMultiCodeFusion,
        "CODEnn": CODEnn,
    }

    if '~' in args.root_path:
        args.root_path = args.root_path.replace('~', os.path.expanduser('~'))

    vectorizer_param = {'min_df': 5}
    # if args.do_debug:
    #     eval_dataset_class = GraphCodeBertStageOfficial(
    #         eval_dataset_name,
    #         code_pre_process_config=PreProcessConfig,
    #         query_pre_process_config=QueryPreProcessConfig,
    #         args=args)

    train_dataset_class = BaseStageCodeSearch(
        dataset_name,
        code_pre_process_config=PreProcessConfig,
        query_pre_process_config=QueryPreProcessConfig,
        vectorizer_param=vectorizer_param, args=args)

    if args.do_debug:
        eval_dataset_class = train_dataset_class
    else:
        eval_dataset_class = GraphCodeBertStageOfficial(
            eval_dataset_name,
            code_pre_process_config=PreProcessConfig,
            query_pre_process_config=QueryPreProcessConfig,
            args=args)
        test_dataset_class = GraphCodeBertStageOfficial(
            test_dataset_name,
            code_pre_process_config=PreProcessConfig,
            query_pre_process_config=QueryPreProcessConfig,
            args=args)

    # 获取 Dataset, Dataloader
    config_class, model_class, tokenizer_class = (RobertaConfig, RobertaModel, RobertaTokenizer)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.encoder_name_or_path,
                                                do_lower_case=True,
                                                cache_dir=None)

    param = {
        'max_length': args.code_length,
        'max_query_length': args.nl_length,
    }
    if args.TrainModel == "GraphCodeBERTMultiCodeFusion":
        dataset = CodeBertTokenTrainPairDataset1(
            train_dataset_class.test_query_tokened, train_dataset_class.retrieval_code_base_tokened,
            param=param, new_query_ids=train_dataset_class.new_query_ids)
        # if args.do_debug:
        #     dataset = CodeBertTokenTrainPairDataset1(
        #         train_dataset_class.test_query_tokened, train_dataset_class.retrieval_code_base_tokened,
        #         param=param, new_query_ids=train_dataset_class.new_query_ids)
        #     dataset.__getitem__(1)
        #     train_dataloader = torch_data.dataloader.DataLoader(
        #         dataset=dataset, batch_size=args.train_batch_size, shuffle=False,
        #         num_workers=args.num_workers, sampler=None, collate_fn=dataset.collate_fn)
        #     for each in train_dataloader:
        #         break
    else:
        dataset = CodeBertTokenTrainPairDataset(
            train_dataset_class.test_query_tokened, train_dataset_class.retrieval_code_base_tokened,
            tokenizer, param=param, new_query_ids=train_dataset_class.new_query_ids)


    if args.local_rank != -1:
        sampler = DistributedSampler(dataset, shuffle=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True
    train_dataloader = torch_data.dataloader.DataLoader(
        dataset=dataset, batch_size=args.train_batch_size, shuffle=shuffle,
        num_workers=args.num_workers, sampler=sampler, collate_fn=dataset.collate_fn)


    # Prepare model, optimizer and schedule (linear warmup and decay)
    config = config_class.from_pretrained(args.encoder_name_or_path)
    if args.TrainModel == "GraphCodeBERTMe":
        model = RobertaModel.from_pretrained(args.encoder_name_or_path)
    else:
        config.num_labels = 2
        model = model_class(config)
        model = model.from_pretrained(args.encoder_name_or_path)
    model = Model_zoo[train_model](model, config, tokenizer, args)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=len(train_dataloader) * args.num_train_epochs)
    model.to(args.device)
    setting_name = add_setting(args)

    # 进行训练
    best_mrr = 0
    checkpoint_prefix = 'checkpoint-best-mrr'
    output_dir = os.path.join(args.root_path, "code_search_baseline_data", train_model, setting_name, dataset_name)
    output_dir = os.path.join(output_dir, '{}'.format(checkpoint_prefix))
    logger.info("output_dir:%s"%output_dir)
    if os.path.exists(os.path.join(output_dir, '{}'.format('model.best.bin'))) and (not args.overwrite):
        # load best model, test, return
        # return
        model_to_save_dir = os.path.join(output_dir, '{}'.format('model.best.bin'))
        logger.info("Test model: load %s" % model_to_save_dir)
        model.load_state_dict(torch.load(model_to_save_dir, map_location=args.device))
        mrr = eval_coclr_model(args, model, tokenizer, test_dataset_class, model, induce=None)
        return

    if os.path.exists(os.path.join(output_dir, '{}'.format('model.temp.bin'))):
        # load last model
        model_to_save_dir = os.path.join(output_dir, '{}'.format('model.temp.bin'))
        try:
            model.load_state_dict(torch.load(model_to_save_dir, map_location="cpu"))
            best_mrr = torch.load(os.path.join(output_dir, '{}'.format('info_dict.bin')), map_location="cpu")['mrr']
        except Exception as e:
            print(e)

        logger.info("load trained model %s \n best mrr: %.4f" % (model_to_save_dir, best_mrr))

    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except Exception as e:
            print(e)

    global writer  # 如果在其他函数中还需要引用可以这样写
    writer = SummaryWriter(log_dir=output_dir)  # 设置记录位置

    model.train()
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
        find_unused_parameters=True)
    elif args.n_gpu > 1 and args.device == torch.device("cuda"):
        model = torch.nn.DataParallel(model)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    for idx in range(0, int(args.num_train_epochs)):
        logger.info("Epoch %d" % idx)
        with tqdm(total=len(train_dataloader)) as phar:
            for dataloader_index, batch in enumerate(train_dataloader):
                if dataloader_index == 8 and args.do_debug:
                    break
                if args.TrainModel == "GraphCodeBERTMultiCodeFusion":
                    code_inputs = batch[1].to(args.device)
                    nl_inputs = batch[0].to(args.device)
                    code_len_tuple = batch[2]
                    loss_dict, predictions = model(
                        code_inputs=code_inputs, code_len_tuple=code_len_tuple,
                          nl_inputs=nl_inputs)
                else:
                    code_inputs = batch[1].to(args.device)
                    nl_inputs = batch[0].to(args.device)
                    # ds_inputs = batch[2].to(args.device)
                    labels = batch[2].to(args.device)
                    loss_dict, predictions = model(code_inputs=code_inputs, nl_inputs=nl_inputs)

                loss = loss_dict["all"]
                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                phar.update(1)
                # phar.set_description("Loss: %.4f, loss_negative: %.4f, loss_positive: %.4f"
                #                      % (float(loss.mean()), float(loss_dict['loss_negative'].mean()),
                #                         float(loss_dict['loss_positive'].mean())))
                phar.set_description("Loss: %.4f" % (float(loss.mean())))
                if (dataloader_index+1) % 100 == 0:
                    pass
                    for each_key in loss_dict:
                        each_loss = loss_dict[each_key].mean().detach().cpu()
                        writer.add_scalar('train/'+each_key,
                                          float(each_loss), idx*len(train_dataloader)+dataloader_index)


                # backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

        if args.local_rank in [0, -1]:
            # Save temp model
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save_dir = os.path.join(output_dir, '{}'.format('model.temp.bin'))
            torch.save(model_to_save.state_dict(), model_to_save_dir)

            mrr = eval_coclr_model(args, model, tokenizer, eval_dataset_class, model, induce=None,
                                   eval_when_train=True)
            logger.info("mrr = %.4f" % mrr)
            writer.add_scalar('val/mrr', float(mrr), idx)
            with open(os.path.join(output_dir,"log.txt"), "a") as f_log:
                f_log.write("mrr = %.4f, Time %s\n" % (mrr, str(time.asctime(time.localtime(time.time())))))

            # Save temp best model
            if mrr > best_mrr:
                best_mrr = mrr
                logger.info("  " + "*" * 20)
                logger.info("  Best mrr:%s", round(best_mrr, 4))
                logger.info("  " + "*" * 20)

                model_to_save = model.module if hasattr(model, 'module') else model
                model_to_save_dir = os.path.join(output_dir, '{}'.format('model.temp.best.bin'))
                torch.save(model_to_save.state_dict(), model_to_save_dir)
                logger.info("Saving model checkpoint to %s", model_to_save_dir)
                info_dict = {
                    "mrr": mrr, "args": args
                }
                torch.save(info_dict, os.path.join(output_dir, '{}'.format('info_dict.bin')))

    if args.local_rank in [0, -1]:
        # Rename to best model
        model_to_save_dir = os.path.join(output_dir, '{}'.format('model.temp.best.bin'))
        os.rename(model_to_save_dir, model_to_save_dir.replace('model.temp.best.bin', 'model.best.bin'))
        if not args.save_temp:
            # remove model.temp.bin
            os.remove(os.path.join(output_dir, '{}'.format('model.temp.bin')))

        model = model.module if hasattr(model, 'module') else model
        model.load_state_dict(torch.load(model_to_save_dir.replace('model.temp.best.bin', 'model.best.bin'),
                                         map_location=args.device))
        eval_coclr_model(args, model, tokenizer, test_dataset_class, model, induce=None,
                               eval_when_train=False)

    return


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = "5"
        sys.argv = "train_Graphcodebert.py " \
                   "--num_workers 1 --train_batch_size 8 --TrainModel CODEnn " \
                   "--encoder_name_or_path microsoft/graphcodebert-base " \
                   "--device cuda --learning_rate 2e-5 --ClassifyLoss bce --max_neg --lang ruby " \
                   "--seed 0 ".split()

        # GraphCodeBERT
        # sys.argv = "train_Graphcodebert.py " \
        #            "--num_workers 1 --train_batch_size 2 --TrainModel GraphCodeBERTMe " \
        #            "--encoder_name_or_path microsoft/graphcodebert-base " \
        #            "--device cuda:1 --learning_rate 2e-5 --lang ruby " \
        #            "--seed 0 --do_debug --window_setting WindowSize_-1,step_25".split()

        # sys.argv = "train_Graphcodebert.py " \
        #            "--num_workers 2 --train_batch_size 8 --TrainModel GraphCodeBERTMultiCodeFusion " \
        #            "--encoder_name_or_path microsoft/graphcodebert-base " \
        #            "--device cuda --learning_rate 2e-5 --lang ruby " \
        #            "--seed 0 --do_debug --window_setting WindowSize_30,step_15 " \
        #            "--split_type line --AttentionCommonType meanpooling".split()

        # For ast split
        sys.argv = "train_Graphcodebert.py " \
                   "--num_workers 2 --train_batch_size 8 --TrainModel GraphCodeBERTMultiCodeFusion " \
                   "--encoder_name_or_path microsoft/graphcodebert-base " \
                   "--device cuda --learning_rate 2e-5 --lang ruby " \
                   "--seed 0 --do_debug --window_setting WindowSize_8,step_4 " \
                   "--split_type ast_subtree --AttentionCommonType meanpooling --min_line 1".split()

        # sys.argv = "train_Graphcodebert.py " \
        #            "--num_workers 1 --train_batch_size 32 --TrainModel CodeBertP " \
        #            "--device cuda --ClassifyLoss bce --lang python --max_neg " \
        #            "--seed 0".split()

    args = get_train_args()
    print(args.lang)
    train_coclr(args)
