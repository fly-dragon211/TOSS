
import argparse
import logging
import os
import pdb
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"

import pickle
import random
import torch
import json
import numpy as np
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)

from tqdm import tqdm, trange
import multiprocessing
from parser_graphCodeBert import DFG_python, DFG_java, DFG_ruby, DFG_go, DFG_php, DFG_javascript
from parser_graphCodeBert import (remove_comments_and_docstrings,
                    tree_to_token_index, tree_to_root_index,
                    index_to_code_token,
                    tree_to_variable_index)
from tree_sitter import Language, Parser
from two_stage_utils import sliding_window
import evaluation
import time


logger = logging.getLogger(__name__)

cpu_cont = 16

dfg_function = {
    'python': DFG_python,
    'java': DFG_java,
    'ruby': DFG_ruby,
    'go': DFG_go,
    'php': DFG_php,
    'javascript': DFG_javascript
}

# load parsers
parsers = {}
for lang in dfg_function:
    LANGUAGE = Language('parser_graphCodeBert/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser, dfg_function[lang]]
    parsers[lang] = parser


def extract_ast_subcode(raw_code, parser, lang, max_depth=4, min_line=5, remove_comments=True):
    """
    按照 AST 解析来抽取 subcode
    """
    # remove comments
    try:
        if remove_comments:
            raw_code = remove_comments_and_docstrings(raw_code, lang)
    except:
        pass
    # obtain dataflow
    if lang == "php":
        raw_code = "<?php" + raw_code + "?>"
    try:
        tree = parser[0].parse(bytes(raw_code, 'utf8'))
        root_node = tree.root_node
        code = raw_code.split('\n')
        start_index = tree_to_root_index(root_node, 0, max_depth=max_depth, min_line=min_line)

        subcode_list = []
        if len(start_index) > 1:
            for i in range(1, len(start_index)):
                subcode_list.append("\n".join(code[start_index[i-1][0]:start_index[i][0]])
                                    + code[start_index[i][0]][0:start_index[i][1]])
        else:
            subcode_list = [raw_code]

    except Exception as e:
        print(e)
        subcode_list = [raw_code]

    return subcode_list

# remove comments, tokenize code and extract dataflow
def extract_dataflow(code, parser, lang, remove_comments=True):
    # remove comments
    try:
        if remove_comments:
            code = remove_comments_and_docstrings(code, lang)
    except:
        pass
    # obtain dataflow
    if lang == "php":
        code = "<?php" + code + "?>"
    try:
        tree = parser[0].parse(bytes(code, 'utf8'))
        root_node = tree.root_node
        tokens_index = tree_to_token_index(root_node)
        code = code.split('\n')
        code_tokens = [index_to_code_token(x, code) for x in tokens_index]
        # tokens_index = tree_to_root_index(root_node, 0)
        index_to_code = {}
        for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx, code)
        try:
            DFG, _ = parser[1](root_node, index_to_code, {})
        except:
            DFG = []
        DFG = sorted(DFG, key=lambda x: x[1])
        indexs = set()
        for d in DFG:
            if len(d[-1]) != 0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG = []
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg = new_DFG
    except:
        dfg = []
        print(code)
        raise Exception("Code error")
    return code_tokens, dfg


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


def convert_examples_to_features(item):
    js, tokenizer, args = item
    # *********************萌萌哒***********************
    # code
    parser = parsers[args.lang]
    # extract data flow
    code_tokens, dfg = extract_dataflow(js['original_string'], parser, args.lang)
    code_tokens = [tokenizer.tokenize('@ ' + x)[1:] if idx != 0 else tokenizer.tokenize(x) for idx, x in
                   enumerate(code_tokens)]
    ori2cur_pos = {}
    ori2cur_pos[-1] = (0, 0)
    for i in range(len(code_tokens)):
        ori2cur_pos[i] = (ori2cur_pos[i - 1][1], ori2cur_pos[i - 1][1] + len(code_tokens[i]))
    code_tokens = [y for x in code_tokens for y in x]
    # truncating
    code_tokens = code_tokens[:args.code_length + args.data_flow_length - 2 - min(len(dfg), args.data_flow_length)]
    code_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    position_idx = [i + tokenizer.pad_token_id + 1 for i in range(len(code_tokens))]
    dfg = dfg[:args.code_length + args.data_flow_length - len(code_tokens)]
    code_tokens += [x[0] for x in dfg]
    position_idx += [0 for x in dfg]
    code_ids += [tokenizer.unk_token_id for x in dfg]
    padding_length = args.code_length + args.data_flow_length - len(code_ids)
    position_idx += [tokenizer.pad_token_id] * padding_length
    code_ids += [tokenizer.pad_token_id] * padding_length
    # reindex
    reverse_index = {}
    for idx, x in enumerate(dfg):
        reverse_index[x[1]] = idx
    for idx, x in enumerate(dfg):
        dfg[idx] = x[:-1] + ([reverse_index[i] for i in x[-1] if i in reverse_index],)
    dfg_to_dfg = [x[-1] for x in dfg]
    dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
    length = len([tokenizer.cls_token])
    dfg_to_code = [(x[0] + length, x[1] + length) for x in dfg_to_code]
    # *********************萌萌哒***********************
    # nl
    nl = ' '.join(js['docstring_tokens'])
    nl_tokens = tokenizer.tokenize(nl)[:args.nl_length - 2]
    nl_tokens = [tokenizer.cls_token] + nl_tokens + [tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id] * padding_length

    return InputFeatures(code_tokens, code_ids, position_idx, dfg_to_code, dfg_to_dfg, nl_tokens, nl_ids, js['url'])


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None, pool=None):
        self.args = args
        if not hasattr(args, "output_dir"):
            args.output_dir = os.path.join(
                args.root_path, "TwoStageModels", 'GraphCodeBERT', 'models', '%s' % (args.lang))
            args.data_flow_length = 64
            pass

        prefix = file_path.split('/')[-1][:-6]
        if self.args.code_length != 256:
            prefix = prefix + "-code_length_%d" % self.args.code_length
        if self.args.nl_length != 128:
            prefix = prefix + "-nl_length_%d" % self.args.nl_length
        cache_file = args.output_dir + '/' + prefix + '.pkl'
        if os.path.exists(cache_file):
        # if os.path.exists(cache_file):
            self.examples = pickle.load(open(cache_file, 'rb'))
        else:
            self.examples = []
            data = []
            with open(file_path) as f:
                for line in f:
                    line = line.strip()
                    js = json.loads(line)
                    data.append((js, tokenizer, args))
            self.examples = pool.map(convert_examples_to_features, tqdm(data, total=len(data)))
            pickle.dump(self.examples, open(cache_file, 'wb'))

        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("code_tokens: {}".format([x.replace('\u0120', '_') for x in example.code_tokens]))
                logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
                logger.info("position_idx: {}".format(example.position_idx))
                logger.info("dfg_to_code: {}".format(' '.join(map(str, example.dfg_to_code))))
                logger.info("dfg_to_dfg: {}".format(' '.join(map(str, example.dfg_to_dfg))))
                logger.info("nl_tokens: {}".format([x.replace('\u0120', '_') for x in example.nl_tokens]))
                logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        # calculate graph-guided masked function
        attn_mask = np.zeros((self.args.code_length + self.args.data_flow_length,
                              self.args.code_length + self.args.data_flow_length), dtype=bool)
        # calculate begin index of node and max length of input
        node_index = sum([i > 1 for i in self.examples[item].position_idx])
        max_length = sum([i != 1 for i in self.examples[item].position_idx])
        # sequence can attend to sequence
        attn_mask[:node_index, :node_index] = True

        # special tokens attend to all tokens
        for idx, i in enumerate(self.examples[item].code_ids):
            if i in [0, 2]:
                attn_mask[idx, :max_length] = True
        # nodes attend to code tokens that are identified from
        for idx, (a, b) in enumerate(self.examples[item].dfg_to_code):
            if a < node_index and b < node_index:
                attn_mask[idx + node_index, a:b] = True
                attn_mask[a:b, idx + node_index] = True
        # nodes attend to adjacent nodes
        for idx, nodes in enumerate(self.examples[item].dfg_to_dfg):
            for a in nodes:
                if a + node_index < len(self.examples[item].position_idx):
                    attn_mask[idx + node_index, a + node_index] = True

        return (torch.tensor(self.examples[item].code_ids),
                torch.tensor(attn_mask),
                torch.tensor(self.examples[item].position_idx),
                torch.tensor(self.examples[item].nl_ids))


class TextDatasetSplit(Dataset):
    """
    - 把 nl 和 code 分开编码
    - 加入划分窗口代码
    """
    def __init__(self, tokenizer, args, file_path=None, pool=None):
        self.args = args
        if not hasattr(args, "output_dir"):
            args.output_dir = os.path.join(
                args.root_path, "TwoStageModels", 'GraphCodeBERT', 'models', '%s' % (args.lang))
            pass
        if not hasattr(args, "data_flow_length"):
            args.data_flow_length = 64

        prefix = file_path.split('/')[-1][:-6]
        if self.args.code_length != 256:
            prefix = prefix + "-code_length_%d" % self.args.code_length
        if self.args.nl_length != 128:
            prefix = prefix + "-nl_length_%d" % self.args.nl_length

        cache_file = args.output_dir + '/' + prefix + '.pkl'
        if os.path.exists(cache_file) and (not args.do_debug):
        # if os.path.exists(cache_file):
            self.examples = pickle.load(open(cache_file, 'rb'))
        else:
            self.examples = []
            data = []
            with open(file_path) as f:
                for line in f:
                    line = line.strip()
                    js = json.loads(line)
                    data.append((js, tokenizer, args))
            # ***********************萌萌哒**************************
            # cdsaaae===- =-= --095304 003535
            window_size, step = self.args.window_size, self.args.step
            if window_size > 0:
                # 仅仅给要检索的 query 得 gt 划分 multi-passages
                new_query_ids = [[each] for each in self.query_ids]
                data_len = len(data)
                for data_index in range(len(data_len)):
                    code = data[data_index][0]['code']
                    code_list = code.split(" ")
                    code_subset_list = [" ".join(each[1]) for each in sliding_window(code_list, window_size, step)]

                    data[data_index][0]['code'] = code_subset_list[0]
                    data[data_index][0]['code'] = code_subset_list[0]

                    pass
            # ***********************萌萌哒**************************
            self.examples = pool.map(convert_examples_to_features, tqdm(data, total=len(data)))
            pickle.dump(self.examples, open(cache_file, 'wb'))

        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("code_tokens: {}".format([x.replace('\u0120', '_') for x in example.code_tokens]))
                logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
                logger.info("position_idx: {}".format(example.position_idx))
                logger.info("dfg_to_code: {}".format(' '.join(map(str, example.dfg_to_code))))
                logger.info("dfg_to_dfg: {}".format(' '.join(map(str, example.dfg_to_dfg))))
                logger.info("nl_tokens: {}".format([x.replace('\u0120', '_') for x in example.nl_tokens]))
                logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        # calculate graph-guided masked function
        attn_mask = np.zeros((self.args.code_length + self.args.data_flow_length,
                              self.args.code_length + self.args.data_flow_length), dtype=bool)
        # calculate begin index of node and max length of input
        node_index = sum([i > 1 for i in self.examples[item].position_idx])
        max_length = sum([i != 1 for i in self.examples[item].position_idx])
        # sequence can attend to sequence
        attn_mask[:node_index, :node_index] = True

        # special tokens attend to all tokens
        for idx, i in enumerate(self.examples[item].code_ids):
            if i in [0, 2]:
                attn_mask[idx, :max_length] = True
        # nodes attend to code tokens that are identified from
        for idx, (a, b) in enumerate(self.examples[item].dfg_to_code):
            if a < node_index and b < node_index:
                attn_mask[idx + node_index, a:b] = True
                attn_mask[a:b, idx + node_index] = True
        # nodes attend to adjacent nodes
        for idx, nodes in enumerate(self.examples[item].dfg_to_dfg):
            for a in nodes:
                if a + node_index < len(self.examples[item].position_idx):
                    attn_mask[idx + node_index, a + node_index] = True

        return (torch.tensor(self.examples[item].code_ids),
                torch.tensor(attn_mask),
                torch.tensor(self.examples[item].position_idx),
                torch.tensor(self.examples[item].nl_ids))



class Model(torch.nn.Module):
    def __init__(self, encoder):
        super(Model, self).__init__()
        self.encoder = encoder

    def forward(self, code_inputs=None, attn_mask=None, position_idx=None, nl_inputs=None):
        if code_inputs is not None:
            nodes_mask = position_idx.eq(0)
            token_mask = position_idx.ge(2)
            inputs_embeddings = self.encoder.embeddings.word_embeddings(code_inputs)

            nodes_to_token_mask = nodes_mask[:, :, None] & token_mask[:, None, :] & attn_mask
            nodes_to_token_mask = nodes_to_token_mask / (nodes_to_token_mask.sum(-1) + 1e-10)[:, :, None]
            avg_embeddings = torch.einsum("abc,acd->abd", nodes_to_token_mask, inputs_embeddings)
            inputs_embeddings = inputs_embeddings * (~nodes_mask)[:, :, None] + avg_embeddings * nodes_mask[:, :, None]
            # import pdb; pdb.set_trace()

            return self.encoder(inputs_embeds=inputs_embeddings, attention_mask=attn_mask, position_ids=position_idx)[1]
        else:
            return self.encoder(nl_inputs, attention_mask=nl_inputs.ne(1))[1]


class GraphCodeBert(object):
    def __init__(self, model_suffix="", lang="python", root_path=None, input_args=None, dataset="GraphCodeBERT_python"):
        super(GraphCodeBert, self).__init__()
        self.dataset = dataset
        if root_path is None:
            root_path = "~/VisualSearch".replace('~', os.path.expanduser('~'))

        store_root_path = os.path.join(root_path, "TwoStageModels", self.dataset)

        model_dir = os.path.join(
            root_path, "TwoStageModels", 'GraphCodeBERT', 'models', '%s%s' % (lang, model_suffix))

        if "valid" in dataset:
            eval_data_file = os.path.join(root_path, "GraphCodeBERT_dataset", lang, 'valid.jsonl')
        else:
            eval_data_file = os.path.join(root_path, "GraphCodeBERT_dataset", lang, 'test.jsonl')
        codebase_file = os.path.join(root_path, "GraphCodeBERT_dataset", lang, 'codebase.jsonl')

        logger.info("eval_data_file: %s" % eval_data_file)

        self.test_query = []
        self.query_ids = []
        self.retrieval_code_base = []
        self.retrieval_raw_code_base = []
        self.code_ids = []

        if input_args is None:
            input_args = argparse.Namespace()
            input_args.code_length = 256
            input_args.nl_length = 128
            input_args.n_gpu = torch.cuda.device_count()
            if input_args.n_gpu >= 1:
                input_args.device = torch.device("cuda")
            else:
                input_args.device = torch.device("cpu")

        args = argparse.Namespace(code_length=input_args.code_length,
                                  codebase_file=codebase_file,
                                  config_name='microsoft/graphcodebert-base', data_flow_length=64,
                                eval_batch_size=256, eval_data_file=eval_data_file,
                                  lang=lang, n_gpu=input_args.n_gpu, device=input_args.device,
                                  model_name_or_path='microsoft/graphcodebert-base',
                                  nl_length=input_args.nl_length,
                                  output_dir=model_dir,
                                  store_root_path=store_root_path,
                                  do_debug=input_args.do_debug,
                                  tokenizer_name='microsoft/graphcodebert-base')

        self.args = args
        tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
        pool = multiprocessing.Pool(cpu_cont)

        query_dataset = TextDataset(tokenizer, args, args.eval_data_file, pool)
        query_sampler = SequentialSampler(query_dataset)
        self.query_dataloader = DataLoader(query_dataset, sampler=query_sampler, batch_size=args.eval_batch_size,
                                      num_workers=1, shuffle=False)

        code_dataset = TextDataset(tokenizer, args, args.codebase_file, pool)
        code_sampler = SequentialSampler(code_dataset)
        self.code_dataloader = DataLoader(code_dataset, sampler=code_sampler, batch_size=args.eval_batch_size,
                                          num_workers=1, shuffle=False)

        self.tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
        pretrained_path = os.path.join(model_dir, "checkpoint-best-mrr/model.bin")
        self.model = Model(RobertaModel.from_pretrained(args.model_name_or_path))
        logger.info("Read %s" % pretrained_path)
        self.model.load_state_dict(torch.load(pretrained_path, map_location=torch.device('cpu')), strict=True)
        self.model.to(args.device)

    def encode_text_and_code(self, read_scores=True, neg_candidate=-1):
        args = self.args

        # store_score_path = os.path.join(self.args.output_dir, os.path.basename(self.args.eval_data_file)+".scores.pkl")
        # if read_scores and os.path.exists(store_score_path):
        #     logger.info("Read "+store_score_path)
        #     return pickle.load(open(store_score_path, 'rb'))

        feature_data_path = os.path.join(
            args.store_root_path, "FeatureData",
            "CSN-"+args.lang)
        if read_scores and os.path.exists(os.path.join(feature_data_path, "GraphCodeBERT.pkl")):
            logger.info("Read "+feature_data_path)
            feature_data_path_name = os.path.join(feature_data_path, "GraphCodeBERT.pkl")
            feature_dict = pickle.load(open(feature_data_path_name, 'rb'))
            nl_vecs, code_vecs = feature_dict["query"].float(), feature_dict["database"].float()

            if torch.cuda.device_count() >= 1:
                time_path1 = feature_data_path_name.replace(".pkl", "") + "-time_gpu.pkl"
            else:
                time_path1 = feature_data_path_name.replace(".pkl", "") + "-time.pkl"
            time_dict1 = pickle.load(open(time_path1, 'rb'))
            cal_query_time = time_dict1["all_query_time"]

            start_time = time.time()
            scores = nl_vecs.mm(code_vecs.T)
            end_time = time.time()
            stage_time = end_time - start_time
            graphcodebert_time = cal_query_time + stage_time
            if neg_candidate != -1:
                graphcodebert_time = cal_query_time + neg_candidate/len(code_vecs)*stage_time

            return scores, graphcodebert_time

        model = self.model

        query_dataloader = self.query_dataloader
        code_dataloader = self.code_dataloader

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        # Eval!
        logger.info("***** Running GraphCodeBERT evaluation *****")
        logger.info("  Num queries = %d", len(query_dataloader.dataset))
        logger.info("  Num codes = %d", len(code_dataloader.dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)

        model.eval()
        # Encode raw query and query ids
        code_vecs = []
        nl_vecs = []
        with tqdm(total=len(query_dataloader.dataset)) as pbar:
            start_time = time.time()
            for i_q, batch in enumerate(query_dataloader):
                if i_q == 0:
                    start_time = time.time()
                pbar.update(len(batch[0]))
                nl_inputs = batch[3].to(args.device)
                with torch.no_grad():
                    nl_vec = model(nl_inputs=nl_inputs)
                    nl_vecs.append(nl_vec.cpu())
                    # break
            all_query_time = time.time()-start_time
            start_time = time.time()
            with torch.no_grad():
                nl_vec = model(nl_inputs=nl_inputs[[0]])
            one_query_time = time.time()-start_time


        with tqdm(total=len(code_dataloader.dataset)) as pbar:
            for batch in code_dataloader:
                pbar.update(len(batch[0]))
                code_inputs = batch[0].to(args.device)
                attn_mask = batch[1].to(args.device)
                position_idx = batch[2].to(args.device)
                with torch.no_grad():
                    code_vec = model(code_inputs=code_inputs, attn_mask=attn_mask, position_idx=position_idx)
                    code_vecs.append(code_vec.cpu())
                    # break

        code_vecs = torch.cat(code_vecs, 0)
        nl_vecs = torch.cat(nl_vecs, 0)
        start_time = time.time()
        scores = nl_vecs.mm(code_vecs.T)
        end_time = time.time()
        stage_time = end_time-start_time

        # pdb.set_trace()

        # *****************************萌萌哒*******************************
        # 保存特征文件，以及 query 计算时间
        if not os.path.exists(feature_data_path):
            os.makedirs(feature_data_path)


        feature_data_path_name = os.path.join(feature_data_path, "GraphCodeBERT.pkl")
        logger.info("Restore to %s" % feature_data_path_name)
        feature_dict = {
            "query": nl_vecs, "database": code_vecs
        }
        if read_scores:
            pickle.dump(feature_dict, open(feature_data_path_name, 'wb'))

        if args.n_gpu >= 1:
            feature_time_path_name = os.path.join(feature_data_path, "GraphCodeBERT-time_gpu.pkl")
            one_query_time = args.n_gpu*one_query_time
            all_query_time = args.n_gpu*all_query_time

        else:
            feature_time_path_name = os.path.join(feature_data_path, "GraphCodeBERT-time.pkl")
        time_dict = {"all_query_time": all_query_time,
            "one_query_time": one_query_time}
        if read_scores:
            pickle.dump(time_dict, open(feature_time_path_name, 'wb'))

        # *****************************萌萌哒*******************************

        # self.code_ids = list(np.arange(0, len(self.retrieval_code_base)))
        # self.query_ids = []
        # codeurl_to_int = {}
        # for i, each in enumerate(code_dataloader.dataset.examples):
        #     if each.url in codeurl_to_int:
        #         print("存在重复: ", each['url'])
        #     codeurl_to_int[each['url']] = i
        # query_urls = []
        # for example in query_dataloader.dataset.examples:
        #     query_urls.append(example.url)
        # for each in query_urls:
        #     self.query_ids.append(codeurl_to_int[each])

        # *****************************萌萌哒*******************************

        # sort_ids = np.argsort(scores.numpy(), axis=-1, order=None)[:, ::-1]
        # nl_urls = []
        # code_urls = []
        # for example in query_dataloader.dataset.examples:
        #     nl_urls.append(example.url)
        #
        # for example in code_dataloader.dataset.examples:
        #     code_urls.append(example.url)
        # ranks = []
        # for url, sort_id in zip(nl_urls, sort_ids):
        #     rank = 0
        #     find = False
        #     for idx in sort_id[:1000]:
        #         if find is False:
        #             rank += 1
        #         if code_urls[idx] == url:
        #             find = True
        #     if find:
        #         ranks.append(1 / rank)
        #     else:
        #         ranks.append(0)
        # print("eval_mrr", float(np.mean(ranks)))

        # ****************************萌萌哒********************************

        return scores, all_query_time+stage_time


if __name__ == "__main__":
    a = GraphCodeBert(model_suffix="")
    a.encode_text_and_code(read_scores=True)
