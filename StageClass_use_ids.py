import torch
import torch.nn as nn
import torch.utils.data as torch_data
import re
import os
import logging
import pickle
import time
import random
import argparse
import multiprocessing
import pdb

import json
import numpy as np
import copy
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords
import nltk
import evaluation
from tqdm import tqdm
# 词性还原
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

import transformers
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel, RobertaForSequenceClassification

from spiral import ronin
from CoCLR_train.models import ModelBinaryCreateNeg, ModelCodeBertBiEncoder, ModelGraphCodeBERTMultiCodeFusion, \
    ModelCodeBertPlus
from GraphCodeBERT import GraphCodeBert, InputFeatures, parsers, extract_dataflow, extract_ast_subcode
from two_stage_utils import sliding_window


logger = logging.getLogger(__name__)
download = True
padding_strategy = 'max_length'


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def convert_examples_to_features(item):
    """js, tokenizer, args, return_item="code_string_noDFG"
    return_item:
    - code_token
    - query_token
    - all
    """
    js_original_string, extracted_string_token, js_docstring_tokens, js_url, tokenizer, args, return_item = item
    return_items = ["all", "code_token", "query_token",  # 0~2
                    "code_string", "query_string", "code_string_noDFG",  # 3~5
                    ]
    if not hasattr(args, "data_flow_length"):
        args.data_flow_length = 64
    if not hasattr(args, "no_comments"):
        args.remove_comments = True
    if return_item in [return_items[0], return_items[1], return_items[3], return_items[5], ]:
        # *********************萌萌哒***********************
        # code
        parser = parsers[args.lang]

        # extract data flow
        dfg = []
        if extracted_string_token is None:  # extracted_string_token 不是 None，后面就没有 dfg 了
            extracted_string_token, dfg = extract_dataflow(js_original_string, parser, args.lang,
                                                           remove_comments=args.remove_comments)
        else:
            pass
        code_tokens = extracted_string_token.copy()
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
        if return_item == "code_string_noDFG":  # "code_string_noDFG"
            code_string = tokenizer.convert_tokens_to_string(code_tokens[1:-1])  # 去掉 cls, end
            return code_string

        code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
        position_idx = [i + tokenizer.pad_token_id + 1 for i in range(len(code_tokens))]
        dfg = dfg[:args.code_length + args.data_flow_length - len(code_tokens)]
        code_tokens += [x[0] for x in dfg]
        if return_item == return_items[3]:  # "code_string"
            code_string = tokenizer.convert_tokens_to_string(code_tokens[1:-1])  # 去掉 cls, end
            return code_string

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

    if return_item in [return_items[0], return_items[2], return_items[4], ]:
        # *********************萌萌哒***********************
        # nl
        nl = ' '.join(js_docstring_tokens)
        nl_tokens = tokenizer.tokenize(nl)[:args.nl_length - 2]
        nl_tokens = [tokenizer.cls_token] + nl_tokens + [tokenizer.sep_token]
        nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
        padding_length = args.nl_length - len(nl_ids)
        nl_ids += [tokenizer.pad_token_id] * padding_length

        if return_item == 'query_string':
            nl_string = tokenizer.convert_tokens_to_string(nl_tokens[1:-1])  # 去掉 cls, end
            return nl_string

    output_dict = {
        "code_tokens": code_tokens, "code_ids": code_ids, "position_idx": position_idx,
        "dfg_to_code": dfg_to_code, "dfg_to_dfg": dfg_to_dfg, "nl_tokens": nl_tokens,
        "nl": nl, "nl_ids": nl_ids, "js_url": js_url, "js_original_string": js_original_string,
        "extracted_string_token": extracted_string_token,

    }
    return output_dict


class TextPreProcess:
    def __init__(self, language="python"):
        global download
        if download:
            nltk.download('wordnet')
            nltk.download('punkt')
            nltk.download('averaged_perceptron_tagger')
            nltk.download('omw-1.4')
            nltk.download('stopwords')
        download=False
        self.language = language
        pass

    def pascal_case_to_snake_case(self, camel_case: str):
        """大驼峰（帕斯卡）转蛇形 split pascal case and snake case"""
        snake_case = re.sub(r"(?P<key>[A-Z][a-z])", r"_\g<key>", camel_case)
        snake_case = re.sub("_", " ", snake_case)
        # snake_case = re.sub(r"(?P<key>[A-Z])", r"_\g<key>", camel_case)
        return snake_case.lower().strip('_')

    def delcommonds(self, content: str):
        """Delete the python comments"""
        count = 1  # 替换个数
        out = content
        if self.language == "python":
            out = re.sub(r'""".*?"""', ' ', content, flags=re.S, count=count)
            # out = re.sub(r'(##.*?\n)', ' ', out)
        elif self.language in {"java", "javascript", "go", "php"}:
            # out = re.sub(r'/\*{1,2}[\s\S]*?\*/', ' ', content, count=count)
            # out = re.sub(r'(\\.*?\n)', ' \n', out)
            pass
            if self.language == "php":
                # out = re.sub(r'(##.*?\n)', ' ', out)
                pass
        elif self.language in {"ruby"}:
            # out = re.sub(r'=begin.*?=end', ' ', content, flags=re.S, count=count)
            # out = re.sub(r'(##.*?\n)', ' ', out)
            pass
        else:
            raise Exception("Not implement!")
        # out = re.sub(r'(#.*?\n)', ' ', out)
        return out

    def no_punctuation(self, sentence: str):
        """Delete the punctuations and digit"""
        sentence = re.sub(r'[^\w\s]', ' ', sentence)
        sentence = re.sub(r'[0-9]+', ' ', sentence)
        return sentence

    # 获取单词的词性
    def get_wordnet_pos(self, tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    def Lemmatizer(self, sentence: str):
        tokens = word_tokenize(sentence)  # 分词
        tagged_sent = pos_tag(tokens)  # 获取单词词性

        wnl = WordNetLemmatizer()
        lemmas_sent = []
        for tag in tagged_sent:
            wordnet_pos = self.get_wordnet_pos(tag[1]) or wordnet.NOUN
            lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))  # 词形还原
        return " ".join(lemmas_sent)

    def clip_to_8(self, s):
        s_list = s.split()
        new_s_list = []
        for i in range(len(s_list)):
            if len(s_list[i]) > 8:
                continue
            new_s_list.append(s_list[i])
        return " ".join(new_s_list)

    def split_with_spiral_for_8(self, string):
        """
        对于长度大于8的单词拆分
        """
        s_list = string.split()
        new_s_list = []
        for i in range(len(s_list)):
            if len(s_list[i]) >= 8:
                new_s_list.extend(ronin.split(s_list[i]))
            else:
                new_s_list.append(s_list[i])
        return " ".join(new_s_list)

    def del_redundant_spaces(self, string):
        """
        remove redundant spaces
        """
        out = re.sub(' +', " ", string)
        return out


class BaseStageCodeSearch:
    def __init__(self, dataset='coclr',
                 code_pre_process_config=[], query_pre_process_config=[], vectorizer_param=None, args=None):
        self.dataset = dataset
        if args is None:
            args = argparse.Namespace(num_workers=16)
            if torch.cuda.device_count() >= 1:
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
            args.device = self.device
            args.eval_batch_size = 256
            args.root_path = "~/VisualSearch".replace('~', os.path.expanduser('~'))
        if not hasattr(args, "neg_candidate"):
            args.neg_candidate = -1  # 默认选择所有 codebase codes 作为 candidate.

        self.args = args
        self.device = args.device
        self.tokenizer = RobertaTokenizer.from_pretrained(args.encoder_name_or_path)

        self.code_pre_process_config = code_pre_process_config
        self.query_pre_process_config = query_pre_process_config
        if not self.code_pre_process_config:
            self.code_pre_process_config = ['to_snake_case', 'no_comment', 'no_punctuation']
        if not self.query_pre_process_config:
            # self.query_pre_process_config = ['to_snake_case', 'no_punctuation']
            self.query_pre_process_config = ['None']

        if vectorizer_param is None:
            self.vectorizer_param = {'min_df': 5, "stop_words": stopwords.words('english')}
        else:
            self.vectorizer_param = vectorizer_param
            if "stop_words" not in vectorizer_param:
                self.vectorizer_param["stop_words"] = stopwords.words('english')

        self.text_precess = TextPreProcess()
        self.test_query = []
        self.query_ids = []
        self.retrieval_code_base = []
        self.retrieval_raw_code_base = []
        self.retrieval_code_base_tokened = []
        self.test_query_tokened = []

        self.code_ids = []
        self.bm25 = evaluation.BM25(stop_words=self.vectorizer_param["stop_words"])
        if self.dataset == 'coclr':
            self._get_data_and_query_coclr()
            self.language = "python"
        elif self.dataset in ["GraphCodeBERT_python", 'GraphCodeBERT_java', 'GraphCodeBERT_php',
                              'GraphCodeBERT_javascript', 'GraphCodeBERT_go', 'GraphCodeBERT_ruby', ]:
            language = self.dataset.split("_")[-1]
            self.language = language
            self.text_precess = TextPreProcess(language)
            self._get_data_and_query_graphcodebert(language)
        elif self.dataset in ["GraphCodeBERTvalid_"+each for each in "go java javascript php ruby python".split()]:
            language = self.dataset.split("_")[-1]
            self.language = language
            self.text_precess = TextPreProcess(language)
            self._get_data_and_query_graphcodebert(language, test_name="valid.jsonl")
        elif self.dataset in ["GraphCodeBERTtrain_"+each for each in "go java javascript php ruby python".split()]:
            language = self.dataset.split("_")[-1]
            self.language = language
            self.text_precess = TextPreProcess(language)
            self._get_data_and_query_graphcodebert_train(language)
            pass
        else:
            raise Exception("Not this %s dataset implement" % self.dataset)

        self.rank = None
        self.mrr_list = None

        self.bm25_score = None
        self.bm25_time = None
        self.jaccard_score = None
        self.jaccard_time = None
        self.tfidf_cos_score = None
        self.tfidf_cos_time = None
        self.bow_cos_score = None
        self.bow_cos_time = None
        self.score = None
        if not hasattr(self, "new_query_ids"):
            self.new_query_ids = None

    def pre_process(self, code, pre_process_config=[]):

        # if not pre_process_config:
        #     pre_process_config = self.code_pre_process_config
        text_precess = self.text_precess
        # 驼峰型转蛇形
        if 'to_snake_case' in pre_process_config:
            code = text_precess.pascal_case_to_snake_case(code).replace('_', ' ')
        # 去掉注释
        if 'no_comment' in pre_process_config:
            code = text_precess.delcommonds(code)
        # 去掉符号
        if "no_punctuation" in pre_process_config:
            code = text_precess.no_punctuation(code)
        # 词性还原
        if "lemmatizer" in pre_process_config:
            code = text_precess.Lemmatizer(code)
        # 单词长度限制为 8
        if "clip_t_8" in pre_process_config:
            code = text_precess.clip_to_8(code)
        # 单词长度超过8就用 spiral 划分
        if "split_with_spiral_for_8" in pre_process_config:
            code = text_precess.split_with_spiral_for_8(code)
        # 去掉多余空格
        if "del_redundant_spaces" in pre_process_config:
            code = text_precess.del_redundant_spaces(code)

        return code

    def eval_score(self, score, only_mrr=False):
        neg_candidate = self.args.neg_candidate
        if neg_candidate == -1:
            mrr_dict = evaluation.get_mrr(score, self.code_ids, self.query_ids, self.device,
                                          new_query_ids=self.new_query_ids)
            mrr = mrr_dict['mrr']
        # first_stage.query_ids
            mrr_list = mrr_dict['mrr_list']
            print("MRR", mrr, end="")
            if only_mrr:
                return {
                "mrr_list": mrr_list, "mrr": mrr,
            }
            recall_output = evaluation.recall_eval(score, self.code_ids, self.query_ids, self.args.num_workers,
                                                   new_query_ids=self.new_query_ids)
            print("\nMRR andRecall: \n%.5f\t%s \n" %
                  (mrr, str(recall_output['text'])))
            output_dict = {
                "mrr_list": mrr_list, "mrr": mrr, "recall_tuple": recall_output['tuple'],
            }
        else:
            if self.args.window_size > 0:
                raise Exception("ToDo: Implements when window_size > 0")
            mrr_list = []
            # 选择 neg_candidate 数量的 codebase:
            if neg_candidate < 200:
                raise Exception("neg_candidate is too small")
            if neg_candidate > len(self.code_ids):
                raise Exception("neg_candidate is too big")

            if neg_candidate <= len(self.query_ids):
                # 直接选择 neg_candidate 个 query 组成 矩阵
                query_ids = self.query_ids[0:neg_candidate]
                small_score = score[0:neg_candidate, query_ids]

                code_ids = list(np.arange(0, neg_candidate))
                query_ids = list(np.arange(0, neg_candidate))

            else:
                # 需要选 neg_candidate - len(self.query_ids) 个负样本
                small_score = score[:, self.query_ids]

                other_query_ids = []

                shuffle_list = list(range(0, len(self.code_ids)))
                random.shuffle(shuffle_list)
                for each in shuffle_list:
                    if each not in self.query_ids:
                        other_query_ids.append(each)
                        if len(other_query_ids) >= neg_candidate-len(self.query_ids):
                            break

                small_score = torch.cat((small_score, score[:, other_query_ids]), dim=1)

                query_ids = list(np.arange(0, small_score.shape[0]))
                code_ids = list(np.arange(0, small_score.shape[1]))

            mrr_dict = evaluation.get_mrr(small_score, code_ids, query_ids, self.device)
            mrr = mrr_dict['mrr']
            mrr_list = mrr_dict['mrr_list']
            print("MRR", mrr, end="")
            recall_output = evaluation.recall_eval(small_score, code_ids, query_ids, self.args.num_workers)
            print("\nMRR andRecall: \n%.5f\t%s \n" %
                  (mrr, str(recall_output['text'])))
            output_dict = {
                "mrr_list": mrr_list, "mrr": mrr, "recall_tuple": recall_output['tuple'],
            }


        return output_dict

    def _get_data_and_query_coclr(self):
        coclr_path = './CoCLR_train/'
        test_500_search_path = os.path.join(coclr_path, 'data/search/cosqa-retrieval-test-500.json')
        retrieval_code_base_path = os.path.join(coclr_path, 'data/search/code_idx_map.txt')

        self.retrieval_code_base = []
        self.retrieval_raw_code_base = []
        self.test_query = []
        self.query_ids = []
        self.code_ids = []
        with open(retrieval_code_base_path, 'r') as f:
            temp = json.loads(f.read())
            for code, code_id in temp.items():
                self.retrieval_raw_code_base.append(code)
                self.retrieval_code_base.append(code)
                self.code_ids.append(code_id)

        with open(test_500_search_path, 'r') as f:
            test_500 = json.load(f)
        print(test_500[0]['doc'])
        print(test_500[0]['code'])
        print(test_500[0]['code_tokens'])

        for each in test_500:
            self.test_query.append(self.pre_process(
                each['doc'], pre_process_config=self.query_pre_process_config))
            self.query_ids.append(each["retrieval_idx"])

        for i in range(len(self.retrieval_code_base)):
            self.retrieval_code_base[i] = self.pre_process(
                self.retrieval_code_base[i], pre_process_config=self.code_pre_process_config)

    def store_raw_codeToken(self, test_name="test.jsonl"):
        self.retrieval_raw_codeToken_base = []

        g_csn_path = os.path.join(self.args.root_path, "GraphCodeBERT_dataset", self.language)
        test_search_path = os.path.join(g_csn_path, test_name)
        retrieval_code_base_path = os.path.join(g_csn_path, 'codebase.jsonl')
        # retrieval_code_base, code_ids
        logger.info("Read:" + retrieval_code_base_path)
        examples = self._get_examples(retrieval_code_base_path)
        for each in examples:
            # 这里放入 code_tokens 为 BPE 后的，eg:
            # ['<s>', 'def', 'Ġset', '_', 'cookie', 'Ġ(', 'Ġself', 'Ġ,', 'Ġname', 'Ġ,', 'Ġvalue',... ]
            self.retrieval_raw_codeToken_base.append(" ".join(each['code_tokens']))

    def store_raw_queryToken(self, test_name="test.jsonl"):
        self.retrieval_raw_queryToken_base = []

        g_csn_path = os.path.join(self.args.root_path, "GraphCodeBERT_dataset", self.language)
        test_search_path = os.path.join(g_csn_path, test_name)
        retrieval_code_base_path = os.path.join(g_csn_path, 'test.jsonl')
        # retrieval_code_base, code_ids
        logger.info("Read:" + retrieval_code_base_path)
        examples = self._get_examples(retrieval_code_base_path)
        for each in examples:
            # 这里放入 code_tokens 为 BPE 后的，eg:
            # ['<s>', 'def', 'Ġset', '_', 'cookie', 'Ġ(', 'Ġself', 'Ġ,', 'Ġname', 'Ġ,', 'Ġvalue',... ]
            self.retrieval_raw_queryToken_base.append(" ".join(each['nl_tokens']))

    def _get_data_and_query_graphcodebert(self, language, test_name="test.jsonl"):
        """
        multi-passage split.
        """
        args = self.args
        neg_candidate = self.args.neg_candidate

        g_csn_path = os.path.join(self.args.root_path, "GraphCodeBERT_dataset", language)
        test_search_path = os.path.join(g_csn_path, test_name)
        retrieval_code_base_path = os.path.join(g_csn_path, 'codebase.jsonl')

        self.retrieval_code_base = []  # 包含需要用到的 code_base
        self.retrieval_raw_code_base = []
        self.test_query = []  # 包含需要用的 test query
        self.query_ids = []  # 每个 query 对应的 code gt
        self.code_ids = []
        self.raw_test_query = []

        # retrieval_code_base, code_ids
        logger.info("Read:" + retrieval_code_base_path)
        examples = self._get_examples(retrieval_code_base_path)
        for each in examples:
            self.retrieval_code_base_tokened.append(each["code_ids"])

            self.retrieval_raw_code_base.append(each['js_original_string'])
            self.retrieval_code_base.append(self.pre_process(
                each['js_original_string'], pre_process_config=self.code_pre_process_config))

        self.code_ids = list(np.arange(0, len(self.retrieval_code_base)))

        if args.do_debug:
            for example in examples:
                self.test_query_tokened.append(example["nl_ids"])
                self.raw_test_query.append(example["nl"])
                self.test_query.append(
                    # self.pre_process(each["docstring"],
                    #                  pre_process_config=self.query_pre_process_config
                    #                  ))
                    self.pre_process(example["nl"],
                                     pre_process_config=self.query_pre_process_config
                                     ))
            self.query_ids = self.code_ids.copy()
        else:
            # query
            logger.info("Read:" + test_search_path)
            test_examples = self._get_examples(test_search_path)
            if args.nl_num != -1:
                logger.info("args.nl_num is %d" % args.nl_num)
                test_examples = test_examples[0:args.nl_num]
            for example in test_examples:
                self.test_query_tokened.append(example["nl_ids"])
                self.raw_test_query.append(example["nl"])
                self.test_query.append(
                    # self.pre_process(each["docstring"],
                    #                  pre_process_config=self.query_pre_process_config
                    #                  ))
                    self.pre_process(example["nl"],
                                     pre_process_config=self.query_pre_process_config
                                     ))

            self.query_ids = []
            codeurl_to_int = {}
            for i, each in enumerate(examples):
                if each['js_url'] in codeurl_to_int:
                    print("存在重复: ", each['url'])
                codeurl_to_int[each['js_url']] = i

            for each in test_examples:
                self.query_ids.append(codeurl_to_int[each['js_url']])

        self._code_precess(examples)

        self.retrieval_code_base_tokened = torch.tensor(self.retrieval_code_base_tokened)
        self.test_query_tokened = torch.tensor(self.test_query_tokened)

    def _get_data_and_query_graphcodebert_train(self, language):
        """
        读取 train.jsonl 文件，其中 query 和 code 一一对应
        """

        g_csn_path = os.path.join(self.args.root_path, "GraphCodeBERT_dataset", language)
        retrieval_code_base_path = os.path.join(g_csn_path, 'train.jsonl')

        self.retrieval_code_base = []
        self.retrieval_raw_code_base = []
        self.test_query = []
        self.query_ids = []
        self.code_ids = []

        # retrieval_code_base, query
        logger.info("Get retrieval_code_base")
        examples = self._get_examples(retrieval_code_base_path)

        for each in examples:
            self.retrieval_code_base_tokened.append(each["code_ids"])
            self.test_query_tokened.append(each["nl_ids"])

            self.retrieval_raw_code_base.append(each["js_original_string"])
            self.retrieval_code_base.append(self.pre_process(
                each["js_original_string"], pre_process_config=self.code_pre_process_config))
            self.test_query.append(
                self.pre_process(each["nl"],
                                 pre_process_config=self.query_pre_process_config
                                 ))

        self.query_ids = list(np.arange(0, len(self.test_query)))
        self.code_ids = list(np.arange(0, len(self.retrieval_raw_code_base)))

        self._code_precess(examples)
        self.retrieval_code_base_tokened = torch.tensor(self.retrieval_code_base_tokened)
        self.test_query_tokened =torch.tensor(self.test_query_tokened)

    def _get_examples(self, jsonl_path):
        pool = multiprocessing.Pool(self.args.num_workers)
        data = []
        if self.args.do_debug:
            # Get codebase url2codeString dict
            url2codeString = {}
            with open(jsonl_path, 'r') as f:
                for line in tqdm(f):
                    line = line.strip()
                    js = json.loads(line)
                    url2codeString[js["url"]] = js["original_string"]
            jsonl_path = jsonl_path.replace("codebase", "test")
        with open(jsonl_path, 'r') as f:
            index = 0
            for line in tqdm(f):
                line = line.strip()
                js = json.loads(line)
                if self.args.do_debug:
                    js["original_string"] = url2codeString[js["url"]]
                    index += 1
                    if index > 80:
                        break
                data.append((js["original_string"], None, js["docstring_tokens"], js["url"],
                             self.tokenizer, self.args, "all"))
        examples = pool.map(convert_examples_to_features, tqdm(data, total=len(data)))
        del data
        return examples

    def _code_precess(self, examples):
        args = self.args
        split_type = self.args.split_type

        window_setting = self.args.window_setting.split(",")
        window_size, step = int(window_setting[0].split("_")[1]), int(window_setting[1].split("_")[1]),
        if step == -1:
            step = window_size
        self.args.window_size, self.args.step = window_size, step
        if window_size <= 0 and split_type in ['token', 'word_space', 'line']:
            return

        logger.info("split the code with " + str(split_type))
        # 仅仅给要检索的 query 得 gt 划分 multi-passages
        new_query_ids = [[each] for each in self.query_ids]

        for index in tqdm(range(0, len(self.query_ids))):
            code_index = self.query_ids[index]

            # 进行划分
            self.args.remove_comments = False
            if split_type == "token":
                code_list = examples[code_index]["extracted_string_token"]
                code_subset_list = [each[1] for each in sliding_window(code_list, window_size, step)]
                if len(code_subset_list) == 1:
                    continue
                for each in code_subset_list:
                    self.retrieval_code_base_tokened.append(convert_examples_to_features(
                        (None, each, [], None, self.tokenizer, self.args, "all")
                    )["code_ids"]
                                                            )
                    self.retrieval_code_base.append(" ".join(each))
                    new_query_ids[index].append(len(self.retrieval_code_base) - 1)
            elif split_type in ["word_space", "line", "ast_subtree"]:
                code = self.retrieval_code_base[code_index]
                if args.split_type == "ast_subtree":
                    parser = parsers[args.lang]

                    code_subset_list = extract_ast_subcode(code, parser, args.lang, min_line=args.min_line,
                                                      remove_comments=True)
                    if window_size > 0:
                        code_subset_list = ["\n".join(each[1]) for each in sliding_window(code_subset_list, window_size, step)]
                else:
                    split_pattern = {
                        "word_space": ' ', "line": '\n'
                    }
                    code = self.pre_process(
                        code, pre_process_config=["del_redundant_spaces"])
                    code_list = re.split(split_pattern[split_type], code)  # support multi word split. eg: re.split(' |,|.', code)
                    code_subset_list = [" ".join(each[1]) for each in sliding_window(code_list, window_size, step)]
                if len(code_subset_list) == 1:
                    continue
                for each in code_subset_list:
                    if len(each) == 0:
                        continue
                    self.retrieval_code_base_tokened.append(convert_examples_to_features(
                        (each, None, [], None, self.tokenizer, self.args, "all")
                        # js_original_string, extracted_string_token, js_docstring_tokens, js_url, tokenizer, args, return_item
                    )["code_ids"]
                                                            )
                    self.retrieval_code_base.append(each)
                    new_query_ids[index].append(len(self.retrieval_code_base) - 1)

            self.args.remove_comments = True
        self.new_query_ids = new_query_ids
        self.code_ids = list(np.arange(0, len(self.retrieval_code_base)))


    def count_vectorizer_eval(self):
        test_500_query = self.test_query
        test_datebase = self.retrieval_code_base
        # 创建transform
        vectorizer = CountVectorizer(stop_words=stopwords.words('english'), min_df=self.vectorizer_param['min_df'])
        # vectorizer = CountVectorizer()
        # 分词并建立词汇表
        vectorizer.fit(test_500_query)
        # 结果输出
        # print(vectorizer.vocabulary_)

        # 编码 query, database vector
        query_vector = vectorizer.transform(test_500_query).toarray()
        database_vector = vectorizer.transform(test_datebase).toarray()
        print(type(database_vector), database_vector.shape)

        score_dict = {}

        self.bm25.fit(test_datebase)
        score = self.bm25.bm25_sim(
            test_500_query, test_datebase, self.device, num_workers=self.args.num_workers).cpu()
        print("\nBM25:", end="")
        self.eval_score(score)
        score_dict['BM25'] = score.cpu()

        # score = evaluation.jaccard_sim(query_vector, database_vector, self.device).cpu()
        # print("\nJaccard:", end="")
        # self.eval_score(score)
        # score_dict['Jaccard'] = score.cpu()
        #
        # score = evaluation.cosine_sim(torch.Tensor(query_vector), torch.Tensor(database_vector), self.device).cpu()
        # print("\nCos:", end="")
        # self.eval_score(score)
        # score_dict['Cos'] = score.cpu()

        return score_dict

    def tfidf_vectorizer_eval(self):
        test_500_query = self.test_query
        test_datebase = self.retrieval_code_base
        # 创建transform
        vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'), min_df=self.vectorizer_param['min_df'])
        # vectorizer = CountVectorizer()
        # 分词并建立词汇表
        vectorizer.fit(test_500_query)
        # 结果输出
        # print(vectorizer.vocabulary_)

        # 编码 query, database vector
        query_vector = vectorizer.transform(test_500_query).toarray()
        database_vector = vectorizer.transform(test_datebase).toarray()
        print(type(database_vector), database_vector.shape)

        score = evaluation.cosine_sim(torch.Tensor(query_vector), torch.Tensor(database_vector), self.device)
        print("Cos:", end="")
        self.eval_score(score)

    def print_query_and_return_list(self, score=None, rank_index=0, topK=5):
        if self.args.window_size > 0:
            raise Exception("ToDo: Implements when indow_size > 0")
        if score is not None:
            mrr_dict = evaluation.get_mrr(score, self.code_ids, self.query_ids, self.device)
            self.mrr_list = mrr_dict['mrr_list']
            self.rank = torch.argsort(-score, dim=1)
        mrr_list = self.mrr_list
        rank = self.rank.clone()

        code_ids, query_ids = self.code_ids, self.query_ids
        mrr_tensor = torch.Tensor(mrr_list)
        query_index = torch.argsort(-mrr_tensor)[rank_index]


        code_index = rank[query_index].tolist()
        gt_index = query_ids[query_index]
        gt_rank = code_index.index(int(gt_index)) + 1

        print("Query: %s. Rank: %d" % (self.test_query[query_index], gt_rank))
        print("GT Code: \n", self.retrieval_raw_code_base[int(gt_index)])
        print("Codes: ")
        for each in code_index[0:topK]:
            print(self.retrieval_raw_code_base[each])

    def compare_two_mrr(self, score1, score2, rank_index=0, topK=5,
                        name_list=['method1', 'method2'], Raw_test_query=False):
        """
        比较两个查询结果
        score: 相似度矩阵
        """
        if self.args.window_size > 0:
            raise Exception("ToDo: Implements when window_size > 0")
        if hasattr(self, "mrr_tensor1"):
            print("Have score1")
            mrr_tensor1 = self.mrr_tensor1
            mrr_tensor2 = self.mrr_tensor2
        else:
            mrr_list1 = evaluation.get_mrr(score1, self.code_ids, self.query_ids, self.device)['mrr_list']
            mrr_list2 = evaluation.get_mrr(score2, self.code_ids, self.query_ids, self.device)['mrr_list']
            mrr_tensor1 = torch.Tensor(mrr_list1)
            mrr_tensor2 = torch.Tensor(mrr_list2)
            self.mrr_tensor1, self.mrr_tensor2 = mrr_tensor1, mrr_tensor2
        # 找 1, 2 两种方法 mrr 相减
        mrr_subduction_rank = torch.argsort((mrr_tensor1 - mrr_tensor2))
        query_index = mrr_subduction_rank[rank_index]

        rank1 = torch.argsort(-score1[query_index])
        rank2 = torch.argsort(-score2[query_index])

        code_ids, query_ids = self.code_ids, self.query_ids

        gt_index = query_ids[query_index]
        code_index1 = rank1.tolist()
        code_index2 = rank2.tolist()
        gt_rank1 = code_index1.index(int(gt_index)) + 1
        gt_rank2 = code_index2.index(int(gt_index)) + 1

        print("Query: %s. \n%s: GT Rank %d. %s: GT Rank %d" %
              (self.test_query[query_index].split("\n")[0], name_list[0], gt_rank1, name_list[1], gt_rank2))
        if Raw_test_query:
            print("Raw test query: %s" % self.raw_test_query[query_index])
        print("-"*20)
        print("GT Code: \n", self.retrieval_raw_code_base[int(gt_index)])
        print("-" * 7)

        print("######### %s ##########" % name_list[0])
        print("Query: %s" % self.test_query[query_index])
        print("Codes: ")
        for each in code_index1[0:topK]:
            print("-" * 20)
            print(self.retrieval_raw_code_base[each])

        print("######### %s ##########" % name_list[1])
        print("Query: %s" % self.test_query[query_index])
        print("Codes: ")
        for each in code_index2[0:topK]:
            print("-" * 20)
            print(self.retrieval_raw_code_base[each])

    def get_bm25_time_score(self, topK, neg_candidate=-1):
        logger.info("get_bm25_time_score")
        test_500_query = self.test_query
        test_datebase = self.retrieval_code_base
        if self.bm25_score is None:
            self.bm25.fit(test_datebase)
            self.bm25_score, self.bm25_time = self.bm25.bm25_sim_with_time(test_500_query, test_datebase)

        index = torch.argsort(-self.bm25_score, -1)[:, :topK]
        topK_score = torch.zeros(index.shape)
        for i, each in enumerate(index):
            topK_score[i] = self.bm25_score[i, each]

        if neg_candidate != -1:
            return self.bm25_score, index, neg_candidate / self.bm25_score.shape[1] * self.bm25_time
        return self.bm25_score, index, self.bm25_time

    def get_jaccard_time_score(self, topK, neg_candidate=-1):
        logger.info("get_jaccard_time_score")
        if self.jaccard_score is None:
            test_500_query = self.test_query
            test_datebase = self.retrieval_code_base
            # 创建transform
            vectorizer = CountVectorizer(stop_words=self.vectorizer_param["stop_words"],
                                         min_df=self.vectorizer_param['min_df'])
            # 分词并建立词汇表
            vectorizer.fit(test_500_query)
            # 编码 query, database vector
            query_vector = vectorizer.transform(test_500_query).toarray()

            database_vector = vectorizer.transform(test_datebase).toarray()
            print(type(database_vector), database_vector.shape)
            start_time = time.time()
            self.jaccard_score = evaluation.jaccard_sim(torch.Tensor(query_vector), torch.Tensor(database_vector),
                                                         self.device).cpu()
            end_time = time.time()
            self.jaccard_time = end_time - start_time

        index = torch.argsort(-self.jaccard_score, -1)[:, :topK]
        topK_score = torch.zeros(index.shape)
        for i, each in enumerate(index):
            topK_score[i] = self.jaccard_score[i, each]

        if neg_candidate != -1:
            return self.jaccard_score, index, neg_candidate/self.jaccard_score.shape[1]*self.jaccard_time
        return self.jaccard_score, index, self.jaccard_time

    def get_tfidf_cos_time_score(self, topK, neg_candidate=-1):
        logger.info("get_tfidf_cos_time_score")
        if self.tfidf_cos_score is None:
            test_500_query = self.test_query
            test_datebase = self.retrieval_code_base
            # 创建transform
            vectorizer = TfidfVectorizer(stop_words=self.vectorizer_param["stop_words"],
                                         min_df=self.vectorizer_param['min_df'])
            # vectorizer = CountVectorizer()
            # 分词并建立词汇表
            vectorizer.fit(test_500_query)
            # 编码 query, database vector
            query_vector = vectorizer.transform(test_500_query).toarray()
            start_time = time.time()
            database_vector = vectorizer.transform(test_datebase).toarray()
            print(type(database_vector), database_vector.shape)
            self.tfidf_cos_score = evaluation.cosine_sim(
                torch.Tensor(query_vector), torch.Tensor(database_vector), self.device).cpu()
            end_time = time.time()
            self.tfidf_cos_time = end_time - start_time

        index = torch.argsort(-self.tfidf_cos_score, -1)[:, :topK]
        topK_score = torch.zeros(index.shape)
        for i, each in enumerate(index):
            topK_score[i] = self.tfidf_cos_score[i, each]

        if neg_candidate != -1:
            return self.tfidf_cos_score, index, neg_candidate/self.tfidf_cos_score.shape[1]*self.tfidf_cos_time
        return self.tfidf_cos_score, index, self.tfidf_cos_time

    def get_bow_cos_time_score(self, topK, neg_candidate=-1):
        logger.info("get_bow_cos_time_score")
        if self.bow_cos_score is None:
            test_500_query = self.test_query
            test_datebase = self.retrieval_code_base
            # 创建transform
            vectorizer = CountVectorizer(stop_words=self.vectorizer_param["stop_words"],
                                         min_df=self.vectorizer_param['min_df'])
            # vectorizer = CountVectorizer()
            # 分词并建立词汇表
            vectorizer.fit(test_500_query)
            # 编码 query, database vector
            query_vector = vectorizer.transform(test_500_query).toarray()
            start_time = time.time()
            database_vector = vectorizer.transform(test_datebase).toarray()
            print(type(database_vector), database_vector.shape)
            self.bow_cos_score = evaluation.cosine_sim(
                torch.Tensor(query_vector), torch.Tensor(database_vector), self.device).cpu()
            end_time = time.time()
            self.bow_cos_time = end_time - start_time

        index = torch.argsort(-self.bow_cos_score, -1)[:, :topK]
        topK_score = torch.zeros(index.shape)
        for i, each in enumerate(index):
            topK_score[i] = self.bow_cos_score[i, each]

        if neg_candidate != -1:
            return self.bow_cos_score, index, neg_candidate/self.bow_cos_score.shape[1]*self.bow_cos_time
        return self.bow_cos_score, index, self.bow_cos_time

    def write_twoStage_result(self, score, stage1_time, stage2_time, topK, result_dir="./result_log", neg_candidate=-1):
        # ****************************************************************
        eval_output_dict_two_stage = self.eval_score(score)

        result_log_path = os.path.join(result_dir, self.language)

        for eval_output_dict, stage_name, stage_time in zip(
                [eval_output_dict_two_stage],
            ["two_stage"],
            [stage1_time+stage2_time]
        ):
            result_log_file = os.path.join(result_log_path, stage_name)
            if not os.path.exists(os.path.abspath(result_log_file)):
                os.makedirs(os.path.abspath(result_log_file))
            (r1, r5, r10, r100, r1000, medr, mAP) = eval_output_dict["recall_tuple"]
            result_str = "Top %s\tNegC %d\t"%(str(topK), neg_candidate) + str(time.asctime(time.localtime(time.time()))) + '\t' + "%.4f\t"*9 % (
                eval_output_dict["mrr"], stage1_time, stage_time, r1, r5, r10, r100, r1000, medr
            )
            with open(os.path.join(result_log_file, self.two_stage_name+".txt"), "a") as f:
                f.write(result_str)
                f.write("\n")
        # ****************************************************************

    def get_induce(self, topK=5, type="TF-IDF", addGT=True):
        """
        index_stage1: index 相似度由大变小
        """
        if type == 'BM25':
            score_stage1, index_stage1, stage1_time = self.get_bm25_time_score(topK)  # index 相似度由大变小
        elif type == 'TF-IDF':
            score_stage1, index_stage1, stage1_time = self.get_tfidf_cos_time_score(topK)  # index 相似度由大变小

        induce = []  # 包含需要计算 score 的 index
        for index in range(0, len(index_stage1)):  # 0~-1 相似度变大
            for each in index_stage1[index][0:topK]:
                induce.append(int(index*len(self.retrieval_code_base)+each))

        # 再加上 GT
        if addGT:
            for each in range(0, len(self.query_ids)):
                induce.append(int(each*len(self.retrieval_code_base) + int(self.query_ids[each])))
            induce = list(set(induce))

        return induce

    def reset_query(self, query_index: list):
        """
        reset the query list refer to query index input。
        reset self.test_query, self.query_ids
        """
        self._check_all_setting()

        self._read_scores_botton = False
        self.test_query = list(np.array(self.all_test_query)[query_index])
        self.query_ids = list(np.array(self.all_query_ids)[query_index])
        if self.new_query_ids is not None:
            self.new_query_ids = list(np.array(self.all_new_query_ids)[query_index])

        self.rank = None
        self.mrr_list = None
        self.bm25_score = None
        self.bm25_time = None
        self.jaccard_score = None
        self.jaccard_time = None
        self.tfidf_cos_score = None
        self.tfidf_cos_time = None
        self.bow_cos_score = None
        self.bow_cos_time = None
        self.score = None
        self.time = None
        self.graphcodebert_score = None
        return

    def _check_all_setting(self):
        if not hasattr(self, "all_query_ids"):
            self.all_query_ids = self.query_ids.copy()
            self.all_test_query = self.test_query.copy()
            self._read_scores_botton = True  # whether read pre-calculate score
        if not hasattr(self, "all_test_query_tokened"):
            self.all_test_query_tokened = self.test_query_tokened.clone()
            if self.new_query_ids is None:
                self.new_query_ids = [[each] for each in self.query_ids]
            self.all_new_query_ids = self.new_query_ids.copy()


class CodeBertPStage(BaseStageCodeSearch):
    def __init__(self,
                 dataset="coclr", code_pre_process_config=[], query_pre_process_config=[],
                 args=None, model_type="classifier"):
        """
        model_type: [classifier, bi_encoder]
        """
        super().__init__(dataset,
                         code_pre_process_config=code_pre_process_config,
                         query_pre_process_config=query_pre_process_config,
                         args=args)
        # Hyper Parameter

        self.batchsize = args.eval_batch_size
        self.dual_softmax = False
        self.max_query_length = 128
        self.model_type = model_type



        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        if self.model_type == "classifier":
            tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")
            model_root = os.path.join(
                self.args.root_path, "TwoStageModels", "CodeBertP",
            )
            config = RobertaConfig.from_pretrained("microsoft/codebert-base")
            config.num_labels = 2
            model = RobertaModel.from_pretrained(
                "microsoft/codebert-base")
            self.model = ModelCodeBertPlus(model, config, tokenizer, self.args)
            self.model.load_state_dict(torch.load(
                os.path.join(model_root, "GraphCodeBERTtrain_"+self.language, "checkpoint-best-mrr", "model.best.bin"),
                map_location="cpu"
            ))
        elif self.model_type == 'bi_encoder':
            model_root = os.path.join(
                self.args.root_path, "TwoStageModels", "CodeBert_bi-encoder")
            config = RobertaConfig.from_pretrained("microsoft/codebert-base")
            config.num_labels = 2
            model_to_save_dir = os.path.join(model_root, "GraphCodeBERTtrain_" + self.language, "checkpoint-best-mrr", "model.bin")
            logger.info("load %s" % model_to_save_dir)
            model = RobertaModel.from_pretrained(
                "microsoft/codebert-base")
            model = ModelCodeBertBiEncoder(model, config, self.tokenizer, args)
            model.load_state_dict(torch.load(model_to_save_dir, map_location="cpu"))
            self.model = model.encoder
        else:
            raise Exception("Not this model_type!")

        if torch.cuda.device_count() > 1:
            self.device = torch.device("cuda")
            self.model = nn.DataParallel(self.model).to(self.device)
        elif torch.cuda.device_count() == 1:
            self.device = torch.device("cuda")
            self.model = self.model.to(self.device)
        else:
            self.device = torch.device("cpu")

        # self.text_precess = TextPreProcess()
        # self.test_query = []
        # self.query_ids = []
        # self.retrieval_code_base = []
        # self.code_ids = []
        self.score = None
        self.time = None
        self.rank = None

        self.two_stage_name = "BM25-CodeBert"

    def get_time_score(self, topK, neg_candidate=-1):
        if self.score is None:
            # ******************************************************************
            args = self.args
            model = self.model
            induce = set([])
            for index in range(0, len(self.query_ids)):
                for each in range(0, len(self.retrieval_code_base)):
                    induce.add(each + index * len(self.retrieval_code_base))
            induce = list(induce)

            # ******************************************************************
            # Stage 2
            transformers.logging.set_verbosity_error()
            distributed_model = self.model
            model = model.module if hasattr(model, 'module') else model
            input_dict = {
                "code_inputs": self.retrieval_code_base_tokened,
                "nl_inputs": self.test_query_tokened,
                "device": args.device,
                "new_query_ids": self.new_query_ids,
                "num_workers": args.num_workers,
                "batch_size": args.eval_batch_size,
                "induce": induce, "distributed_model": distributed_model, "eval_dataset_class": self,
            }
            score_stage2, time_dict = model.predict(
                input_dict
            )
            stage2_time = time_dict["score_time"] * torch.cuda.device_count()
            self.score = score_stage2
            self.time = stage2_time

        if neg_candidate != -1:
            get_time = neg_candidate / self.score.shape[1] * self.time
        index = torch.argsort(-self.score, -1)[:, :topK]
        return self.score.clone(), index, get_time

    def evaluation_two_stage_multi_stage1(self, topK, score_stage1_list, index_stage1_list,
                                          stage1_time_list, result_dir="./result_log", neg_candidate=-1,
                                          fusion_type="rerank"):
        """
        index_stage1: index 相似度由大变小

        score_stage1_list: 默认 score_stage1_list[0] 效果最好，选取其放到下一阶段。

        """
        # ******************************************************************
        args = self.args
        model = self.model
        induce = set([])
        score_stage1_norm_list = score_stage1_list
        for index_i, index_stage1 in enumerate(index_stage1_list):
            score_stage1_norm_list[index_i] = evaluation.l2norm(score_stage1_list[index_i], -1)
             # 包含需要计算 score 的 index
            for index in range(0, len(index_stage1)):  # 0~-1 相似度变大
                for each in index_stage1[index][0:topK]:
                    induce.add(each + index * len(self.retrieval_code_base))
        induce = list(induce)
        # score = torch.stack(score_stage1_list, dim=0).mean(dim=0) - 5
        logger.info("%s needs to be best.", self.two_stage_name.split("-")[0].split("_")[0])
        score_stage1 = score_stage1_norm_list[0] - 5

        stage1_time = np.sum(stage1_time_list)

        # ******************************************************************
        # Stage 2
        transformers.logging.set_verbosity_error()
        distributed_model = self.model
        model = model.module if hasattr(model, 'module') else model
        input_dict = {
            "code_inputs": self.retrieval_code_base_tokened,
            "nl_inputs": self.test_query_tokened,
            "device": args.device,
            "new_query_ids": self.new_query_ids,
            "num_workers": args.num_workers,
            "batch_size": args.eval_batch_size,
            "induce": induce, "distributed_model": distributed_model, "eval_dataset_class": self,
        }
        score_stage2, time_dict = model.predict(
            input_dict
        )
        stage2_time = time_dict["score_time"] * torch.cuda.device_count()

        # ******************************************************************
        # fusion
        score_stage2 = evaluation.l2norm(score_stage2, -1)
        combined_tensor = torch.stack(list(score_stage1_norm_list) + [score_stage2], dim=0)

        if fusion_type=="rerank":
            score = score_stage1.clone()
            for each in induce:
                code_num = len(self.code_ids)
                score[each // code_num][each % code_num] = \
                    score_stage2[each // code_num][each % code_num]

        # ******************************************************************
        score = score.float()

        score = evaluation.l2norm(score, -1)
        if self.dual_softmax:
            score = torch.softmax(score, -1)
            score = torch.softmax(score, 0)

        self.write_twoStage_result(score, stage1_time, stage2_time, topK, result_dir=result_dir, neg_candidate=neg_candidate)

        return score


    def evaluation_two_stage_data_fusion(self, topK, score_stage1_list, index_stage1_list,
                                          stage1_time_list, result_dir="./result_log", neg_candidate=-1,
                                          fusion_types=["rerank"]):
        """
        index_stage1: index 相似度由大变小

        score_stage1_list: 默认 score_stage1_list[0] 效果最好，选取其放到下一阶段。

        """
        # ******************************************************************
        args = self.args
        model = self.model
        induce = set([])
        score_stage1_norm_list = score_stage1_list
        for index_i, index_stage1 in enumerate(index_stage1_list):
            score_stage1_norm_list[index_i] = evaluation.z_norm(score_stage1_list[index_i], dim=-1)
             # 包含需要计算 score 的 index
            for index in range(0, len(index_stage1)):  # 0~-1 相似度变大
                for each in index_stage1[index][0:topK]:
                    induce.add(each + index * len(self.retrieval_code_base))
        induce = list(induce)
        # score = torch.stack(score_stage1_list, dim=0).mean(dim=0) - 5
        logger.info("%s needs to be best.", self.two_stage_name.split("-")[0].split("_")[0])
        score_stage1 = score_stage1_norm_list[0] - 5

        stage1_time = np.sum(stage1_time_list)

        # ******************************************************************
        # Stage 2
        transformers.logging.set_verbosity_error()
        distributed_model = self.model
        model = model.module if hasattr(model, 'module') else model
        input_dict = {
            "code_inputs": self.retrieval_code_base_tokened,
            "nl_inputs": self.test_query_tokened,
            "device": args.device,
            "new_query_ids": self.new_query_ids,
            "num_workers": args.num_workers,
            "batch_size": args.eval_batch_size,
            "induce": induce, "distributed_model": distributed_model, "eval_dataset_class": self,
        }
        score_stage2, time_dict = model.predict(
            input_dict
        )
        stage2_time = time_dict["score_time"] * torch.cuda.device_count()

        # ******************************************************************
        # fusion
        score_stage2 = evaluation.z_norm(score_stage2, dim=-1)
        combined_tensor = torch.stack(list(score_stage1_norm_list) + [score_stage2], dim=0)
        try:
            torch.save(combined_tensor.half(),
                       self.args.root_path +
                       "/code_search_baseline_data/GraphCodeBERT_dataset/FeatureData/python/bm25_graph_codebert_combined_tensor1.bin")
        except Exception as e:
            print(e)
        pdb.set_trace()

        logger.info("Set minimum to zero!")

        for i in range(combined_tensor.shape[0]):
            combined_tensor[i] = combined_tensor[i] - combined_tensor[i].min(dim=-1, keepdim=True).values.repeat(
                1, combined_tensor[i].shape[1])

        # self.eval_score(score)
        for fusion_type in fusion_types:
            if fusion_type=="rerank":
                score = score_stage1.clone()
                for each in induce:
                    code_num = len(self.code_ids)
                    score[each // code_num][each % code_num] = \
                        score_stage2[each // code_num][each % code_num]
            elif fusion_type=="CombANZ":
                score = torch.sum(combined_tensor, dim=0) / combined_tensor.bool().int().sum(dim=0)
            elif fusion_type=="Max":
                score = torch.max(combined_tensor, dim=0).values
            elif fusion_type=="Min":
                score = torch.min(combined_tensor, dim=0).values
            elif fusion_type=="CombMNZ":
                score = torch.sum(combined_tensor, dim=0) * combined_tensor.bool().int().sum(dim=0)
            elif fusion_type=="CombSUM":
                score = torch.sum(combined_tensor, dim=0)
            elif fusion_type=="BordaCount":
                score = (torch.argsort(torch.argsort(combined_tensor, dim=-1))).sum(0)

            # ******************************************************************
            score = score.float()
            # score = evaluation.l2norm(score, -1)
            if self.dual_softmax:
                score = torch.softmax(score, -1)
                score = torch.softmax(score, 0)
            self.write_twoStage_result(score, stage1_time, stage2_time, topK, result_dir=result_dir, neg_candidate=neg_candidate)

        return score


class CoCLRModelStage(BaseStageCodeSearch):
    def __init__(self, dataset="coclr", code_pre_process_config=[], args=None):
        super().__init__(dataset, code_pre_process_config, args=args)
        # Hyper Parameter
        if dataset == "GraphCodeBERT_python":
            self.model_dir = os.path.join(
                self.args.root_path, "TwoStageModels", "CoCLR_model", "GraphCodeBERTtrain_python",  "checkpoint-best-mrr")

        self.batchsize = 2048
        self.dual_softmax = False
        self.max_query_length = 77

        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        config = RobertaConfig.from_pretrained("microsoft/codebert-base")
        config.num_labels = 2
        model = RobertaModel(config)
        self.model = ModelBinaryCreateNeg(model, config, self.tokenizer, args)
        logger.info(
            'Reload from {} using args.checkpoint_path'.format(os.path.join(self.model_dir, 'model.bin')))
        self.model.load_state_dict(
            torch.load(
                os.path.join(self.model_dir, 'model.bin'),
                map_location=torch.device('cpu')),
            strict=False)

        # if torch.cuda.device_count() > 1:
        #     self.model = nn.DataParallel(self.model).to(self.device)
        # elif torch.cuda.device_count() == 1:
        self.model = self.model.to(self.device)  # predict 不需要并行

        self.score = None
        self.one_sim_calculate_time = None
        self.rank = None

        self.two_stage_name = "BM25-CoCLRModel"

    def _get_score_one_sim_time(self):
        if self.score is not None:
            return self.score.clone(), self.one_sim_calculate_time

        args = self.args
        model = self.model
        model = model.module if hasattr(model, 'module') else model
        model.eval()

        # 获取 Dataset, Dataloader
        logger.info("load query_token")
        query_tokens = None
        for each_query in torch_data.dataloader.DataLoader(dataset=self.test_query, batch_size=128, shuffle=False,
                                                           num_workers=args.num_workers):
            query_token = self.tokenizer(
                each_query, return_tensors='pt',
                truncation=True, padding=padding_strategy, max_length=args.nl_length)['input_ids']
            query_tokens = query_token if query_tokens is None else torch.cat((query_tokens, query_token), dim=0)

        logger.info("load code_tokens")
        code_tokens = None
        for each_query in torch_data.dataloader.DataLoader(dataset=self.retrieval_code_base, batch_size=128, shuffle=False,
                                                           num_workers=args.num_workers):
            code_token = self.tokenizer(
                each_query, return_tensors='pt',
                truncation=True, padding=padding_strategy, max_length=args.code_length)['input_ids']
            code_tokens = code_token if code_tokens is None else torch.cat((code_tokens, code_token), dim=0)

        input_dict = {
            "code_inputs": code_tokens,
            "nl_inputs": query_tokens,
            "device": args.device,
            "new_query_ids": self.new_query_ids,
            "num_workers": args.num_workers,
            "batch_size": args.eval_batch_size,
            "induce": None, "distributed_model": self.model, "eval_dataset_class": self,
        }
        # score, time_dict = model.predict_old(code_tokens, query_tokens, args.device, args.num_workers)
        score, time_dict = model.predict(input_dict)

        self.score = evaluation.z_norm(score, dim=-1).detach()
        self.one_sim_calculate_time = time_dict["nl_time"]/len(self.test_query) + \
                                      time_dict["code_time"]/len(self.retrieval_code_base) + \
            time_dict["score_time"]/len(self.test_query)/len(self.retrieval_code_base)

        self.eval_score(self.score)
        return self.score.clone(), self.one_sim_calculate_time

    def get_time_score(self, topK, neg_candidate=-1):
        score, one_sim_calculate_time = self._get_score_one_sim_time()
        self.time = score.shape[0] * score.shape[1] * one_sim_calculate_time

        if neg_candidate != -1:
            self.time = neg_candidate/score.shape[1]*self.time
        # self.eval_score(score)['mrr']

        index = torch.argsort(-self.score, -1)[:, :topK]

        return score, index, self.time

    # def evaluation(self, score_first_stage=None, topK=100, store_score=True):
    #     raise Exception("Not implement")

    def evaluation_two_stage(self, topK, score_stage1, index_stage1, stage1_time):
        """
        index_stage1: index 相似度由大变小
        """
        score_stage1 = evaluation.l2norm(score_stage1, -1)

        score = score_stage1 - 5

        # Stage 2 CoCLR
        score_stage2, one_sim_calculate_time_stage2 = self._get_score_one_sim_time()
        stage2_time = one_sim_calculate_time_stage2*(len(self.test_query)*topK)

        for index in range(0, len(index_stage1)):  # 0~-1 相似度变大
            for each in index_stage1[index][0:topK]:
                score[index, each] = score_stage2[index, each]

        score = evaluation.l2norm(score, -1)
        if self.dual_softmax:
            score = torch.softmax(score, -1)
            score = torch.softmax(score, 0)

            # score[index, inds[index][0:(code_num-topK)]] = -2 + score_first_stage[index, inds[index][0:(code_num-topK)]]

        self.write_twoStage_result(score, stage1_time, stage2_time, topK)

        return score

    def evaluation_two_stage_multi_stage1(self, topK, score_stage1_list, index_stage1_list, stage1_time_list,
                                          result_dir="./result_log", neg_candidate=-1):
        """

        index_stage1: index 相似度由大变小

        score_stage1_list: 默认 score_stage1_list[0] 效果最好，选取其放到下一阶段。

        """
        # ******************************************************************

        induce = set([])
        for index_i, index_stage1 in enumerate(index_stage1_list):
            score_stage1_list[index_i] = evaluation.l2norm(score_stage1_list[index_i], -1)
             # 包含需要计算 score 的 index
            for index in range(0, len(index_stage1)):  # 0~-1 相似度变大
                for each in index_stage1[index][0:topK]:
                    induce.add(each + index * len(self.retrieval_code_base))

        # score = torch.stack(score_stage1_list, dim=0).mean(dim=0) - 5
        logger.info("%s needs to be best.", self.two_stage_name.split("-")[0].split("_")[0])
        score = score_stage1_list[0] - 5

        stage1_time = np.sum(stage1_time_list)

        # ******************************萌萌哒************************************
        # Stage 2
        score_stage2, one_sim_calculate_time_stage2 = self._get_score_one_sim_time()
        stage2_time = one_sim_calculate_time_stage2*(len(induce))
        for each in list(induce):
            query_index = each // len(self.retrieval_code_base)
            code_index = each % len(self.retrieval_code_base)
            score[query_index, code_index] = score_stage2[query_index, code_index]


        score = evaluation.l2norm(score, -1)
        if self.dual_softmax:
            score = torch.softmax(score, -1)
            score = torch.softmax(score, 0)

            # score[index, inds[index][0:(code_num-topK)]] = -2 + score_first_stage[index, inds[index][0:(code_num-topK)]]

        self.write_twoStage_result(score, stage1_time, stage2_time, topK, result_dir=result_dir, neg_candidate=neg_candidate)

        return score


class GraphCodeBertStageOfficial(BaseStageCodeSearch):
    def __init__(self, dataset="GraphCodeBERT_python", code_pre_process_config=[],
                 query_pre_process_config=[], args=None):

        super().__init__(dataset, code_pre_process_config=code_pre_process_config,
                         query_pre_process_config=query_pre_process_config,
                         args=args)
        # Hyper Parameter
        if torch.cuda.device_count() >= 1:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.batchsize = 128
        self.dual_softmax = False

        self.model = GraphCodeBert(lang=self.language, input_args=self.args, dataset=dataset)

        self.score = None
        self.rank = None

        self.graphcodebert_score = None
        self.graphcodebert_time = None
        self.query_index = "all"

    def get_time_score(self, topK, neg_candidate=-1):
        read_scores = not self.args.online_cal

        logger.info("GraphCodeBERT get_time_score")
        if self.graphcodebert_score is None:
            self.graphcodebert_score, self.graphcodebert_time = \
                self.model.encode_text_and_code(neg_candidate=neg_candidate,
                                                read_scores=read_scores)
            # self.eval_score(scores)

        if self.query_index != "all":
            score = self.graphcodebert_score[self.query_index]
        else:
            score = self.graphcodebert_score.clone()
        index = torch.argsort(-score, -1)[:, :topK]

        return score, index, self.graphcodebert_time

    def get_induce(self, topK=5, type="", addGT=True):
        """
        index_stage1: index 相似度由大变小
        """
        score_stage1, index_stage1, stage1_time = self.get_time_score(topK)  # index 相似度由大变小

        induce_graph = []  # 包含需要计算 score 的 index
        for index in range(0, len(index_stage1)):  # 0~-1 相似度变大
            for each in index_stage1[index][0:topK]:
                induce_graph.append(int(index*len(self.retrieval_code_base)+each))

        # 再加上 GT
        if addGT:
            for each in range(0, len(self.query_ids)):
                induce_graph.append(int(each*len(self.retrieval_code_base) + int(self.query_ids[each])))
            induce_graph = list(set(induce_graph))

        return induce_graph

    def reset_query(self, query_index: list):
        """
        reset the query list refer to query index input。
        reset self.test_query, self.query_ids
        """
        self._check_all_setting()
        self._read_scores_botton = False
        self.test_query = list(np.array(self.all_test_query)[query_index])
        self.query_ids = list(np.array(self.all_query_ids)[query_index])
        if self.new_query_ids is not None:
            self.new_query_ids = list(np.array(self.all_new_query_ids)[query_index])

        self.query_index = query_index
        return


def get_args(return_parser=False):

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--result_log", default="./result_log", type=str,
                        help="the path the store metric result")

    parser.add_argument("--lang", default=None, type=str,
                        help="language.")
    parser.add_argument("--root_path", default="~/VisualSearch", type=str,
                        help="root path.")
    parser.add_argument("--num_workers", default=16, type=int,
                        help="num_workers of dataloader.")
    parser.add_argument("--neg_candidate", default=-1, type=int,
                        help="code candidate number.")

    parser.add_argument("--device", default="cuda", type=str,
                        help="cpu or cuda")
    parser.add_argument("--run_function", default="main_codebert_multi_stage1_allLang", type=str,
                        help="check the function of run.py")

    parser.add_argument("--online_cal", action='store_true',
                        help="Whether to calculate the feature online (Not read the pre-calculate file).")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--do_debug", action='store_true',
                        help="Whether to run in debug setting.")

    # ********************************************
    # TOSS
    parser.add_argument('--nl_num', type=int, default=-1,
                        help="-1 is all natural language.")

    # ********************************************
    # Code Length
    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")
    parser.add_argument("--code_length", default=200, type=int,
                        help="Optional Code input sequence length after tokenization.")
    parser.add_argument("--data_flow_length", default=16, type=int,
                        help="Optional Code idata flow length.")
    parser.add_argument("--window_setting", default="WindowSize_-1,step_-1", type=str,
                        help="Optional sliding window window_setting. WindowSize_30,step_15")
    parser.add_argument("--split_type", default="token", type=str,
                        help="split window type: ['token', 'word_space', 'line', 'ast_subtree']")
    parser.add_argument("--min_line", default=5, type=int,
                        help="Mininum code block line is 5. (if split_type == ast_subtree)")  # if split_type == "ast_subtree"

    # Training parameters
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as encoder_name_or_path")
    parser.add_argument("--encoder_name_or_path", default="microsoft/codebert-base", type=str,
                        help="Optional encoder_name_or_path")
    parser.add_argument("--warmup_steps", default=10, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--num_train_epochs", default=10, type=int,
                        help="num_train_epochs")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="num_workers of dataloader.")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="num_workers of dataloader.")

    parser.add_argument("--TrainModel", default="CoCLR_model", type=str,
                        help="The model need to be trained. ['CoCLR_model', 'CodeBert_bi-encoder']")
    parser.add_argument("--save_temp", action='store_true',
                        help="Whether to save temp model file.")
    parser.add_argument("--overwrite", action='store_true',
                        help="Whether to overwrite best model file.")
    # For GraphCodeBERTMultiCodeFusion Model
    parser.add_argument("--AttentionWithAve", action='store_true',
                        help="whether add average embedding.")
    parser.add_argument("--AttentionCommonType", default="two_layer", type=str,
                        help="two_layer, one_layer, meanpooling, maxpooling.")

    # Loss
    parser.add_argument("--margin", default=2, type=float,
                        help="margin of triplet ranking loss.")
    parser.add_argument("--nce_weight", default=1, type=float,
                        help="weight of nce loss.")
    parser.add_argument("--max_neg", action='store_true',
                        help="Whether to use hard negative miner.")
    parser.add_argument("--encoder_l2norm", action='store_true',
                        help="Whether to add l2norm after encoder embedding.")
    parser.add_argument("--ClassifyLoss", default="ce", type=str,
                        help="The Classify Loss. ['ce', 'bce']")

    # Not Using parameters
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    # ********************************************
    if return_parser:
        return parser

    # print arguments
    args = parser.parse_args()

    if '~' in args.root_path:
        args.root_path = args.root_path.replace('~', os.path.expanduser('~'))

    # set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    # set device
    device = torch.device("cuda" if (torch.cuda.is_available() and args.device == "cuda") else args.device)
    args.n_gpu = torch.cuda.device_count() if (torch.cuda.is_available() and args.device == "cuda") else 0
    args.device = device
    logger.info("device: %s, n_gpu: %s", device, args.n_gpu)
    logger.info(str(args))

    # Set seed
    set_seed(args.seed)

    # window
    window_setting = args.window_setting.split(",")
    args.window_size, args.step = int(window_setting[0].split("_")[1]), int(window_setting[1].split("_")[1]),
    if args.window_size > 0:
        if args.split_type in ['token', 'word_space']:
            args.code_length = args.window_size + 20
        if args.step <= 0:
            args.step = args.window_size

    vargs = vars(copy.copy(args))
    vargs['device'] = str(vargs['device'])
    logger.info((json.dumps(vargs, indent=2)))
    return args


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        sys.argv = "StageClass_use_ids.py --num_workers 1 " \
                   "--seed 123456 --lang python".split()
    args = get_args()

    pre_process_config = ['to_snake_case', 'no_comment']
    query_pre_process_config = ['None']
    vectorizer_param = {'min_df': 1}

    dataset = "GraphCodeBERT_" + args.lang
    a = CodeBertStage(dataset,
                             code_pre_process_config=pre_process_config,
                             query_pre_process_config=query_pre_process_config, args=args)


