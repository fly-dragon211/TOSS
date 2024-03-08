import os
import pdb
import sys

import logging
import pickle
import time

import json
import evaluation
import numpy as np

from StageClass_use_ids import BaseStageCodeSearch, \
    GraphCodeBertStageOfficial, CodeBertPStage
from GraphCodeBERT import InputFeatures
from nltk.corpus import stopwords

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s",
                    datefmt = '%m/%d/%Y %H:%M:%S',)
logger = logging.getLogger(__name__)


def main_codebert_multi_stage1_allLang(args):
    lang = args.lang
    dataset = "GraphCodeBERT_" + lang
    name_stage1_array = np.array([
        'GraphCodeBert', 'BM25', 'Jaccard', 'TfidfCos', 'Bow'])

    multi_stage1 = [[0], [1], [0,1]]

    vectorizer_param = {'min_df': 5}

    pre_process_configs = np.array(['to_snake_case', 'no_stop_words',
                                    'split_with_spiral_for_8', 'lemmatizer', 'no_punctuation', 'no_comment'])

    pre_process_config = list(pre_process_configs[[0,1,2,3,4,5]])
    query_pre_process_config = list(set(pre_process_config) - set(['no_comment']))
    a = BaseStageCodeSearch(
        dataset,
        code_pre_process_config=pre_process_config,
        query_pre_process_config=query_pre_process_config,
        vectorizer_param=vectorizer_param, args=args)

    pre_process_config = [
        'no_comment',
        # 'None',
    ]
    b = CodeBertPStage(
        dataset,
        code_pre_process_config=pre_process_config,
        args=args)

    graph_code_bert = GraphCodeBertStageOfficial(
        dataset,
        code_pre_process_config=pre_process_config,
        args=args)


    # ************************萌萌哒*************************
    # 得到 one stage的结果
    for index, each in enumerate([
        graph_code_bert.get_time_score(1),
        a.get_bm25_time_score(1), a.get_jaccard_time_score(1),
        a.get_tfidf_cos_time_score(1), a.get_bow_cos_time_score(1)
    ]):
        score = each[0]
        b.two_stage_name = name_stage1_array[index]
        if not args.do_debug:
            b.write_twoStage_result(score, 1, 1, 1)

    # for topK in [1500]:
    for topK in [5, 10, 100]:
        print("Topk: ", topK, end="\n")
        score_stage1_list = []
        index_stage1_list = []
        stage1_time_list = []

        time_score_list = [
            graph_code_bert.get_time_score(topK),
            a.get_bm25_time_score(topK), a.get_jaccard_time_score(topK),
            a.get_tfidf_cos_time_score(topK), a.get_bow_cos_time_score(1)
        ]
        # time_score_list = [a.get_tfidf_cos_time_score(topK), a.get_jaccard_time_score(topK),
        #                    ]
        for score_stage1, index_stage1, stage1_time in time_score_list:
            score_stage1_list.append(score_stage1)
            index_stage1_list.append(index_stage1)
            stage1_time_list.append(stage1_time)

        # ************************萌萌哒*************************
        for each in multi_stage1:
            b.two_stage_name = "%s-CodeBert" % ("_".join(name_stage1_array[each]))
            logger.info(b.two_stage_name)
            b.evaluation_two_stage_multi_stage1(
                topK, np.array(score_stage1_list)[each],
                np.array(index_stage1_list)[each], np.array(stage1_time_list)[each])
        # ************************萌萌哒*************************



if __name__ == "__main__":
    import sys
    from StageClass_use_ids import get_args

    if len(sys.argv) == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = "1"
        sys.argv = "run.py --num_workers 1 --root_path ~/VisualSearch " \
                   "--device cuda --lang python --code_length 256 --eval_batch_size 32 " \
                   "--seed 123456".split()
        "--online_cal --do_debug --window_setting WindowSize_256,step_200 --run_function get_time_two_stage --nl_num 100"

    args = get_args()
    main_codebert_multi_stage1_allLang(args)
