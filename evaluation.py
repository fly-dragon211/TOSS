import pdb

import torch
import torch.utils.data as torch_data
import numpy as np
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import time


class BM25Dataset(torch_data.Dataset):
    def __init__(self, querys, datebase, vectorizer, avdl, b=0.75, k1=1.6):
        super().__init__()
        self.querys = querys
        self.datebase = datebase
        self.vectorizer = vectorizer

        self.b = b
        self.k1 = k1
        self.avdl = avdl
        self.X = None
        self.len_X = None

    def __len__(self):
        return self.querys.shape[0]

    def __getitem__(self, item):
        one_q_vector = self.querys[item]
        database_vectors = self.datebase
        idf = self.vectorizer._tfidf.idf_[None, one_q_vector.indices] - 1.
        b, k1, avdl = self.b, self.k1, self.avdl
        device = torch.device("cpu")

        if self.len_X is None:
            self.len_X = database_vectors.sum(1).A1
        len_X = self.len_X

        with torch.no_grad():
            database_vectors = torch.Tensor(database_vectors[:, one_q_vector.indices].toarray()).to(device)
            len_X = torch.Tensor(len_X).to(device)
            idf = torch.Tensor(idf).to(device)
            denom = database_vectors + (k1 * (1 - b + b * len_X / avdl))[:, None]
            numer = database_vectors.mul(idf.repeat(database_vectors.shape[0], 1)) * (k1 + 1)
            bm25_score = (numer / denom).sum(1)

        return bm25_score, item

class BM25(object):
    def __init__(self, b=0.75, k1=1.6, stop_words=stopwords.words('english')):
        self.vectorizer = TfidfVectorizer(norm=None, smooth_idf=False,
                                          stop_words=stop_words)
        self.b = b
        self.k1 = k1
        self.X = None
        self.len_X = None

    def fit(self, X):
        """ Fit IDF to documents X """
        self.vectorizer.fit(X)
        y = super(TfidfVectorizer, self.vectorizer).transform(X)
        self.avdl = y.sum(1).mean()

    def vector_transform(self, one_q_vector, database_vectors, idf, device=torch.device("cpu")):
        """ Calculate BM25 between query q and documents X
        store_X: 是否存储 X 计算中间变量，加速
        """

        b, k1, avdl = self.b, self.k1, self.avdl

        if self.len_X is None:
            self.len_X = database_vectors.sum(1).A1
        len_X = self.len_X

        # idf(t) = log [ n / df(t) ] + 1 in sklearn, so it need to be coneverted
        # to idf(t) = log [ n / df(t) ] with minus 1
        # idf = self.vectorizer._tfidf.idf_[None, one_q_vector.indices] - 1.

        # idf = torch.Tensor([self.vectorizer._tfidf.idf_[q_vector.indices] - 1 for q_vector in q_vectors])


        # ********************GPU begin*************************
        with torch.no_grad():
            database_vectors = torch.Tensor(database_vectors[:, one_q_vector.indices].toarray()).to(device)
            len_X = torch.Tensor(len_X).to(device)
            idf = torch.Tensor(idf).to(device)
            denom = database_vectors + (k1 * (1 - b + b * len_X / avdl))[:, None]
            numer = database_vectors.mul(idf.repeat(database_vectors.shape[0], 1)) * (k1 + 1)
            bm25_score = (numer / denom).sum(1)
        # ********************GPU end*************************

        return bm25_score

    def bm25_sim(self, query_list: list, datebase: list, device=torch.device("cpu"), num_workers=20):
        datebase = super(TfidfVectorizer, self.vectorizer).transform(datebase)
        querys = super(TfidfVectorizer, self.vectorizer).transform(query_list)
        assert sparse.isspmatrix_csr(querys[0])

        dataset = BM25Dataset(querys, datebase, self.vectorizer, self.avdl)
        dataloader = torch_data.dataloader.DataLoader(
            dataset=dataset, batch_size=1, shuffle=False, num_workers=num_workers)

        score = torch.zeros((querys.shape[0], datebase.shape[0]))

        for doc_scores, item in tqdm(dataloader):
            score[item] = doc_scores

        return score

    def bm25_sim_with_time(self, query_list: list, datebase: list):
        datebase = super(TfidfVectorizer, self.vectorizer).transform(datebase)
        querys = super(TfidfVectorizer, self.vectorizer).transform(query_list)
        assert sparse.isspmatrix_csr(querys[0])

        dataset = BM25Dataset(querys, datebase, self.vectorizer, self.avdl)
        dataloader = torch_data.dataloader.DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=16)

        score = torch.zeros((querys.shape[0], datebase.shape[0]))

        start_time = None
        for doc_scores, item in tqdm(dataloader):
            if start_time is None:
                start_time = time.time()
            score[item] = doc_scores
        end_time = time.time()

        return score, end_time-start_time

    def bm25_sim_old(self, query_list: list, datebase: list, device=torch.device("cpu")):
        score = None
        doc_scores_list = []
        datebase = super(TfidfVectorizer, self.vectorizer).transform(datebase)
        querys = super(TfidfVectorizer, self.vectorizer).transform(query_list)
        assert sparse.isspmatrix_csr(querys[0])

        with tqdm(total=querys.shape[0]) as pbar:
            for i, query in enumerate(querys):
                pbar.update(1)
                # tokenized_query = query.split(" ")
                doc_scores = self.vector_transform(query, datebase, device=device).unsqueeze(0)
                doc_scores_list.append(doc_scores)

                if (i+1) % 200 == 0:
                    temp = torch.cat(doc_scores_list, dim=0).cpu()
                    doc_scores_list = []
                    if score is None:
                        score = temp
                    else:
                        score = torch.cat((score, temp), dim=0)
            if len(doc_scores_list) > 0:
                score = torch.cat((score, torch.cat(doc_scores_list, dim=0).cpu()), dim=0)
        return score

    def vector_transform_old(self, one_q_vector, database_vectors, device=torch.device("cpu")):
        """ Calculate BM25 between query q and documents X
        store_X: 是否存储 X 计算中间变量，加速
        """

        b, k1, avdl = self.b, self.k1, self.avdl

        if self.len_X is None:
            self.len_X = database_vectors.sum(1).A1
        len_X = self.len_X

        # idf(t) = log [ n / df(t) ] + 1 in sklearn, so it need to be coneverted
        # to idf(t) = log [ n / df(t) ] with minus 1
        idf = self.vectorizer._tfidf.idf_[None, one_q_vector.indices] - 1.


        # convert to csc for better column slicing "https://zhuanlan.zhihu.com/p/36122299"
        # X = X.tocsc()[:, q.indices]
        # denom = X + (k1 * (1 - b + b * len_X / avdl))[:, None]
        # numer = X.multiply(np.broadcast_to(idf, X.shape)) * (k1 + 1)
        # bm25_score = (numer / denom).sum(1).A1


        # ********************GPU begin*************************
        with torch.no_grad():
            database_vectors = torch.Tensor(database_vectors[:, one_q_vector.indices].toarray()).to(device)
            len_X = torch.Tensor(len_X).to(device)
            idf = torch.Tensor(idf).to(device)
            denom = database_vectors + (k1 * (1 - b + b * len_X / avdl))[:, None]
            numer = database_vectors.mul(idf.repeat(database_vectors.shape[0], 1)) * (k1 + 1)
            bm25_score = (numer / denom).sum(1)
        # ********************GPU end*************************

        return bm25_score

    def transform_old(self, q, X, store_X=True, device=torch.device("cpu")):
        """ Calculate BM25 between query q and documents X
        store_X: 是否存储 X 计算中间变量，加速
        """

        b, k1, avdl = self.b, self.k1, self.avdl

        # apply CountVectorizer
        if store_X:
            if self.X is None:
                self.X = super(TfidfVectorizer, self.vectorizer).transform(X)
                X = self.X
            else:
                X = self.X
        else:
            X = super(TfidfVectorizer, self.vectorizer).transform(X)
        len_X = X.sum(1).A1
        q, = super(TfidfVectorizer, self.vectorizer).transform([q])
        assert sparse.isspmatrix_csr(q)

        # idf(t) = log [ n / df(t) ] + 1 in sklearn, so it need to be coneverted
        # to idf(t) = log [ n / df(t) ] with minus 1
        idf = self.vectorizer._tfidf.idf_[None, q.indices] - 1.


        # convert to csc for better column slicing "https://zhuanlan.zhihu.com/p/36122299"
        # X = X.tocsc()[:, q.indices]
        # denom = X + (k1 * (1 - b + b * len_X / avdl))[:, None]
        # numer = X.multiply(np.broadcast_to(idf, X.shape)) * (k1 + 1)
        # bm25_score = (numer / denom).sum(1).A1


        # ********************GPU begin*************************
        with torch.no_grad():
            X = torch.Tensor(X[:, q.indices].toarray()).to(device)
            len_X = torch.Tensor(len_X).to(device)
            idf = torch.Tensor(idf).to(device)
            denom = X + (k1 * (1 - b + b * len_X / avdl))[:, None]
            numer = X.mul(idf.repeat(X.shape[0], 1)) * (k1 + 1)
            bm25_score = (numer / denom).sum(1)
        # ********************GPU end*************************

        return bm25_score



def jaccard_sim(query, retrieval_base, device=torch.device("cpu")):
    with torch.no_grad():
        query, retrieval_base = torch.Tensor(query), torch.Tensor(retrieval_base)
        query = (query > 0).float()
        retrieval_base = (retrieval_base > 0).float().to(device)
        eps = 1e-8
        score = None
        for each_query in tqdm(torch.chunk(query, len(query)//48 + 1, dim=0)):
            # ***************GPU begin******************
            each_query = each_query.to(device)
            score_temp_list = []
            for each_retrieval_base in torch.chunk(retrieval_base, len(retrieval_base)//2048 + 1, dim=0):
                base_num = each_retrieval_base.size(0)
                each_query1 = each_query.unsqueeze(1).repeat(1, base_num, 1)
                intersection = torch.min(each_query1, each_retrieval_base).sum(-1)
                union = torch.max(each_query1, each_retrieval_base).sum(-1) + eps
                score_temp1 = (intersection / union)
                score_temp_list.append(score_temp1)
            score_temp = torch.cat(score_temp_list, dim=1)
            # ***************GPU end******************

            score_temp = score_temp.cpu()
            if score is None:
                score = score_temp
            else:
                score = torch.cat((score, score_temp), dim=0)
    return score


def jaccard_sim1(query, retrieval_base, device=torch.device("cpu")):
    query, retrieval_base = torch.Tensor(query), torch.Tensor(retrieval_base)
    query = (query > 0).float()
    retrieval_base = (retrieval_base > 0).float()
    eps = 1e-8
    score = None
    base_num = retrieval_base.size(0)
    for each in query:

        each = each.unsqueeze(0).repeat(base_num, 1)
        intersection = torch.min(each, retrieval_base).sum(-1)
        union = torch.max(each, retrieval_base).sum(-1) + eps
        score_temp = (intersection / union).unsqueeze(0)

        if score is None:
            score = score_temp
        else:
            score = torch.cat((score, score_temp), dim=0)
    return score


def l2norm(X, eps=1e-13, dim=-1):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps + 1e-14
    X = torch.div(X, norm)
    return X


def l1norm(X, eps=1e-13, dim=-1):
    """L2-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps + 1e-14
    X = torch.div(X, norm)
    return X

def z_norm(a, eps=1e-12, dim=-1, type="z"):
    # Z-score standardization
    mean_a = torch.mean(a)
    std_a = torch.std(a)
    n1 = (a - mean_a) / std_a
    # print(n1, torch.mean(n1), torch.std(n1))

    if type == 'Min-Max':
        # Min-Max scaling
        min_a = torch.min(a, dim=dim, keepdim=True).values
        max_a = torch.max(a, dim=dim, keepdim=True).values
        n = (a - min_a) / (max_a+eps)
    else:
        # Z-score standardization
        mean_a = torch.mean(a, dim=dim, keepdim=True)
        std_a = torch.std(a, dim=dim, keepdim=True)
        n = (a - mean_a) / (std_a+eps)
    return n


def cosine_sim(query, retrio, device=torch.device("cpu")):
    """Cosine similarity between all the query and retrio pairs
    """
    query, retrio = l2norm(query), l2norm(retrio)  # m*d, n*d
    score = None
    retrio = retrio.to(device)
    for each_query in tqdm(torch.chunk(query, len(query)//256 + 1, dim=0)):
        # ************GPU begin**************
        with torch.no_grad():
            each_query = each_query.to(device)  # 48*d
            score_temp = each_query.mm(retrio.t())  # 48*n
        # ************GPU end**************
        score_temp = score_temp.cpu()
        score = score_temp if score is None else torch.cat((score, score_temp), 0)

    return score


def evaluate(label_matrix):
    label_matrix = label_matrix.astype(int)
    ranks = np.zeros(label_matrix.shape[0])
    aps = np.zeros(label_matrix.shape[0])

    for index in range(len(ranks)):
        rank = np.where(label_matrix[index] == 1)[0] + 1
        ranks[index] = rank[0]

        aps[index] = np.mean([(i + 1.) / rank[i] for i in range(len(rank))])

    r1, r5, r10, r100, r1000, r5000 = [100.0 * np.mean([x <= k for x in ranks]) for k in [1, 5, 10, 100, 1000, 5000]]
    medr = np.floor(np.median(ranks))
    meanr = ranks.mean()
    mir = (1.0 / ranks).mean()
    mAP = aps.mean()

    return (r1, r5, r10, r100, r1000, r5000, medr, meanr, mir, mAP)


def recall_eval(scores, code_ids, query_ids, num_workers=16, new_query_ids=None):
    print("Get Recall")
    # scores, code_ids, query_ids, new_query_ids = graph_code_bert1.score, graph_code_bert1.code_ids, graph_code_bert1.query_ids, graph_code_bert1.new_query_ids,
    scores = np.array(scores)
    code_ids = code_ids.copy()
    query_ids = query_ids.copy()

    # 转换成字符型
    for idx in range(len(code_ids)):
        code_ids[idx] = str(code_ids[idx])
    for idx in range(len(query_ids)):
        query_ids[idx] = str(query_ids[idx])
        if new_query_ids is not None:
            for each in new_query_ids[idx][1:]:
                code_ids[each] = str(new_query_ids[idx][0])
    # 开始计算
    inds = np.argsort(scores, axis=1)
    label_matrix = np.zeros(inds.shape)
    # for index in tqdm(torch_data.dataloader.DataLoader(
    #         dataset=list(range(inds.shape[0])), batch_size=1, shuffle=False, num_workers=num_workers)):
    for index in list(range(inds.shape[0])):
        index = int(index)
        ind = inds[index][::-1]
        gt_index = np.where(np.array(code_ids)[ind] == query_ids[index].split('#')[0])[0]
        label_matrix[index][gt_index] = 1

    (r1, r5, r10, r100, r1000, r5000, medr, meanr, mir, mAP) = evaluate(label_matrix)
    output = {
        'text': "%.5f \t%.5f \t%.5f \t%.5f \t%.5f" % (r1, r5, r10, r100, r1000),
        'tuple': (r1, r5, r10, r100, r1000, medr, mAP)
    }
    return output


def get_mrr(scores, code_ids, query_ids, device=torch.device("cpu"), new_query_ids=None):
    scores = scores.to(device)
    mrr = 0
    mrr_list = []
    print("Get MRR")
    for query_idx in tqdm(range(len(scores))):
        rank = torch.argsort(-scores[query_idx]).tolist()
        if new_query_ids is not None:
            codeId2Rank = {}
            for each in range(len(rank)):
                codeId2Rank[rank[each]] = each
            try:
                items = [1 / (codeId2Rank[int(each)] + 1) for each in new_query_ids[query_idx]]
            except:
                pdb.set_trace()
            item = max(items)
        else:
            try:
                item = 1 / (rank[0:3000].index(int(query_ids[query_idx])) + 1)
            except:
                item = 0
        mrr += item
        mrr_list.append(item)
    mrr = mrr / len(scores)

    return_dict = {
        "mrr": mrr,
        "mrr_list": mrr_list

    }
    return return_dict