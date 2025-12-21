import numpy as np
import math
import faiss
import numba as nb

@nb.njit(nopython=True)
def compute_ranking_metrics(testusers, testdata, traindata, user_rank_pred_items,topk=20):
    all_metrics = []
    for i in range(len(testusers)):
        u = testusers[i]
        one_metrics = []
        test_items = testdata[i]
        pos_length = len(test_items)
        # pred_items找出ranking结果中排除训练样本的前topk个items
        mask_items = traindata[i]
        pred_items_all = user_rank_pred_items[u]
        max_length_candicate = len(mask_items) + topk
        pred_items = [item for item in pred_items_all[:max_length_candicate]
                      if item not in mask_items][:topk]
        hit_value = 0
        dcg_value = 0
        for idx in range(topk):
            if pred_items[idx] in test_items:
                hit_value += 1
                dcg_value += math.log(2) / math.log(idx + 2)
        target_length = min(topk, pos_length)
        idcg = 0.0
        for i in range(target_length):
            idcg = idcg + math.log(2) / math.log(i + 2)
        hr_cur = hit_value / target_length
        recall_cur = hit_value / pos_length
        ndcg_cur = dcg_value / idcg
        one_metrics.append([hr_cur, recall_cur, ndcg_cur])
        all_metrics.append(one_metrics)
    return all_metrics


def num_faiss_evaluate(_test_ratings, _train_ratings,  _user_matrix, _item_matrix,Topk=20):
    '''
    Evaluation for ranking results
    Topk-largest based on faiss search
    Speeding computation based on numba
    '''

    ###  faiss search  ###
    
    query_vectors = _user_matrix
    test_users = list(_test_ratings.keys())
    dim = _user_matrix.shape[-1]
    index = faiss.IndexFlatIP(dim)
    index.add(_item_matrix)
    max_mask_items_length = max(
        len(_train_ratings[user]) for user in _train_ratings.keys())
    sim, _user_rank_pred_items = index.search(
        query_vectors, Topk+max_mask_items_length)

    testdata = [_test_ratings[user] for user in _test_ratings.keys()]
    traindata = [_train_ratings[user] for user in _test_ratings.keys()]
    all_metrics = compute_ranking_metrics(nb.typed.List(test_users), nb.typed.List(testdata),
                                          nb.typed.List(traindata), nb.typed.List(_user_rank_pred_items),topk=Topk)

    all_metrics=np.array(all_metrics).T
    hr_out=np.sum(all_metrics[0])
    recall_out=np.sum(all_metrics[1])
    ndcg_out=np.sum(all_metrics[2])


    return hr_out,recall_out,ndcg_out


def test_acc_batch(_test_U2I, _train_U2I,  _user_matrix, _item_matrix,topk=20):
    test_users = list(_test_U2I.keys())
    batch_id = 0
    batch_size=35000
    data_size = len(test_users)
    hr_all,recall_all,ndcg_all=0.0,0.0,0.0
    while batch_id < data_size:
        if batch_id + batch_size <= data_size:
            batch_users=[test_users[idx] for idx in range(batch_id, batch_size + batch_id)]
            batch_id += batch_size
        else:
            batch_users=[test_users[idx] for idx in range(batch_id, data_size)]
            batch_id = data_size

        hr_out,recall_out,ndcg_out=num_faiss_evaluate({key: _test_U2I[key] for key in batch_users},{key: _train_U2I[key] for key in batch_users},_user_matrix,_item_matrix,Topk=topk)
        hr_all=hr_all+hr_out
        recall_all=recall_all+recall_out
        ndcg_all=ndcg_all+ndcg_out
    hr_all=hr_all/data_size
    recall_all=recall_all/data_size
    ndcg_all=ndcg_all/data_size
    return hr_all,recall_all,ndcg_all

def compute_ranking_metrics_split(testusers, testdata, traindata, user_rank_pred_items, item_is_pop, topk=20):
    pop_hr_sum = 0.0
    pop_recall_sum = 0.0
    pop_ndcg_sum = 0.0
    pop_cnt = 0
    unpop_hr_sum = 0.0
    unpop_recall_sum = 0.0
    unpop_ndcg_sum = 0.0
    unpop_cnt = 0
    for i in range(len(testusers)):
        u = testusers[i]
        test_items = testdata[i]
        mask_items = traindata[i]
        pred_items_all = user_rank_pred_items[i]
        max_length_candicate = len(mask_items) + topk
        pred_items = [item for item in pred_items_all[:max_length_candicate] if item not in mask_items][:topk]
        pos_length_pop = 0
        pos_length_unpop = 0
        for it in test_items:
            if item_is_pop[it] == 1:
                pos_length_pop += 1
            else:
                pos_length_unpop += 1
        pop_hit = 0
        pop_dcg = 0.0
        unpop_hit = 0
        unpop_dcg = 0.0
        for idx in range(topk):
            it = pred_items[idx]
            if it in test_items:
                if item_is_pop[it] == 1:
                    pop_hit += 1
                    pop_dcg += math.log(2) / math.log(idx + 2)
                else:
                    unpop_hit += 1
                    unpop_dcg += math.log(2) / math.log(idx + 2)
        target_pop = min(topk, pos_length_pop)
        target_unpop = min(topk, pos_length_unpop)
        idcg_pop = 0.0
        for j in range(target_pop):
            idcg_pop += math.log(2) / math.log(j + 2)
        idcg_unpop = 0.0
        for j in range(target_unpop):
            idcg_unpop += math.log(2) / math.log(j + 2)
        if target_pop > 0 and pos_length_pop > 0 and idcg_pop > 0:
            pop_hr_sum += pop_hit / target_pop
            pop_recall_sum += pop_hit / pos_length_pop
            pop_ndcg_sum += pop_dcg / idcg_pop
            pop_cnt += 1
        if target_unpop > 0 and pos_length_unpop > 0 and idcg_unpop > 0:
            unpop_hr_sum += unpop_hit / target_unpop
            unpop_recall_sum += unpop_hit / pos_length_unpop
            unpop_ndcg_sum += unpop_dcg / idcg_unpop
            unpop_cnt += 1
    return pop_hr_sum, pop_recall_sum, pop_ndcg_sum, pop_cnt, unpop_hr_sum, unpop_recall_sum, unpop_ndcg_sum, unpop_cnt

def test_acc_batch_pop_split(_test_U2I, _train_U2I,  _user_matrix, _item_matrix, item_is_pop, topk=20):
    test_users = list(_test_U2I.keys())
    batch_id = 0
    batch_size=35000
    data_size = len(test_users)
    hr_all,recall_all,ndcg_all=0.0,0.0,0.0
    pop_hr_sum_total = 0.0
    pop_recall_sum_total = 0.0
    pop_ndcg_sum_total = 0.0
    pop_cnt_total = 0
    unpop_hr_sum_total = 0.0
    unpop_recall_sum_total = 0.0
    unpop_ndcg_sum_total = 0.0
    unpop_cnt_total = 0
    while batch_id < data_size:
        if batch_id + batch_size <= data_size:
            batch_users=[test_users[idx] for idx in range(batch_id, batch_size + batch_id)]
            batch_id += batch_size
        else:
            batch_users=[test_users[idx] for idx in range(batch_id, data_size)]
            batch_id = data_size
        hr_out,recall_out,ndcg_out=num_faiss_evaluate({key: _test_U2I[key] for key in batch_users},{key: _train_U2I[key] for key in batch_users},_user_matrix,_item_matrix,Topk=topk)
        hr_all=hr_all+hr_out
        recall_all=recall_all+recall_out
        ndcg_all=ndcg_all+ndcg_out
        query_vectors = _user_matrix[batch_users]
        dim = _user_matrix.shape[-1]
        index = faiss.IndexFlatIP(dim)
        index.add(_item_matrix)
        max_mask_items_length = max(len(_train_U2I[user]) for user in batch_users)
        sim, _user_rank_pred_items = index.search(query_vectors, topk+max_mask_items_length)
        testdata = [_test_U2I[user] for user in batch_users]
        traindata = [_train_U2I[user] for user in batch_users]
        pop_hr_sum, pop_recall_sum, pop_ndcg_sum, pop_cnt, unpop_hr_sum, unpop_recall_sum, unpop_ndcg_sum, unpop_cnt = compute_ranking_metrics_split(batch_users, testdata, traindata, nb.typed.List(_user_rank_pred_items), item_is_pop, topk=topk)
        pop_hr_sum_total += pop_hr_sum
        pop_recall_sum_total += pop_recall_sum
        pop_ndcg_sum_total += pop_ndcg_sum
        pop_cnt_total += pop_cnt
        unpop_hr_sum_total += unpop_hr_sum
        unpop_recall_sum_total += unpop_recall_sum
        unpop_ndcg_sum_total += unpop_ndcg_sum
        unpop_cnt_total += unpop_cnt
    hr_all=hr_all/data_size
    recall_all=recall_all/data_size
    ndcg_all=ndcg_all/data_size
    pop_hr = pop_hr_sum_total / pop_cnt_total if pop_cnt_total>0 else 0.0
    pop_recall = pop_recall_sum_total / pop_cnt_total if pop_cnt_total>0 else 0.0
    pop_ndcg = pop_ndcg_sum_total / pop_cnt_total if pop_cnt_total>0 else 0.0
    unpop_hr = unpop_hr_sum_total / unpop_cnt_total if unpop_cnt_total>0 else 0.0
    unpop_recall = unpop_recall_sum_total / unpop_cnt_total if unpop_cnt_total>0 else 0.0
    unpop_ndcg = unpop_ndcg_sum_total / unpop_cnt_total if unpop_cnt_total>0 else 0.0
    return hr_all,recall_all,ndcg_all,pop_hr,pop_recall,pop_ndcg,unpop_hr,unpop_recall,unpop_ndcg
