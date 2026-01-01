import argparse
import os.path
import datetime
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import math
import scipy as scipy

import os
import metrics, dataloader, utils, mini_batch_test
import ast


def main_args():
    # model description
    args = argparse.ArgumentParser(description="PAAC_Final_Align")

    # dataset
    args.add_argument('--dataset_name', default='gowalla', type=str)

    ##OOD
    args.add_argument('--dataset_path', default='OOD_Data', type=str)
    args.add_argument('--result_path', default='OOD_result', type=str)

    args.add_argument('--bpr_num_neg', default=1, type=int)

    # LightGCN model
    args.add_argument('--model', default='PAAC_Final', type=str)
    args.add_argument('--decay', default=0.0001, type=float)
    args.add_argument('--lr', default=0.001, type=float)
    args.add_argument('--batch_size', default=2048, type=int)
    args.add_argument('--layers_list', default='[3]', type=str)
    args.add_argument('--eps', default=0.2, type=float)
    args.add_argument('--cl_rate_list', default='[0.2]', type=str)
    args.add_argument('--temperature_list', default='[0.2]', type=str)
    args.add_argument('--seed', default=12345, type=int)
    args.add_argument('--align_reg_list', default='[1]', type=str)
    args.add_argument('--lambada_list', default='[0.2]', type=str)
    args.add_argument('--gama_list', default='[0.2]', type=str)
    args.add_argument('--tau_list', default='[0.1]', type=str)
    args.add_argument('--pop_gamma_list', default='[0.8]', type=str)
    
    # 新增: User 对齐 Loss 的权重
    # 建议设小一点，比如 0.01 到 0.1，作为正则项
    args.add_argument('--user_align_weight', default=0.1, type=float)

    # train
    args.add_argument('--device', default=0, type=int)
    args.add_argument('--EarlyStop', default=10, type=int)
    args.add_argument('--emb_size', default=64, type=int)
    args.add_argument('--num_epoch', default=1, type=int)

    args.add_argument(
        '--topks', default='[20]', type=str)

    return args.parse_args()


class PAAC(torch.nn.Module):
    def __init__(self, config, data):
        super(PAAC, self).__init__()

        # model parameters
        self.emb_size = config.emb_size
        self.decay = config.decay
        self.layers = config.layers
        self.device = torch.device(config.device)
        self.eps = config.eps
        self.cl_rate = config.cl_rate
        self.temperature = config.temperature
        self.pop_train = data.pop_train_count
        self.lambda2 = config.lambda2
        self.gamma = config.gamma
        self.pop_gamma = config.pop_gamma
        self.user_align_weight = config.user_align_weight

        # data
        self.num_users = data.num_users
        self.num_items = data.num_items
        self.adj = data.norm_adj.to(self.device)

        # embedding
        user_emb_weight = torch.nn.init.normal_(torch.empty(
            self.num_users, self.emb_size), mean=0, std=0.1)
        item_emb_weight = torch.nn.init.normal_(torch.empty(
            self.num_items, self.emb_size), mean=0, std=0.1)
        self.user_embeddings = torch.nn.Embedding(
            self.num_users, self.emb_size, _weight=user_emb_weight)
        self.item_embeddings = torch.nn.Embedding(
            self.num_items, self.emb_size, _weight=item_emb_weight)

        # === 核心模块: 偏好匹配 (Gexingdu) ===
        # 保留显式建模用户对流行度的偏好
        self.W_pop = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(self.emb_size, self.emb_size)))
        self.b_pop = torch.nn.Parameter(torch.zeros(1))
        self.w1 = torch.nn.Parameter(torch.ones(1))
        self.tau = config.tau

        # Buffer for popularity
        pop_values = torch.tensor(self.pop_train, dtype=torch.float)
        self.register_buffer('max_pop', torch.max(pop_values) + 1e-8)
        self.register_buffer('pop_count_tensor', pop_values)

    # 标准 LightGCN Forward
    def forward(self, perturbed=False):
        ego_embeddings = torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], dim=0)
        all_emb = []
        for _ in range(self.layers):
            ego_embeddings = torch.sparse.mm(self.adj, ego_embeddings)
            if perturbed:
                random_noise = torch.rand_like(ego_embeddings).to(self.device)
                ego_embeddings = ego_embeddings + torch.sign(ego_embeddings) * F.normalize(random_noise,
                                                                                           dim=1) * self.eps
            all_emb = all_emb + [ego_embeddings]
        all_emb = torch.stack(all_emb, dim=1)
        all_emb = torch.mean(all_emb, dim=1)
        user_emb, item_emb = torch.split(
            all_emb, [self.num_users, self.num_items])
        return user_emb, item_emb

    def bpr_loss(self, user_emb, pos_emb, neg_emb, pos_idx, neg_idx):
        # === 核心逻辑：Gexingdu 匹配 ===
        # 1. Interest Score
        pos_int_dot = torch.mul(user_emb, pos_emb).sum(dim=1)
        neg_int_dot = torch.mul(user_emb, neg_emb).sum(dim=1)
        pos_interest = torch.where(pos_int_dot < 0, torch.exp(pos_int_dot), pos_int_dot + 1.0)
        neg_interest = torch.where(neg_int_dot < 0, torch.exp(neg_int_dot), neg_int_dot + 1.0)

        # 2. Popularity Normalization
        pos_f = self.pop_count_tensor[pos_idx]
        neg_f = self.pop_count_tensor[neg_idx]
        pos_p = torch.pow(pos_f / self.max_pop, self.pop_gamma) 
        neg_p = torch.pow(neg_f / self.max_pop, self.pop_gamma)

        # 3. Bilinear Sensitivity beta
        u_w = torch.matmul(user_emb, self.W_pop)
        pos_beta = torch.sigmoid(torch.mul(u_w, pos_emb).sum(dim=1) + self.b_pop)
        neg_beta = torch.sigmoid(torch.mul(u_w, neg_emb).sum(dim=1) + self.b_pop)

        # 4. Gaussian Kernel
        pos_m_pop = torch.exp(-torch.pow(pos_p - pos_beta, 2) / self.tau)
        neg_m_pop = torch.exp(-torch.pow(neg_p - neg_beta, 2) / self.tau)

        # 5. Final Score
        pos_score = pos_interest * (self.w1 * pos_m_pop)
        neg_score = neg_interest * (self.w1 * neg_m_pop)

        bpr_loss = -torch.log(10e-8 + torch.sigmoid(pos_score - neg_score)).mean()
        
        l2_loss = self.decay * (user_emb.norm(2) + pos_emb.norm(2) + neg_emb.norm(2) + self.W_pop.norm(2))
        return bpr_loss, l2_loss

    def cl_loss(self, u_idx, i_idx, j_idx):
        u_idx = torch.tensor(u_idx)
        bacth_pop, batch_unpop = utils.split_bacth_items(i_idx, self.pop_train)
        batch_users = torch.unique(u_idx).type(torch.long).to(self.device)
        bacth_pop = torch.tensor(bacth_pop)
        bacth_pop = torch.unique(bacth_pop).type(torch.long).to(self.device)
        batch_unpop = torch.tensor(batch_unpop)
        batch_unpop = torch.unique(batch_unpop).type(torch.long).to(self.device)
        
        user_view_1, item_view_1 = self.forward(perturbed=True)
        user_view_2, item_view_2 = self.forward(perturbed=True)
        
        user_cl_loss = metrics.InfoNCE(
            user_view_1[batch_users], user_view_2[batch_users], self.temperature) * self.cl_rate
        item_cl_pop = self.gamma * metrics.InfoNCE_i(item_view_1[bacth_pop], item_view_2[bacth_pop],
                                                     item_view_2[batch_unpop], self.temperature, self.lambda2)
        item_cl_unpop = (1 - self.gamma) * metrics.InfoNCE_i(item_view_1[batch_unpop], item_view_2[batch_unpop],
                                                             item_view_2[bacth_pop], self.temperature, self.lambda2)
        item_cl_loss = (item_cl_pop + item_cl_unpop) * self.cl_rate
        cl_loss = user_cl_loss + item_cl_loss
        return cl_loss, user_cl_loss, item_cl_loss

    # === 新增: 偏好流形对齐 Loss (User Alignment) ===
# === 改进版：基于高斯核的软对齐 (Soft Alignment) ===
    def get_user_align_loss(self, user_emb):
        # 1. 计算偏好分数 beta [Batch, 1]
        u_w = torch.matmul(user_emb, self.W_pop)
        beta_score = u_w.sum(dim=1, keepdim=True) 
        
        # 2. 计算 Batch 内两两之间的偏好距离矩阵
        #    dist_beta[i, j] = (beta_i - beta_j)^2
        #    利用广播机制: [B, 1] - [1, B] -> [B, B]
        beta_diff = beta_score - beta_score.t()
        dist_beta = beta_diff.pow(2)
        
        # 3. 生成软对齐权重 (Soft Weights) - 使用高斯核
        #    sigma 是控制“软度”的超参，越小越关注极度相似的，越大越宽容
        #    这里复用 self.tau 或者给一个单独的 sigma，比如 0.5
        sigma = 0.5
        #    weights[i, j] 接近 1 表示两人偏好极度相似，接近 0 表示不相似
        alignment_weights = torch.exp(-dist_beta / sigma)
        
        #    把对角线（自己对自己）的权重设为 0，避免 Loss 被自己主导
        mask = torch.eye(user_emb.size(0), device=self.device).bool()
        alignment_weights = alignment_weights.masked_fill(mask, 0.0)

        # 4. 计算 Embedding 的相似度矩阵
        #    先归一化，再算 Cosine Similarity
        user_emb_norm = F.normalize(user_emb, dim=1)
        #    sim_emb[i, j] 是用户 i 和 j 的 Embedding 相似度
        sim_emb = torch.mm(user_emb_norm, user_emb_norm.t())
        
        # 5. 计算加权对齐 Loss
        #    逻辑：如果 weights[i,j] 很大（偏好相似），我们希望 sim_emb[i,j] 也很大（接近1）
        #    Loss = sum(Weight * (1 - Sim)) / sum(Weight)
        #    这就只惩罚 "偏好相似但 Embedding 不相似" 的情况
        
        #    (1 - sim_emb) 是距离，范围 [0, 2]
        weighted_dist = alignment_weights * (1.0 - sim_emb)
        
        #    归一化 Loss，除以权重的和，避免 Batch Size 影响
        align_loss = weighted_dist.sum() / (alignment_weights.sum() + 1e-8)
        
        return align_loss
    def batch_loss(self, u_idx, i_idx, j_idx):
        user_embedding, item_embedding = self.forward(perturbed=False)
        user_emb = user_embedding[u_idx]
        pos_emb = item_embedding[i_idx]
        neg_emb = item_embedding[j_idx]
        
        # 1. Main Task: BPR Loss
        bpr_loss, l2_loss = self.bpr_loss(user_emb, pos_emb, neg_emb, i_idx, j_idx)
        
        # 2. CL Loss
        cl_loss, user_cl_loss, item_cl_loss = self.cl_loss(u_idx, i_idx, j_idx)
        
        # 3. New Task: User Alignment Loss
        #    利用偏好排序，拉近偏好相似用户的距离
        align_loss = self.get_user_align_loss(user_emb)
        
        # Total Loss
        batch_loss = bpr_loss + l2_loss + cl_loss + self.user_align_weight * align_loss
        
        return batch_loss, bpr_loss, l2_loss, cl_loss, user_cl_loss, item_cl_loss

    def predict(self, user_idx, item_idx):
        user_embedding, item_embedding = self.forward(perturbed=False)
        u_emb = user_embedding[user_idx]
        i_emb = item_embedding[item_idx]
        
        dot = torch.mul(u_emb, i_emb).sum(dim=-1)
        interest = torch.where(dot < 0, torch.exp(dot), dot + 1.0)
        
        f_i = self.pop_count_tensor[item_idx]
        p_i = torch.pow(f_i / self.max_pop, self.pop_gamma)
        
        u_w = torch.matmul(u_emb, self.W_pop)
        beta = torch.sigmoid(torch.mul(u_w, i_emb).sum(dim=-1) + self.b_pop)
        m_pop = torch.exp(-torch.pow(p_i - beta, 2) / self.tau)
        
        return interest * (self.w1 * m_pop)

    def full_predict(self):
        user_emb, item_emb = self.forward(perturbed=False)
        interest_dot = torch.matmul(user_emb, item_emb.t())
        interest = torch.where(interest_dot < 0, torch.exp(interest_dot), interest_dot + 1.0)
        
        p_i = torch.pow(self.pop_count_tensor / self.max_pop, self.pop_gamma)
        u_w = torch.matmul(user_emb, self.W_pop)
        beta = torch.sigmoid(torch.matmul(u_w, item_emb.t()) + self.b_pop)
        m_pop = torch.exp(-torch.pow(p_i.unsqueeze(0) - beta, 2) / self.tau)
        
        scores = interest * (self.w1 * m_pop)
        return user_emb, item_emb, scores


def test(model):
    user_embedding, item_embedding = model.forward()
    return user_embedding.detach().cpu().numpy(), item_embedding.detach().cpu().numpy()


def train(config, data, model, optimizer, early_stopping, logger, train_step=1):
    model.train()
    for epoch in range(config.num_epoch):
        start = datetime.datetime.now()
        train_res = {
            'bpr_loss': 0.0,
            'emb_loss': 0.0,
            'cl_loss': 0.0,
            'batch_loss': 0.0,
            'group_align_loss': 0.0, 
        }
        # train
        with tqdm(total=math.ceil(len(data.training_data) / config.batch_size), desc=f'Epoch {epoch}',
                  unit='batch') as pbar:
            for n, batch in enumerate(dataloader.next_batch_pairwise(data, config.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                batch_loss, bpr_loss, l2_loss, cl_loss, user_cl_loss, item_cl_loss = model.batch_loss(
                    user_idx, pos_idx, neg_idx)
                
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                
                train_res['bpr_loss'] += bpr_loss.item()
                train_res['emb_loss'] += l2_loss.item()
                train_res['batch_loss'] += batch_loss.item()
                train_res['cl_loss'] += cl_loss.item()

                pbar.set_postfix({'loss (batch)': batch_loss.item()})
                pbar.update(1)
        
        # Log 归一化
        num_batches = math.ceil(len(data.training_data) / config.batch_size)
        for k in train_res:
            if k != 'group_align_loss':
                train_res[k] /= num_batches

        user_emb, item_emb = model.forward()
        for _ in range(train_step):
            G1, G2 = dataloader.user_items_2_group_pop(data)
            align_loss = utils.alignment_user(item_emb[G1], item_emb[G2]) * config.align_reg
            optimizer.zero_grad()
            align_loss.backward()
            optimizer.step()
            train_res['group_align_loss'] += align_loss.item()

        train_res['group_align_loss'] = train_res['group_align_loss'] / train_step

        training_logs = 'epoch: %d, ' % epoch
        for name, value in train_res.items():
            training_logs += name + ':' + '%.6f' % value + ' '
        logger.info(training_logs)
        trin_time = datetime.datetime.now()

        # val
        model.eval()
        user_embedding, item_embedding = test(model)
        val_hr, val_recall, val_ndcg = mini_batch_test.test_acc_batch(
            data.val_U2I, data.train_U2I, user_embedding, item_embedding)
        
        # Pop/Unpop Split Evaluation (Validation)
        item_is_pop = utils.build_global_pop_mask(data.pop_train_count, 0.5)
        _, _, _, val_pop_hr, val_pop_recall, val_pop_ndcg, val_unpop_hr, val_unpop_recall, val_unpop_ndcg = mini_batch_test.test_acc_batch_pop_split(
            data.val_U2I, data.train_U2I, user_embedding, item_embedding, item_is_pop)

        logger.info(
            'val_hr@100:{:.6f}   val_recall@100:{:.6f}   val_ndcg@100:{:.6f}   val_pop_hr:{:.6f}   val_pop_recall:{:.6f}   val_pop_ndcg:{:.6f}   val_unpop_hr:{:.6f}   val_unpop_recall:{:.6f}   val_unpop_ndcg:{:.6f}   train_time:{}s   test_tiem:{}s'.format(
                val_hr, val_recall, val_ndcg, val_pop_hr, val_pop_recall, val_pop_ndcg, val_unpop_hr, val_unpop_recall, val_unpop_ndcg, (trin_time - start).seconds,
                (datetime.datetime.now() - trin_time).seconds))

        # early_stopping
        early_stopping(val_hr, model, epoch)
        if early_stopping.early_stop:
            logger.info("Early stopping")
            break


def main(config):
    ISOTIMEFORMAT = '%m%d-%H%M%S'
    timestamp = str(datetime.datetime.now().strftime(ISOTIMEFORMAT))
    file_name = '_'.join((str(config.layers), str(config.cl_rate), str(config.align_reg), str(config.gamma),
                          str(config.lambda2), timestamp))

    result_path = '/'.join((config.result_path,
                            config.model, config.dataset_name, file_name))
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    logger_file_name = os.path.join(result_path, 'train_logger')
    logger = utils.get_logger(logger_file_name)
    for name, value in vars(config).items():
        logger.info('%20s =======> %-20s' % (name, value))

    # Seed
    if config.seed:
        utils.setup_seed(config.seed)

    # Load Data
    logger.info('------Load Data-----')
    data = dataloader.Data(config, logger)
    data.norm_adj = dataloader.LaplaceGraph(
        data.num_users, data.num_items, data.train_U2I).generate()

    # Load Model
    logger.info('------Load Model-----')
    model = PAAC(config, data)
    model.to(model.device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # EarlyStopping
    early_stopping = utils.EarlyStopping(logger,
                                         config.EarlyStop, verbose=True, path=result_path)

    # train model
    train(config, data, model, optimizer, early_stopping, logger)

    # #test model

    model_dict = model.load_state_dict(torch.load(result_path + "/best_val_epoch.pt"))
    user_embedding, item_embedding = test(model)

    val_hr, val_recall, val_ndcg = mini_batch_test.test_acc_batch(
        data.val_U2I, data.train_U2I, user_embedding, item_embedding)
    logger.info('=======Best   performance=====\nval_hr@20:{:.6f}   val_recall@20:{:.6f}   val_ndcg@20:{:.6f} '.format(
        val_hr, val_recall, val_ndcg))
    
    # Pop/Unpop Split Evaluation (Validation)
    item_is_pop = utils.build_global_pop_mask(data.pop_train_count, 0.5)
    _, _, _, val_pop_hr, val_pop_recall, val_pop_ndcg, val_unpop_hr, val_unpop_recall, val_unpop_ndcg = mini_batch_test.test_acc_batch_pop_split(
        data.val_U2I, data.train_U2I, user_embedding, item_embedding, item_is_pop)
    logger.info('=======Best   performance=====\nval_pop_hr@20:{:.6f}   val_pop_recall@20:{:.6f}   val_pop_ndcg@20:{:.6f}   val_unpop_hr@20:{:.6f}   val_unpop_recall@20:{:.6f}   val_unpop_ndcg@20:{:.6f} '.format(
        val_pop_hr, val_pop_recall, val_pop_ndcg, val_unpop_hr, val_unpop_recall, val_unpop_ndcg))

    test_OOD_hr, test_OOD_recall, test_OOD_ndcg = mini_batch_test.test_acc_batch(
        data.test_U2I, data.train_U2I, user_embedding, item_embedding)
    logger.info(
        '=======Best   performance=====\ntest_OOD_hr@20:{:.6f}   test_OOD_recall@20:{:.6f}   test_OOD_ndcg@20:{:.6f} '.format(
            test_OOD_hr, test_OOD_recall, test_OOD_ndcg))
            
    # Pop/Unpop Split Evaluation (Test OOD)
    _, _, _, test_OOD_pop_hr, test_OOD_pop_recall, test_OOD_pop_ndcg, test_OOD_unpop_hr, test_OOD_unpop_recall, test_OOD_unpop_ndcg = mini_batch_test.test_acc_batch_pop_split(
        data.test_U2I, data.train_U2I, user_embedding, item_embedding, item_is_pop)
    logger.info('=======Best   performance=====\ntest_OOD_pop_hr@20:{:.6f}   test_OOD_pop_recall@20:{:.6f}   test_OOD_pop_ndcg@20:{:.6f}   test_OOD_unpop_hr@20:{:.6f}   test_OOD_unpop_recall@20:{:.6f}   test_OOD_unpop_ndcg@20:{:.6f} '.format(
        test_OOD_pop_hr, test_OOD_pop_recall, test_OOD_pop_ndcg, test_OOD_unpop_hr, test_OOD_unpop_recall, test_OOD_unpop_ndcg))

    test_IID_hr, test_IID_recall, test_IID_ndcg = mini_batch_test.test_acc_batch(
        data.test_iid_U2I, data.train_U2I, user_embedding, item_embedding)
    logger.info(
        '=======Best   performance=====\ntest_IID_hr@20:{:.6f}   test_IID_recall@20:{:.6f}   test_IID_ndcg@20:{:.6f} '.format(
            test_IID_hr, test_IID_recall, test_IID_ndcg))
            
    # Pop/Unpop Split Evaluation (Test IID)
    _, _, _, test_IID_pop_hr, test_IID_pop_recall, test_IID_pop_ndcg, test_IID_unpop_hr, test_IID_unpop_recall, test_IID_unpop_ndcg = mini_batch_test.test_acc_batch_pop_split(
        data.test_iid_U2I, data.train_U2I, user_embedding, item_embedding, item_is_pop)
    logger.info('=======Best   performance=====\ntest_IID_pop_hr@20:{:.6f}   test_IID_pop_recall@20:{:.6f}   test_IID_pop_ndcg@20:{:.6f}   test_IID_unpop_hr@20:{:.6f}   test_IID_unpop_recall@20:{:.6f}   test_IID_unpop_ndcg@20:{:.6f} '.format(
        test_IID_pop_hr, test_IID_pop_recall, test_IID_pop_ndcg, test_IID_unpop_hr, test_IID_unpop_recall, test_IID_unpop_ndcg))
    return val_hr, val_recall, val_ndcg, test_OOD_hr, test_OOD_recall, test_OOD_ndcg, test_IID_hr, test_IID_recall, test_IID_ndcg, result_path


if __name__ == '__main__':
    config = main_args()
    result_path = '/'.join((config.result_path,
                            config.model, config.dataset_name))
    if not os.path.exists(result_path):
        os.makedirs(result_path)
   
    for cl_rate in ast.literal_eval(config.cl_rate_list):
        for layers in ast.literal_eval(config.layers_list):
            for align_reg in ast.literal_eval(config.align_reg_list):
                for temperature in ast.literal_eval(config.temperature_list):
                    for lambda2 in ast.literal_eval(config.lambada_list):
                        for gamma in ast.literal_eval(config.gama_list):
                            for tau in ast.literal_eval(config.tau_list):
                                for pop_gamma in ast.literal_eval(config.pop_gamma_list):
                                    f = open('/'.join((config.result_path, config.model, config.dataset_name)) + '/best_performace.txt', 'a+')
                                    config.temperature = temperature
                                    config.cl_rate = cl_rate
                                    config.layers = layers
                                    config.align_reg = align_reg
                                    config.lambda2 = lambda2
                                    config.gamma = gamma
                                    config.pop_gamma = pop_gamma
                                    config.tau=tau
                                    val_hr, val_recall, val_ndcg, test_OOD_hr, test_OOD_recall, test_OOD_ndcg, test_IID_hr, test_IID_recall, test_IID_ndcg, result_path = main(
                                        config)
                                    f.write('\n')
                                    f.write("PAAE")
                                    f.write(
                                        '\n ====layers:{}===cl-rate:{}===align_reg:{}===gamma:{}====lambda2:{}====tau:{}====pop_gamma:{}====\n  best_hr@20:{}=====best_recall@20:{}====best_ndcg@20:{}\n test_OOD_hr@20:{:.6f}   test_OOD_recall@20:{:.6f}   test_OOD_ndcg@20:{:.6f}\n test_IID_hr@20:{:.6f}   test_IID_recall@20:{:.6f}   test_IID_ndcg@20:{:.6f} \n  Resulst_path:{}\n '
                                        .format(config.layers, config.cl_rate, config.align_reg, config.gamma, config.lambda2, config.tau, config.pop_gamma,
                                                val_hr, val_recall, val_ndcg, test_OOD_hr, test_OOD_recall, test_OOD_ndcg,
                                                test_IID_hr, test_IID_recall, test_IID_ndcg, result_path))
                                    f.write('\n')
                                    f.close()