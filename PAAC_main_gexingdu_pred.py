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
    args = argparse.ArgumentParser(description="PAAC")

    # dataset
    args.add_argument('--dataset_name', default='gowalla', type=str)

    ##OOD
    args.add_argument('--dataset_path', default='OOD_Data', type=str)
    args.add_argument('--result_path', default='OOD_result', type=str)

    args.add_argument('--bpr_num_neg', default=1, type=int)

    # LightGCN model
    args.add_argument('--model', default='PAAC', type=str)
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
    # args.add_argument('--align_reg_list', default='[100]', type=str)

    # Inference Mode
    # 'simple': Dot product only (Disentangled/Debiased Inference)
    # 'complex': Interest * (1 + w * m_pop) (Consistent Inference)
    args.add_argument('--inference_mode', default='simple', type=str, choices=['simple', 'complex'])

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

        # model
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
        self.inference_mode = config.inference_mode

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

        # New parameters for Personalized Popularity Preference
        self.W_pop = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(self.emb_size, self.emb_size)))
        self.b_pop = torch.nn.Parameter(torch.zeros(1))
        self.w1 = torch.nn.Parameter(torch.ones(1))
        self.tau = config.tau

        # Calculate global max pop for normalization
        # Register as buffer to ensure it moves to device with model
        pop_values = torch.tensor(self.pop_train, dtype=torch.float)
        self.register_buffer('max_pop', torch.max(pop_values) + 1e-8)
        self.register_buffer('pop_count_tensor', pop_values)
        
        # Cache for efficient evaluation
        self.user_emb_cache = None
        self.item_emb_cache = None

    #
    def forward(self, perturbed=False):
        ego_embeddings = torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], dim=0)

        # all_emb = [ego_embeddings]

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
        # 1. Interest Score (LightGCN)
        pos_int_dot = torch.mul(user_emb, pos_emb).sum(dim=1)
        neg_int_dot = torch.mul(user_emb, neg_emb).sum(dim=1)
        pos_interest = torch.where(pos_int_dot < 0, torch.exp(pos_int_dot), pos_int_dot + 1.0)
        neg_interest = torch.where(neg_int_dot < 0, torch.exp(neg_int_dot), neg_int_dot + 1.0)

        # 2. Popularity Normalization + Smoothing (p_i = (f_i/f_max)^gamma)
        # Use registered buffer for efficiency and device safety
        pos_f = self.pop_count_tensor[pos_idx]
        neg_f = self.pop_count_tensor[neg_idx]
        
        pos_p = torch.pow(pos_f / self.max_pop, self.pop_gamma) 
        neg_p = torch.pow(neg_f / self.max_pop, self.pop_gamma)

        # 3. Bilinear Sensitivity beta
        u_w = torch.matmul(user_emb, self.W_pop)
        pos_beta = torch.sigmoid(torch.mul(u_w, pos_emb).sum(dim=1) + self.b_pop)
        neg_beta = torch.sigmoid(torch.mul(u_w, neg_emb).sum(dim=1) + self.b_pop)

        # 4. Gaussian Kernel Adaptation M_pop
        pos_m_pop = torch.exp(-torch.pow(pos_p - pos_beta, 2) / self.tau)
        neg_m_pop = torch.exp(-torch.pow(neg_p - neg_beta, 2) / self.tau)

        # 5. Final Score
        # Residual Connection Version as requested
        pos_score = pos_interest * (1.0 + self.w1 * pos_m_pop)
        neg_score = neg_interest * (1.0 + self.w1 * neg_m_pop)

        bpr_loss = -torch.log(10e-8 + torch.sigmoid(pos_score - neg_score)).mean()
        l2_loss = self.decay * (user_emb.norm(2) + pos_emb.norm(2) + neg_emb.norm(2) + self.W_pop.norm(2))
        return bpr_loss, l2_loss

    def cl_loss(self, u_idx, i_idx, j_idx):
        # batch里采样
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

    def batch_loss(self, u_idx, i_idx, j_idx):

        user_embedding, item_embedding = self.forward(perturbed=False)
        user_emb = user_embedding[u_idx]
        pos_emb = item_embedding[i_idx]
        neg_emb = item_embedding[j_idx]
        bpr_loss, l2_loss = self.bpr_loss(user_emb, pos_emb, neg_emb, i_idx, j_idx)
        cl_loss, user_cl_loss, item_cl_loss = self.cl_loss(u_idx, i_idx, j_idx)
        batch_loss = bpr_loss + l2_loss + cl_loss
        return batch_loss, bpr_loss, l2_loss, cl_loss, user_cl_loss, item_cl_loss

    def predict(self, user_idx, item_idx):
        user_embedding, item_embedding = self.forward(perturbed=False)
        u_emb = user_embedding[user_idx]
        i_emb = item_embedding[item_idx]
        
        dot = torch.mul(u_emb, i_emb).sum(dim=-1)
        
        if self.inference_mode == 'simple':
            # Disentangled Inference: Use only pure interest (dot product)
            return dot
            
        # Complex Inference (Consistent with Training)
        interest = torch.where(dot < 0, torch.exp(dot), dot + 1.0)
        
        # Popularity processing
        # item_idx can be iterable
        f_i = self.pop_count_tensor[item_idx]
        p_i = torch.pow(f_i / self.max_pop, self.pop_gamma)
        
        u_w = torch.matmul(u_emb, self.W_pop)
        beta = torch.sigmoid(torch.mul(u_w, i_emb).sum(dim=-1) + self.b_pop)
        m_pop = torch.exp(-torch.pow(p_i - beta, 2) / self.tau)
        
        return interest * (1.0 + self.w1 * m_pop)

    def full_predict(self):
        """
        Calculates scores for all user-item pairs efficiently using matrix operations.
        Returns: user_emb, item_emb, scores_matrix
        """
        user_emb, item_emb = self.forward(perturbed=False)
        
        # 1. Interest Matrix
        # [num_users, num_items]
        interest_dot = torch.matmul(user_emb, item_emb.t())
        
        if self.inference_mode == 'simple':
            # Disentangled Inference
            return user_emb, item_emb, interest_dot

        interest = torch.where(interest_dot < 0, torch.exp(interest_dot), interest_dot + 1.0)
        
        # 2. Popularity & Beta
        # [num_items]
        p_i = torch.pow(self.pop_count_tensor / self.max_pop, self.pop_gamma)
        
        # [num_users, emb_size]
        u_w = torch.matmul(user_emb, self.W_pop)
        
        # Beta: [num_users, num_items]
        # u_w (U, D) @ item_emb.T (D, I) -> (U, I) + scalar
        beta = torch.sigmoid(torch.matmul(u_w, item_emb.t()) + self.b_pop)
        
        # 3. Adaptation Matrix M_pop
        # p_i broadcast to (U, I)
        m_pop = torch.exp(-torch.pow(p_i.unsqueeze(0) - beta, 2) / self.tau)
        
        # 4. Final Scores
        scores = interest * (1.0 + self.w1 * m_pop)
        
        return user_emb, item_emb, scores
    
    def precompute_embeddings(self):
        """
        Precompute and cache embeddings for efficient evaluation.
        Should be called before validation/testing loops.
        """
        with torch.no_grad():
            self.user_emb_cache, self.item_emb_cache = self.forward(perturbed=False)

    def clear_cache(self):
        """
        Clear cached embeddings to free memory or before next training step.
        """
        self.user_emb_cache = None
        self.item_emb_cache = None

    def batch_predict(self, user_indices):
        """
        Predict scores for a batch of users against ALL items.
        Args:
            user_indices: list or tensor of user indices [batch_size]
        Returns:
            scores: [batch_size, num_items]
        """
        # Use cached embeddings if available (Efficient Evaluation)
        if self.user_emb_cache is not None and self.item_emb_cache is not None:
            user_emb = self.user_emb_cache
            item_emb = self.item_emb_cache
        else:
            # Fallback to re-computing (Slow)
            user_emb, item_emb = self.forward(perturbed=False)
        
        u_emb = user_emb[user_indices] # [B, D]
        # item_emb is [N, D]
        
        # 1. Interest (Simple Dot Product)
        # [B, N]
        interest_dot = torch.matmul(u_emb, item_emb.t())
        
        # As requested: For Valid/Test, we strictly use the dot product (Simple Inference).
        # This aligns with the "Disentangled Learning" strategy where we train with bias modeling
        # but predict with pure interest.
        
        return interest_dot

    def get_mpop_for_analysis(self, user_indices, item_indices):
        """
        Helper function to extract M_pop values for specific user-item pairs.
        Used for correlation analysis.
        """
        user_emb, item_emb = self.forward(perturbed=False)
        u_emb = user_emb[user_indices]
        i_emb = item_emb[item_indices]
        
        # Popularity
        f_i = self.pop_count_tensor[item_indices]
        p_i = torch.pow(f_i / self.max_pop, self.pop_gamma)
        
        # Beta
        u_w = torch.matmul(u_emb, self.W_pop)
        beta = torch.sigmoid(torch.mul(u_w, i_emb).sum(dim=-1) + self.b_pop)
        
        # M_pop
        tau = self.tau if self.tau > 1e-6 else 0.1
        m_pop = torch.exp(-torch.pow(p_i - beta, 2) / tau)
        
        return m_pop, f_i


def analyze_mpop_correlation(model, data, logger):
    """
    Analyzes the correlation between M_pop values and Item Popularity.
    This provides empirical evidence that M_pop captures popularity information.
    """
    logger.info("====== Starting M_pop Correlation Analysis ======")
    model.eval()
    
    # Sample a subset of test data for analysis (to save time)
    # We'll use the test_U2I dictionary
    test_users = list(data.test_U2I.keys())
    # Sample 1000 users or all if less than 1000
    import random
    sampled_users = random.sample(test_users, min(1000, len(test_users)))
    
    all_mpops = []
    all_pops = []
    
    with torch.no_grad():
        for u in sampled_users:
            items = list(data.test_U2I[u])
            if not items:
                continue
                
            u_tensor = torch.tensor([u] * len(items)).to(model.device)
            i_tensor = torch.tensor(items).to(model.device)
            
            mpops, pops = model.get_mpop_for_analysis(u_tensor, i_tensor)
            
            all_mpops.extend(mpops.cpu().numpy().tolist())
            all_pops.extend(pops.cpu().numpy().tolist())
    
    if not all_mpops:
        logger.info("No data for correlation analysis.")
        return

    # Calculate Pearson Correlation
    import numpy as np
    mpop_arr = np.array(all_mpops)
    pop_arr = np.array(all_pops)
    
    # Avoid division by zero in correlation calculation
    if np.std(mpop_arr) == 0 or np.std(pop_arr) == 0:
        logger.info("Standard deviation is zero, cannot calculate correlation.")
        return

    correlation = np.corrcoef(mpop_arr, pop_arr)[0, 1]
    
    logger.info(f"Analyzed {len(all_mpops)} interactions.")
    logger.info(f"Pearson Correlation between M_pop and Item Popularity (Count): {correlation:.4f}")
    
    # Average M_pop for different popularity groups
    # Group by log(pop) to handle long-tail distribution
    # Simple split: High Pop (Top 20%) vs Low Pop
    threshold = np.percentile(pop_arr, 80)
    high_pop_mask = pop_arr >= threshold
    low_pop_mask = pop_arr < threshold
    
    avg_mpop_high = np.mean(mpop_arr[high_pop_mask]) if np.any(high_pop_mask) else 0.0
    avg_mpop_low = np.mean(mpop_arr[low_pop_mask]) if np.any(low_pop_mask) else 0.0
    
    # Calculate Pearson Correlation for High Pop and Low Pop separately
    corr_high = 0.0
    if np.sum(high_pop_mask) > 1 and np.std(mpop_arr[high_pop_mask]) > 0 and np.std(pop_arr[high_pop_mask]) > 0:
        corr_high = np.corrcoef(mpop_arr[high_pop_mask], pop_arr[high_pop_mask])[0, 1]
        
    corr_low = 0.0
    if np.sum(low_pop_mask) > 1 and np.std(mpop_arr[low_pop_mask]) > 0 and np.std(pop_arr[low_pop_mask]) > 0:
        corr_low = np.corrcoef(mpop_arr[low_pop_mask], pop_arr[low_pop_mask])[0, 1]
    
    logger.info(f"Popularity Threshold (Top 20%): {threshold}")
    logger.info(f"Avg M_pop [High Pop]: {avg_mpop_high:.4f} | Pearson Corr [High Pop]: {corr_high:.4f}")
    logger.info(f"Avg M_pop [Low Pop]: {avg_mpop_low:.4f}  | Pearson Corr [Low Pop]: {corr_low:.4f}")
    
    if avg_mpop_high > avg_mpop_low:
        logger.info("Observation: M_pop is higher for popular items, indicating it captures popularity signal.")
    else:
        logger.info("Observation: M_pop distribution is complex or inversely related.")
        
    logger.info("====== End Analysis ======")


def test(model):
    # Returns the model itself to be used by custom test functions
    return model


def train(config, data, model, optimizer, early_stopping, logger, train_step=1):
    model.train()
    for epoch in range(config.num_epoch):
        start = datetime.datetime.now()
        train_res = {
            'bpr_loss': 0.0,
            'emb_loss': 0.0,
            'cl_loss': 0.0,
            'batch_loss': 0.0,
            'align_loss': 0.0,
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
        train_res['bpr_loss'] = train_res['bpr_loss'] / math.ceil(len(data.training_data) / config.batch_size)
        train_res['emb_loss'] = train_res['emb_loss'] / math.ceil(len(data.training_data) / config.batch_size)
        train_res['batch_loss'] = train_res['batch_loss'] / math.ceil(len(data.training_data) / config.batch_size)
        train_res['cl_loss'] = train_res['cl_loss'] / math.ceil(len(data.training_data) / config.batch_size)

        user_emb, item_emb = model.forward()
        for _ in range(train_step):
            G1, G2 = dataloader.user_items_2_group_pop(data)
            align_loss = utils.alignment_user(item_emb[G1], item_emb[G2]) * config.align_reg
            optimizer.zero_grad()
            align_loss.backward()
            optimizer.step()
            train_res['align_loss'] += align_loss.item()

        train_res['align_loss'] = train_res['align_loss'] / train_step

        training_logs = 'epoch: %d, ' % epoch
        for name, value in train_res.items():
            training_logs += name + ':' + '%.6f' % value + ' '
        logger.info(training_logs)
        trin_time = datetime.datetime.now()

        # val
        model.eval()
        
        # Precompute embeddings for efficient evaluation
        model.precompute_embeddings()
        
        # Use custom test function with model
        val_hr, val_recall, val_ndcg = mini_batch_test.test_acc_batch_custom(
            data.val_U2I, data.train_U2I, model)
        
        # Pop/Unpop Split Evaluation (Validation)
        item_is_pop = utils.build_global_pop_mask(data.pop_train_count, 0.5)
        _, _, _, val_pop_hr, val_pop_recall, val_pop_ndcg, val_unpop_hr, val_unpop_recall, val_unpop_ndcg = mini_batch_test.test_acc_batch_pop_split_custom(
            data.val_U2I, data.train_U2I, model, item_is_pop)

        # Analyze M_pop Correlation during validation
        if epoch % 1 == 0: # Analyze every epoch
             analyze_mpop_correlation(model, data, logger)
             
        # Clear cache after evaluation to free memory
        model.clear_cache()

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
    # user_embedding, item_embedding = test(model) # Not needed anymore
    
    # Precompute embeddings for final testing
    model.eval()
    model.precompute_embeddings()

    val_hr, val_recall, val_ndcg = mini_batch_test.test_acc_batch_custom(
        data.val_U2I, data.train_U2I, model)
    logger.info('=======Best   performance=====\nval_hr@20:{:.6f}   val_recall@20:{:.6f}   val_ndcg@20:{:.6f} '.format(
        val_hr, val_recall, val_ndcg))
    
    # Pop/Unpop Split Evaluation (Validation)
    item_is_pop = utils.build_global_pop_mask(data.pop_train_count, 0.5)
    _, _, _, val_pop_hr, val_pop_recall, val_pop_ndcg, val_unpop_hr, val_unpop_recall, val_unpop_ndcg = mini_batch_test.test_acc_batch_pop_split_custom(
        data.val_U2I, data.train_U2I, model, item_is_pop)
    logger.info('=======Best   performance=====\nval_pop_hr@20:{:.6f}   val_pop_recall@20:{:.6f}   val_pop_ndcg@20:{:.6f}   val_unpop_hr@20:{:.6f}   val_unpop_recall@20:{:.6f}   val_unpop_ndcg@20:{:.6f} '.format(
        val_pop_hr, val_pop_recall, val_pop_ndcg, val_unpop_hr, val_unpop_recall, val_unpop_ndcg))

    test_OOD_hr, test_OOD_recall, test_OOD_ndcg = mini_batch_test.test_acc_batch_custom(
        data.test_U2I, data.train_U2I, model)
    logger.info(
        '=======Best   performance=====\ntest_OOD_hr@20:{:.6f}   test_OOD_recall@20:{:.6f}   test_OOD_ndcg@20:{:.6f} '.format(
            test_OOD_hr, test_OOD_recall, test_OOD_ndcg))
            
    # Pop/Unpop Split Evaluation (Test OOD)
    _, _, _, test_OOD_pop_hr, test_OOD_pop_recall, test_OOD_pop_ndcg, test_OOD_unpop_hr, test_OOD_unpop_recall, test_OOD_unpop_ndcg = mini_batch_test.test_acc_batch_pop_split_custom(
        data.test_U2I, data.train_U2I, model, item_is_pop)
    logger.info('=======Best   performance=====\ntest_OOD_pop_hr@20:{:.6f}   test_OOD_pop_recall@20:{:.6f}   test_OOD_pop_ndcg@20:{:.6f}   test_OOD_unpop_hr@20:{:.6f}   test_OOD_unpop_recall@20:{:.6f}   test_OOD_unpop_ndcg@20:{:.6f} '.format(
        test_OOD_pop_hr, test_OOD_pop_recall, test_OOD_pop_ndcg, test_OOD_unpop_hr, test_OOD_unpop_recall, test_OOD_unpop_ndcg))

    test_IID_hr, test_IID_recall, test_IID_ndcg = mini_batch_test.test_acc_batch_custom(
        data.test_iid_U2I, data.train_U2I, model)
    logger.info(
        '=======Best   performance=====\ntest_IID_hr@20:{:.6f}   test_IID_recall@20:{:.6f}   test_IID_ndcg@20:{:.6f} '.format(
            test_IID_hr, test_IID_recall, test_IID_ndcg))
            
    # Pop/Unpop Split Evaluation (Test IID)
    _, _, _, test_IID_pop_hr, test_IID_pop_recall, test_IID_pop_ndcg, test_IID_unpop_hr, test_IID_unpop_recall, test_IID_unpop_ndcg = mini_batch_test.test_acc_batch_pop_split_custom(
        data.test_iid_U2I, data.train_U2I, model, item_is_pop)
    logger.info('=======Best   performance=====\ntest_IID_pop_hr@20:{:.6f}   test_IID_pop_recall@20:{:.6f}   test_IID_pop_ndcg@20:{:.6f}   test_IID_unpop_hr@20:{:.6f}   test_IID_unpop_recall@20:{:.6f}   test_IID_unpop_ndcg@20:{:.6f} '.format(
        test_IID_pop_hr, test_IID_pop_recall, test_IID_pop_ndcg, test_IID_unpop_hr, test_IID_unpop_recall, test_IID_unpop_ndcg))
        
    # Analyze M_pop Correlation
    analyze_mpop_correlation(model, data, logger)
    
    model.clear_cache()
    
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
                                    f.write("PAAC_main_gexingdu_pred")
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
                                    f.write(
                                        '\n ====layers:{}===cl-rate:{}===align_reg:{}===gamma:{}====lambda2:{}====tau:{}====pop_gamma:{}====\n  best_hr@20:{}=====best_recall@20:{}====best_ndcg@20:{}\n test_OOD_hr@20:{:.6f}   test_OOD_recall@20:{:.6f}   test_OOD_ndcg@20:{:.6f}\n test_IID_hr@20:{:.6f}   test_IID_recall@20:{:.6f}   test_IID_ndcg@20:{:.6f} \n  Resulst_path:{}\n '
                                        .format(config.layers, config.cl_rate, config.align_reg, config.gamma, config.lambda2, config.tau, config.pop_gamma,
                                                val_hr, val_recall, val_ndcg, test_OOD_hr, test_OOD_recall, test_OOD_ndcg,
                                                test_IID_hr, test_IID_recall, test_IID_ndcg, result_path))
                                    f.write('\n')
                                    f.close()
    # f.close()
