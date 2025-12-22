import argparse
import os.path
import datetime
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import math
import scipy.stats
import numpy as np

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
    # args.add_argument('--align_reg_list', default='[100]', type=str)
    
    # New Hyperparameters
    args.add_argument('--tau', default=0.25, type=float)
    args.add_argument('--lambda_p', default=0.1, type=float)
    args.add_argument('--lambda_o', default=0.1, type=float)
    args.add_argument('--lambda_r', default=0.1, type=float)
    args.add_argument('--alpha', default=0.0, type=float)

    # train
    args.add_argument('--device', default=0, type=int)
    args.add_argument('--EarlyStop', default=10, type=int)
    args.add_argument('--emb_size', default=64, type=int)
    args.add_argument('--num_epoch', default=1000, type=int)

    args.add_argument(
        '--topks', default='[20]', type=str)

    return args.parse_args()


class LightGCN_Encoder(torch.nn.Module):
    def __init__(self, layers, num_users, num_items, adj, device, eps):
        super(LightGCN_Encoder, self).__init__()
        self.layers = layers
        self.num_users = num_users
        self.num_items = num_items
        self.adj = adj
        self.device = device
        self.eps = eps

    def forward(self, user_emb_weight, item_emb_weight, perturbed=False):
        ego_embeddings = torch.cat([user_emb_weight, item_emb_weight], dim=0)
        all_emb = []
        for _ in range(self.layers):
            ego_embeddings = torch.sparse.mm(self.adj, ego_embeddings)
            if perturbed:
                random_noise = torch.rand_like(ego_embeddings).to(self.device)
                ego_embeddings = ego_embeddings + torch.sign(ego_embeddings) * F.normalize(random_noise, dim=1) * self.eps
            all_emb.append(ego_embeddings)
        all_emb = torch.stack(all_emb, dim=1)
        all_emb = torch.mean(all_emb, dim=1)
        user_emb, item_emb = torch.split(all_emb, [self.num_users, self.num_items])
        return user_emb, item_emb


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
        self.tau = config.tau
        
        # New Hyperparameters
        self.lambda_p = config.lambda_p
        self.lambda_o = config.lambda_o
        self.lambda_r = config.lambda_r
        self.alpha = config.alpha

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
            
        # Three LightGCN Encoders
        self.gcn_global = LightGCN_Encoder(self.layers, self.num_users, self.num_items, self.adj, self.device, self.eps)
        self.gcn_attr = LightGCN_Encoder(self.layers, self.num_users, self.num_items, self.adj, self.device, self.eps)
        self.gcn_pop = LightGCN_Encoder(self.layers, self.num_users, self.num_items, self.adj, self.device, self.eps)

        # Transformation Layers for Decoupling
        self.trans_attr = torch.nn.Sequential(
            torch.nn.Linear(self.emb_size, self.emb_size),
            torch.nn.Tanh()
        )
        self.trans_pop = torch.nn.Sequential(
            torch.nn.Linear(self.emb_size, self.emb_size),
            torch.nn.Tanh()
        )

        # Popularity Predictor
        self.pop_predictor = torch.nn.Sequential(
            torch.nn.Linear(self.emb_size, self.emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_size, 1)
        )
        
        # Popularity Labels
        self.pop_labels = self.generate_pop_labels().to(self.device)

    def generate_pop_labels(self):
        # Calculate Rank-Percentile
        pop_count = np.array(self.pop_train)
        ranks = scipy.stats.rankdata(pop_count, method='average')
        percentiles = ranks / len(pop_count)
        # Apply temperature smoothing
        labels = np.power(percentiles, 1.0 / self.tau)
        return torch.tensor(labels, dtype=torch.float32)

    def forward(self, perturbed=False):
        # Forward pass for all three branches
        # Global Branch (Original)
        u_orig, i_orig = self.gcn_global(self.user_embeddings.weight, self.item_embeddings.weight, perturbed)
        
        # Attribute Branch (Transformed)
        u_attr_in = self.trans_attr(self.user_embeddings.weight)
        i_attr_in = self.trans_attr(self.item_embeddings.weight)
        u_attr, i_attr = self.gcn_attr(u_attr_in, i_attr_in, perturbed)
        
        # Popularity Branch (Transformed)
        u_pop_in = self.trans_pop(self.user_embeddings.weight)
        i_pop_in = self.trans_pop(self.item_embeddings.weight)
        u_pop, i_pop = self.gcn_pop(u_pop_in, i_pop_in, perturbed)
        
        return u_orig, i_orig, u_attr, i_attr, u_pop, i_pop

    def bpr_loss(self, user_emb, pos_emb, neg_emb):
        pos_score = torch.mul(user_emb, pos_emb).sum(dim=1)
        neg_score = torch.mul(user_emb, neg_emb).sum(dim=1)
        bpr_loss = -torch.log(10e-8 + torch.sigmoid(pos_score - neg_score)).mean()
        
        l2_loss = torch.norm(user_emb, p=2) + torch.norm(pos_emb, p=2)
        l2_loss = self.decay * l2_loss

        return bpr_loss, l2_loss
    
    def cl_loss(self, u_idx, i_idx, j_idx, perturbed=True):
        # Using Attribute Branch for CL
        batch_users = torch.unique(torch.tensor(u_idx)).type(torch.long).to(self.device)
        bacth_pop, batch_unpop = utils.split_bacth_items(i_idx, self.pop_train)
        bacth_pop = torch.unique(torch.tensor(bacth_pop)).type(torch.long).to(self.device)
        batch_unpop = torch.unique(torch.tensor(batch_unpop)).type(torch.long).to(self.device)
        
        # Get embeddings from Attribute Branch with perturbation
        _, _, u_attr1, i_attr1, _, _ = self.forward(perturbed=True)
        _, _, u_attr2, i_attr2, _, _ = self.forward(perturbed=True)
        
        user_view_1, user_view_2 = u_attr1, u_attr2
        item_view_1, item_view_2 = i_attr1, i_attr2

        user_cl_loss = metrics.InfoNCE(
            user_view_1[batch_users], user_view_2[batch_users], self.temperature) * self.cl_rate
        item_cl_pop = self.gamma * metrics.InfoNCE_i(item_view_1[bacth_pop], item_view_2[bacth_pop], item_view_2[batch_unpop], self.temperature, self.lambda2)
        item_cl_unpop = (1 - self.gamma) * metrics.InfoNCE_i(item_view_1[batch_unpop], item_view_2[batch_unpop], item_view_2[bacth_pop], self.temperature, self.lambda2)
        item_cl_loss = (item_cl_pop + item_cl_unpop) * self.cl_rate
        cl_loss = user_cl_loss + item_cl_loss
        return cl_loss, user_cl_loss, item_cl_loss

    def batch_loss(self, u_idx, i_idx, j_idx):
        u_idx_t = torch.tensor(u_idx).long().to(self.device)
        i_idx_t = torch.tensor(i_idx).long().to(self.device)
        j_idx_t = torch.tensor(j_idx).long().to(self.device)
        
        u_orig, i_orig, u_attr, i_attr, u_pop, i_pop = self.forward(perturbed=False)
        
        # --- A. Popularity Prediction Loss ---
        # Predict popularity score from i_pop
        batch_i_pop = i_pop[i_idx_t]
        pred_pop = self.pop_predictor(batch_i_pop).squeeze()
        target_pop = self.pop_labels[i_idx_t]
        loss_pop = F.mse_loss(pred_pop, target_pop)
        
        # --- B. Orthogonal Decoupling Loss ---
        # User orthogonality
        batch_u_attr = u_attr[u_idx_t]
        batch_u_pop = u_pop[u_idx_t]
        u_attr_norm = F.normalize(batch_u_attr, dim=1, eps=1e-8)
        u_pop_norm = F.normalize(batch_u_pop, dim=1, eps=1e-8)
        loss_ortho_u = torch.abs((u_attr_norm * u_pop_norm).sum(dim=1)).mean()
        
        # Item orthogonality
        batch_i_attr = i_attr[i_idx_t]
        batch_i_pop = i_pop[i_idx_t]
        i_attr_norm = F.normalize(batch_i_attr, dim=1, eps=1e-8)
        i_pop_norm = F.normalize(batch_i_pop, dim=1, eps=1e-8)
        loss_ortho_i = torch.abs((i_attr_norm * i_pop_norm).sum(dim=1)).mean()
        
        loss_ortho = loss_ortho_u + loss_ortho_i
        
        # --- C. Reconstruction Loss ---
        # Reconstruct original embeddings from attr + pop
        # Detach original embeddings to serve as stable anchors
        u_orig_detach = u_orig.detach()
        i_orig_detach = i_orig.detach()
        
        loss_recon_u = F.mse_loss(u_attr + u_pop, u_orig_detach)
        loss_recon_i = F.mse_loss(i_attr + i_pop, i_orig_detach)
        loss_recon = loss_recon_u + loss_recon_i
        
        # --- D. Task Losses (Dual BPR) ---
        
        # 1. Global BPR Loss (Training the anchor)
        u_orig_batch = u_orig[u_idx_t]
        i_orig_pos = i_orig[i_idx_t]
        i_orig_neg = i_orig[j_idx_t]
        bpr_loss_orig, l2_loss_orig = self.bpr_loss(u_orig_batch, i_orig_pos, i_orig_neg)
        
        # 2. Attribute BPR Loss (Debiased learning)
        u_attr_batch = u_attr[u_idx_t]
        i_attr_pos = i_attr[i_idx_t]
        i_attr_neg = i_attr[j_idx_t]
        bpr_loss_attr, l2_loss_attr = self.bpr_loss(u_attr_batch, i_attr_pos, i_attr_neg)
        
        # Contrastive Loss (using Attribute Branch)
        cl_loss, user_cl_loss, item_cl_loss = self.cl_loss(u_idx, i_idx, j_idx)
        
        # Combine Task Losses
        # Note: l2_loss is calculated on embeddings, so we just take one (e.g., from orig) or sum them. 
        # Since embeddings are shared at input but transformed, l2_loss_orig penalizes the base embeddings.
        loss_task = bpr_loss_orig + bpr_loss_attr + l2_loss_orig + cl_loss
        
        # --- Total Loss ---
        loss_total = loss_task + self.lambda_p * loss_pop + self.lambda_o * loss_ortho + self.lambda_r * loss_recon
        
        return loss_total, bpr_loss_attr, l2_loss_orig, cl_loss, user_cl_loss, item_cl_loss, loss_pop, loss_ortho, loss_recon


def test(model):
    # Inference using Attribute Branch + alpha * Popularity Branch
    model.eval()
    with torch.no_grad():
        u_orig, i_orig, u_attr, i_attr, u_pop, i_pop = model.forward(perturbed=False)
        
        user_final = u_attr + model.alpha * u_pop
        item_final = i_attr + model.alpha * i_pop
        
    return user_final.detach().cpu().numpy(), item_final.detach().cpu().numpy()


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
            'pop_loss': 0.0,
            'ortho_loss': 0.0,
            'recon_loss': 0.0
        }
        # train
        with tqdm(total=math.ceil(len(data.training_data) / config.batch_size), desc=f'Epoch {epoch}',
                  unit='batch') as pbar:
            for n, batch in enumerate(dataloader.next_batch_pairwise(data, config.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                batch_loss, bpr_loss, l2_loss, cl_loss, user_cl_loss, item_cl_loss, loss_pop, loss_ortho, loss_recon = model.batch_loss(
                    user_idx, pos_idx, neg_idx)
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                
                train_res['bpr_loss'] += bpr_loss.item()
                train_res['emb_loss'] += l2_loss.item()
                train_res['batch_loss'] += batch_loss.item()
                train_res['cl_loss'] += cl_loss.item()
                train_res['pop_loss'] += loss_pop.item()
                train_res['ortho_loss'] += loss_ortho.item()
                train_res['recon_loss'] += loss_recon.item()

                pbar.set_postfix({'loss': batch_loss.item()})
                pbar.update(1)
                
        # Average losses
        num_batches = math.ceil(len(data.training_data) / config.batch_size)
        for k in train_res:
            train_res[k] /= num_batches

        # Alignment Step (Optional, kept from original if needed, usually on Attribute embeddings)
        # Using Attribute embeddings for alignment as they represent user interest
        u_orig, i_orig, u_attr, i_attr, u_pop, i_pop = model.forward(perturbed=False)
        for _ in range(train_step):
            G1, G2 = dataloader.user_items_2_group_pop(data)
            align_loss = utils.alignment_user(i_attr[G1], i_attr[G2]) * config.align_reg
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
        user_embedding, item_embedding = test(model)
        val_hr, val_recall, val_ndcg = mini_batch_test.test_acc_batch(
            data.val_U2I, data.train_U2I, user_embedding, item_embedding)
        logger.info(
            'val_hr@100:{:.6f}   val_recall@100:{:.6f}   val_ndcg@100:{:.6f}   train_time:{}s   test_tiem:{}s'.format(
                val_hr, val_recall, val_ndcg, (trin_time - start).seconds,
                (datetime.datetime.now() - trin_time).seconds))
        item_is_pop = utils.build_global_pop_mask(data.pop_train_count, 0.5)
        _, _, _, val_pop_hr, val_pop_recall, val_pop_ndcg, val_unpop_hr, val_unpop_recall, val_unpop_ndcg = mini_batch_test.test_acc_batch_pop_split(
            data.val_U2I, data.train_U2I, user_embedding, item_embedding, item_is_pop)
        logger.info('val_pop_hr@20:{:.6f}   val_pop_recall@20:{:.6f}   val_pop_ndcg@20:{:.6f}   val_unpop_hr@20:{:.6f}   val_unpop_recall@20:{:.6f}   val_unpop_ndcg@20:{:.6f}'.format(
            val_pop_hr, val_pop_recall, val_pop_ndcg, val_unpop_hr, val_unpop_recall, val_unpop_ndcg))

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
    _, _, _, test_OOD_pop_hr, test_OOD_pop_recall, test_OOD_pop_ndcg, test_OOD_unpop_hr, test_OOD_unpop_recall, test_OOD_unpop_ndcg = mini_batch_test.test_acc_batch_pop_split(
        data.test_U2I, data.train_U2I, user_embedding, item_embedding, item_is_pop)
    logger.info('=======Best   performance=====\ntest_OOD_pop_hr@20:{:.6f}   test_OOD_pop_recall@20:{:.6f}   test_OOD_pop_ndcg@20:{:.6f}   test_OOD_unpop_hr@20:{:.6f}   test_OOD_unpop_recall@20:{:.6f}   test_OOD_unpop_ndcg@20:{:.6f} '.format(
        test_OOD_pop_hr, test_OOD_pop_recall, test_OOD_pop_ndcg, test_OOD_unpop_hr, test_OOD_unpop_recall, test_OOD_unpop_ndcg))
    test_IID_hr, test_IID_recall, test_IID_ndcg = mini_batch_test.test_acc_batch(
        data.test_iid_U2I, data.train_U2I, user_embedding, item_embedding)
    logger.info(
        '=======Best   performance=====\ntest_IID_hr@20:{:.6f}   test_IID_recall@20:{:.6f}   test_IID_ndcg@20:{:.6f} '.format(
            test_IID_hr, test_IID_recall, test_IID_ndcg))
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
    f = open('/'.join((config.result_path, config.model, config.dataset_name)) + '/best_performace.txt', 'a+')
    for cl_rate in ast.literal_eval(config.cl_rate_list):
        for layers in ast.literal_eval(config.layers_list):
            for align_reg in ast.literal_eval(config.align_reg_list):
                for temperature in ast.literal_eval(config.temperature_list):
                    for lambda2 in ast.literal_eval(config.lambada_list):
                        for gamma in ast.literal_eval(config.gama_list):
                            config.temperature = temperature
                            config.cl_rate = cl_rate
                            config.layers = layers
                            config.align_reg = align_reg
                            config.lambda2 = lambda2
                            config.gamma = gamma
                            val_hr, val_recall, val_ndcg, test_OOD_hr, test_OOD_recall, test_OOD_ndcg, test_IID_hr, test_IID_recall, test_IID_ndcg, result_path = main(
                                config)
                            f.write('\n')
                            f.write(
                                '\n ====layers:{}===cl-rate:{}===align_reg:{}===gamma:{}====lambda2:{}\n  best_hr@20:{}=====best_recall@20:{}====best_ndcg@20:{}\n test_OOD_hr@20:{:.6f}   test_OOD_recall@20:{:.6f}   test_OOD_ndcg@20:{:.6f}\n test_IID_hr@20:{:.6f}   test_IID_recall@20:{:.6f}   test_IID_ndcg@20:{:.6f} \n  Resulst_path:{}\n '
                                .format(config.layers, config.cl_rate, config.align_reg, config.gamma, config.lambda2,
                                        val_hr, val_recall, val_ndcg, test_OOD_hr, test_OOD_recall, test_OOD_ndcg,
                                        test_IID_hr, test_IID_recall, test_IID_ndcg, result_path))
                            f.write('\n')
    f.close()
