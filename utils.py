import logging
import numpy as np
import random
import os
import torch
import torch.nn.functional as F


def get_logger(filename, verbosity=1, name=None):
    filename = filename + '.txt'
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s]%(message)s"
    )
    logger = logging.getLogger(name)
    logger.handlers=[]
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "a")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

    
def setup_seed(seed):

    os.environ['PYTHONHASHSEED']=str(seed)


    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self,logger, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.logger=logger

    def __call__(self, val_loss, model,epoch):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            #torch.save(model.state_dict(), self.path + 'best_val_epoch_'+str(self.counter)+'_epoch_'+str(epoch)+'.pt')
            self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        # if epoch%10==0:
        #     torch.save(model.state_dict(), self.path + 'epoch'+str(epoch)+'.pt')


    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            self.logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            torch.save(model.state_dict(), self.path + '/best_val_epoch.pt')
        torch.save(model.state_dict(), self.path + '/the_final_epoch.pt')
        self.val_loss_min = val_loss

def split_bacth_items(items,popular):
    G1,G2=[],[]
    items_sorted=list(np.array(items)[np.argsort(np.array(popular)[items])])
    num=int(len(items_sorted)/2)
    G1.extend(items_sorted[0:num]) # 不热门
    G2.extend(items_sorted[num:]) # 热门
    return np.array(G1),np.array(G2)

def alignment_user(x, y):
    x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
    return (x - y).norm(p=2, dim=1).pow(2).mean()

def alignment_user_weighted(x, y, c_i, c_j, eps=1e-8):
    """
    带权对齐：在基础对齐损失上乘以基于流行度的系数 lambda。
    lambda = max(0, r(x, y)) / (log(c_i) * log(c_j))
    其中 r 使用余弦相似度，并在计算时对梯度切断，避免“作弊”。
    """
    x_norm = F.normalize(x, dim=-1)
    y_norm = F.normalize(y, dim=-1)
    base_loss = (x_norm - y_norm).norm(p=2, dim=1).pow(2)

    # 计算权重（与梯度分离，避免模型通过 r 直接优化）
    r_xy = F.cosine_similarity(x.detach(), y.detach(), dim=-1)
    pop_term = torch.log(c_i.float() + eps) * torch.log(c_j.float() + eps) + eps
    lambda_weight = torch.clamp(r_xy, min=0.0) / pop_term

    return (base_loss * lambda_weight).mean()

def build_global_pop_mask(popular_counts, split_ratio=0.5):
    idx_sorted = np.argsort(np.array(popular_counts))
    n = len(idx_sorted)
    half = int(n * split_ratio)
    item_is_pop = np.zeros(n, dtype=np.int8)
    pop_idx = idx_sorted[half:]
    item_is_pop[pop_idx] = 1
    return item_is_pop
