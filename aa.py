#!/usr/bin/env python3
import os, argparse
import numpy as np
from scipy.stats import chisquare, ks_2samp

def read_u2i(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            parts = line.split()
            items = list(map(int, parts[1:]))
            data.append(items)
    return data

def item_counts(u2i):
    from collections import Counter
    c = Counter()
    total = 0
    for items in u2i:
        c.update(items)
        total += len(items)
    return c, total

def align_counts(c1, c2):
    keys = sorted(set(c1.keys()) | set(c2.keys()))
    a1 = np.array([c1.get(k, 0) for k in keys], dtype=np.float64)
    a2 = np.array([c2.get(k, 0) for k in keys], dtype=np.float64)
    return a1, a2

def jsd(p, q, eps=1e-12):
    m = 0.5 * (p + q)
    def kl(x, y):
        return np.sum(x * np.log((x + eps) / (y + eps)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

def gini(x):
    s = x.sum()
    if s == 0: 
        return 0.0
    xs = np.sort(x)
    n = len(xs)
    cum = np.cumsum(xs)
    return (n + 1 - 2.0 * np.sum(cum) / cum[-1]) / n

def top_share(x, frac=0.01):
    n = len(x)
    if n == 0: 
        return 0.0
    k = max(1, int(n * frac))
    return float(np.sum(np.sort(x)[-k:])) / float(np.sum(x)) if np.sum(x) > 0 else 0.0

def analyze_dataset(root, name, topk=20):
    ddir = os.path.join(root, name)
    train_path = os.path.join(ddir, 'train.txt')
    test_path = os.path.join(ddir, 'test.txt')
    if not (os.path.isfile(train_path) and os.path.isfile(test_path)):
        return None
    train_u2i = read_u2i(train_path)
    test_u2i = read_u2i(test_path)
    c_train, n_train = item_counts(train_u2i)
    c_test, n_test = item_counts(test_u2i)
    a_train, a_test = align_counts(c_train, c_test)
    p_train = a_train / (a_train.sum() if a_train.sum() > 0 else 1)
    p_test  = a_test  / (a_test.sum()  if a_test.sum()  > 0 else 1)
    jsd_val = jsd(p_train, p_test)
    scale = (a_train.sum() / a_test.sum()) if a_test.sum() > 0 else 1.0
    expected = a_test * scale
    chi_res = chisquare(a_train, f_exp=expected) if a_train.sum() > 0 and a_test.sum() > 0 else None
    chi_p = chi_res.pvalue if chi_res is not None else 1.0
    train_lens = np.array([len(x) for x in train_u2i], dtype=np.int64)
    test_lens  = np.array([len(x) for x in test_u2i], dtype=np.int64)
    ks_res = ks_2samp(train_lens, test_lens) if len(train_lens) > 0 and len(test_lens) > 0 else None
    ks_p = ks_res.pvalue if ks_res is not None else 1.0
    g_train = gini(a_train)
    g_test  = gini(a_test)
    share_train = top_share(a_train, frac=0.01)
    share_test  = top_share(a_test, frac=0.01)
    non_iid = (jsd_val > 0.05) or (chi_p < 0.05) or (ks_p < 0.05)
    return {
        'dataset': name,
        'jsd': jsd_val,
        'chi_p': chi_p,
        'ks_p': ks_p,
        'gini_train': g_train,
        'gini_test': g_test,
        'top1pct_share_train': share_train,
        'top1pct_share_test': share_test,
        'non_iid': non_iid,
        'n_items_train': int((a_train > 0).sum()),
        'n_items_test': int((a_test  > 0).sum()),
        'n_inter_train': int(a_train.sum()),
        'n_inter_test': int(a_test.sum())
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='OOD_Data')
    ap.add_argument('--dataset', default=None)
    args = ap.parse_args()
    names = []
    if args.dataset:
        names = [args.dataset]
    else:
        if not os.path.isdir(args.root):
            print('missing_root', args.root)
            return
        for n in os.listdir(args.root):
            if os.path.isdir(os.path.join(args.root, n)):
                names.append(n)
    results = []
    for n in names:
        r = analyze_dataset(args.root, n)
        if r is not None:
            results.append(r)
    for r in results:
        print('dataset=', r['dataset'])
        print('jsd=', round(r['jsd'], 6), 'chi_p=', round(r['chi_p'], 6), 'ks_p=', round(r['ks_p'], 6))
        print('gini_train=', round(r['gini_train'], 6), 'gini_test=', round(r['gini_test'], 6))
        print('top1pct_share_train=', round(r['top1pct_share_train'], 6), 'top1pct_share_test=', round(r['top1pct_share_test'], 6))
        print('n_items_train=', r['n_items_train'], 'n_items_test=', r['n_items_test'], 'n_inter_train=', r['n_inter_train'], 'n_inter_test=', r['n_inter_test'])
        print('non_iid=', r['non_iid'])
        print('-----')
    if not results:
        print('no_datasets')

if __name__ == '__main__':
    main()