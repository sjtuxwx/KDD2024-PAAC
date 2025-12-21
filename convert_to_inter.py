#!/usr/bin/env python3
import os
import argparse

def read_lines(path):
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield line

def parse_u2i(line):
    parts = line.split()
    if len(parts) <= 1:
        return None, []
    uid = int(parts[0])
    items = [int(x) for x in parts[1:]]
    return uid, items

def convert_file(in_path, out_path, rating, timestamp):
    with open(out_path, 'w') as w:
        w.write('user_id:token\titem_id:token\trating:float\ttimestamp:float\n')
        for line in read_lines(in_path):
            uid, items = parse_u2i(line)
            if uid is None or not items:
                continue
            for it in items:
                w.write(f'{uid}\t{it}\t{rating:.1f}\t{timestamp:.1f}\n')

def convert_dataset(root, name, out_dir, rating, timestamp):
    ddir = os.path.join(root, name)
    train_in = os.path.join(ddir, 'train.txt')
    test_in = os.path.join(ddir, 'test.txt')
    if not os.path.isfile(train_in):
        return False
    os.makedirs(out_dir, exist_ok=True)
    train_out = os.path.join(out_dir, f'{name}_train.inter')
    convert_file(train_in, train_out, rating, timestamp)
    if os.path.isfile(test_in):
        test_out = os.path.join(out_dir, f'{name}_test.inter')
        convert_file(test_in, test_out, rating, timestamp)
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='OOD_Data')
    ap.add_argument('--dataset', default=None)
    ap.add_argument('--out', default='transfered_data')
    ap.add_argument('--rating', type=float, default=1.0)
    ap.add_argument('--timestamp', type=float, default=0.0)
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
    converted = []
    for n in names:
        ok = convert_dataset(args.root, n, args.out, args.rating, args.timestamp)
        if ok:
            converted.append(n)
    if converted:
        print('converted', ' '.join(converted))
    else:
        print('no_converted')

if __name__ == '__main__':
    main()

