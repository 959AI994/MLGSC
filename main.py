'''
# 训练与测试文件
'''
import os
import sys
import time
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans

from dataset import split_dataset
from model.dualgcn import DualGCN
from utils.evaluator import cluster_accuracy
from model.gcsc import GCSC
from model.gcsck import GCSCKernel
from utils.utils import compute_adjacent_matrix, compute_ppr


def get_config(config_path):
    f = open(config_path, 'r')
    return json.load(f)


def train_test(config, model: DualGCN, cluster, dataloader_train, adj1, adj2, diff1, diff2, 
               optimizer, loss_func, loss_curve: list, acc_curve: list):
    '''
    ## 训练测试函数
    ---
    loss_func: loss函数
    loss_curve: loss曲线
    acc_curve: acc曲线 包括OA,Kappa,NMI,CA
    ---
    return: 此epoch的loss, 此epoch测试acc
    '''
    device = config['device']
    train_loss = 0.0
    true = []
    embs = []

    model.train()
    for emp_feat, spe_feat, label in dataloader_train:      # for循环只执行一次（一张大图）

        emp_feat = emp_feat.float().to(device)
        spe_feat = spe_feat.float().to(device)
        label = label.long().to(device)
        
        sample_size = label.size(0)
        lbl_1 = torch.ones(1, sample_size * 2)
        lbl_2 = torch.zeros(1, sample_size * 2)
        lbl = torch.cat((lbl_1, lbl_2), 1).to(config['device'])              # 正例反例

        # idx = np.random.permutation(emp_feat.size(1))                        # 打乱feature
        # ba, bd, bf = [], [], []
        # for i in idx:
        #     ba.append(adj[i: i + sample_size, i: i + sample_size])      # 原adj取样
        #     bd.append(diff[i: i + sample_size, i: i + sample_size])     # diff取样
        #     bf.append(features[i: i + sample_size])                     # feature取样

        # 聚类任务，output直接忽略，emb是经过attention后的结果，emb1和2分别是两条分支GCN的结果
        output, emb, logits1, logits2 = model(emp_feat, spe_feat, adj1, adj2, diff1, diff2)
        embs.append(emb.detach().cpu().numpy())
        true.append(label.cpu().numpy())
        loss = loss_func(logits1, lbl) + loss_func(logits2, lbl)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()


    embs = np.concatenate(embs)     # 保存所有emb
    true = np.concatenate(true)
    cluster_acc = test(embs, true, cluster)
    
    loss_curve.append(train_loss)
    acc_curve.append(cluster_acc)

    return train_loss, cluster_acc


def test(embs, true, cluster):  
    print('==> [cluster] cluster model: %s' % str(type(cluster)))
    cluster_pred = cluster.fit_predict(embs)
    cluster_acc, class_acc = cluster_accuracy(true, cluster_pred)
    cluster_acc.extend(class_acc)       # 为了方便返回把聚类精度和类别精度放一个list中
    # print('=================================\n'
    #       '\t\tCLUSTER RESULTS\n'
    #       '=================================')
    # print('%10s %10s %10s' % ('OA', 'Kappa', 'NMI',))
    # print('%10.4f %10.4f %10.4f' % (cluster_acc[0], cluster_acc[1], cluster_acc[2]))
    return cluster_acc


def main(config, dataset_name, pretrained=None):
    device = torch.device(config['device'])

    # data 是全集
    train_data, num_classes = split_dataset(config, dataset_name, shuffle=False)
    dataloader_train = DataLoader(train_data, shuffle=False, batch_size=len(train_data))   # 全图一起
    # 如果有直接加载,没有就生成一下adj和diff
    if os.path.exists(config['save_path'] + '/' + dataset_name + '_adj1.npy') and not config['regenerate_mat']:  
        print('Loading matrix.')
        adj1  = np.load(config['save_path'] + '/' + dataset_name + '_adj1.npy')
        adj2  = np.load(config['save_path'] + '/' + dataset_name + '_adj2.npy')
        diff1 = np.load(config['save_path'] + '/' + dataset_name + '_diff1.npy')
        diff2 = np.load(config['save_path'] + '/' + dataset_name + '_diff2.npy')
    else:
        print('Generating matrix.')
        for emp_feat, spe_feat, label in dataloader_train:
            adj1 = compute_adjacent_matrix(emp_feat.numpy(), config['n_neighbors'])
            adj2 = compute_adjacent_matrix(spe_feat.numpy(), config['n_neighbors'])
            diff1 = compute_ppr(adj1)
            diff2 = compute_ppr(adj2)
            np.save(config['save_path'] + '/' + dataset_name + '_adj1.npy', adj1)
            np.save(config['save_path'] + '/' + dataset_name + '_adj2.npy', adj2)
            np.save(config['save_path'] + '/' + dataset_name + '_diff1.npy', diff1)
            np.save(config['save_path'] + '/' + dataset_name + '_diff2.npy', diff2)

    model = DualGCN(config, num_classes)
    model.to(device)

    # cluster = GCSC(num_classes, REG_Coef_, NEIGHBORING_, RO_)
    cluster = KMeans(num_classes, random_state=0, n_init='auto')
    loss_func = torch.nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    loss_curve = [ ]            # 保存一下训练时的loss曲线
    acc_curve  = [ ]
    best_oa    = 0.0            # 模型保存的依据：OA
    tr_start = time.time()          # 训练开始的时间戳

    if pretrained is not None:      # 使用预训练模型就直接测试完返回结果
        model.load_state_dict(pretrained)
        model.eval()
        true = []
        embs = []
        for emp_feat, spe_feat, label in dataloader_train:      # for循环只执行一次（一张大图）

            emp_feat = emp_feat.float().to(device)
            feat = feat.float().to(device)
            label = label.long().to(device)

            output, emb, __, __ = model(emp_feat, spe_feat, adj1, adj2, diff1, diff2)
            embs.append(emb.detach().cpu().numpy())
            true.append(label.cpu().numpy())

        embs = np.concatenate(embs)     # 保存所有emb
        true = np.concatenate(true)
        cluster_acc = test(embs, true, cluster)
        acc_curve.append(cluster_acc)
        loss_curve = [-1] * len(embs)   # 使用-1填补空位
        return acc_curve, loss_curve, time.time() - tr_start

    # 正常训练
    for epoch in range(config['num_epoches']):
        ep_start = time.time()      # 该epoch开始的时间戳

        # [train start]
        print('==> Epoch: %03d[train]' % epoch)
        train_loss, cluster_acc = train_test(config, model, cluster, dataloader_train, adj1, adj2, diff1, diff2, 
                                             optimizer, loss_func, loss_curve, acc_curve)
        # [train end]

        print("==> Epoch: %03d: train_loss=%.5f, OA=%.5f, Kappa=%.5f, NMI=%.5f, time=%ds"
            % (epoch, train_loss, cluster_acc[0], cluster_acc[1], cluster_acc[2], time.time() - ep_start))
        
        if config['save_model'] and cluster_acc[0] > best_oa:   # 保存OA最高的参数
            best_oa = cluster_acc[0]
            savepath = config['save_path'] + '/model.pkl'
            torch.save(model.state_dict(), savepath)
            print('### Save model in {}. ###' % savepath)
    
    return loss_curve, acc_curve, time.time() - tr_start

def report_csv(loss_curve, acc_curve, save_path):
    import pandas as pd
    n_class = len(acc_curve[0]) - 3
    epoches = len(acc_curve)
    columns = ['loss', 'OA', 'Kappa', 'NMI'] + [str(i) for i in range(n_class)]
    index = ['Epoch ' + str(i) for i in range(epoches)]
    data = np.concatenate((np.array(loss_curve).reshape(-1, 1), np.array(acc_curve)), axis=1)
    df = pd.DataFrame(data, index=index, columns=columns)
    df.to_csv(save_path)


if __name__ == '__main__':
    config = get_config('./config.json')

    argc, argv = len(sys.argv), sys.argv        # 对命令行操作的支持
    dataset = 'Indian_pines'
    pretrained = None
    if argc == 2:                               # 参数为1不管，参数大于2忽略后面的参数
        dataset = argv[1]
    elif argc == 3:
        dataset = argv[1]
        pretrained = config['save_path'] + '/' + argv[2]
    

    loss_curve, acc_curve, __ = main(config, dataset, pretrained)
    report_csv(loss_curve, acc_curve, config['result_path'] + '/info.csv')

