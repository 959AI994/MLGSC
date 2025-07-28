'''
# 训练与测试文件
'''
import os
import sys
import time
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from datetime import datetime
from sklearn.preprocessing import minmax_scale

from dataset import split_dataset
from model.dualgcn2 import DualGCN
from utils.evaluator import cluster_accuracy
from utils.utils import compute_adjacent_matrix
from model.gcsc import GCSC
from model.gcsck import GCSCKernel

# 创建颜色表，color_tabel 是一个包含颜色值的数组
color_tabel = np.array([[255, 254, 137], [3,  28,  241], [255, 89,    1], [5,   255, 133],
                        [255,   2, 251], [89,  1,  255], [3,   171, 255], [12,  255,   7],
                        [172, 175,  84], [160, 78, 158], [101, 173, 255], [60,   91, 112],
                        [104, 192,  63], [139, 69,  46], [119, 255, 172], [254, 255,   3]])
# 将颜色表进行归一化，将颜色值的范围映射到(0, 1)之间
color_tabel = minmax_scale(color_tabel, feature_range=(0, 1))

# 获取配置信息的函数，读取指定路径的配置文件并返回配置信息
def get_config(config_path):
    f = open(config_path, 'r')
    return json.load(f)


def train(config, model: DualGCN, dataloader_train, optimizer, loss_curve: list, loss_func=None):
    '''
    ## 训练函数
    ---
    loss_curve: loss曲线
    loss_func: 如果提供了就用这个求loss,否则用默认loss求
    '''
    device = config['device']
    train_loss = 0.0
    i = 0
    model.train()
    # 使用 tqdm 在训练过程中显示进度条，循环执行 dataloader_train 中的数据批次
    for emp_feat, spe_feat, label in tqdm(dataloader_train):      # for循环只执行一次（一张大图）

        emp_feat = emp_feat.float().to(device)
        spe_feat = spe_feat.float().to(device)
        label = label.long().to(device)

        # 根据 emp_feat 和 spe_feat 计算邻接矩阵 adj1 和 adj2
        adj1 = compute_adjacent_matrix(emp_feat.numpy(), config['n_neighbors'])
        adj2 = compute_adjacent_matrix(spe_feat.numpy(), config['n_neighbors'])
        adj1 = torch.from_numpy(adj1).float().to(device)
        adj2 = torch.from_numpy(adj2).float().to(device)

        # 调用 model 的 forward 方法进行前向传播，得到嵌入结果 emb1, emb2, diff1, diff2
        emb1, emb2, diff1, diff2 = model(emp_feat, spe_feat, adj1, adj2)    # 四个返回值都是对比学习中间结果
        if loss_func is None:
            # 若没有提供 loss_func，则调用 model 的 loss_func 方法计算损失
            loss = model.loss_func(emb1, emb2, diff1, diff2)
        else:
            # 若提供了 loss_func，则分别对 emb1, emb2 计算损失并取平均
            loss = (loss_func(emb1, diff1) + loss_func(emb2, diff2)) / 2
        # 清零优化器的梯度
        optimizer.zero_grad()
        # 反向传播计算梯度
        loss.backward()
        # 使用优化器进行参数更新
        optimizer.step()
        # 累加训练损失
        train_loss += loss.item()
        i += 1
    train_loss = train_loss / i
    # 将训练损失添加到损失曲线中
    loss_curve.append(train_loss)
    return train_loss

# 对聚类模型进行评估
def cluster_eval(config, model: DualGCN, cluster, dataloader, acc_curve: list):
    device = config['device']
    # 将模型切换到评估模式，避免在评估时计算梯度
    model.eval()
    embs = []  # 存储嵌入向量
    true = []  # 存储真实标签
    with torch.no_grad():
        # 使用 tqdm 在评估过程中显示进度条，循环执行 dataloader 中的数据批次
        for emp_feat, spe_feat, label in tqdm(dataloader):
            emp_feat = emp_feat.float().to(device)
            spe_feat = spe_feat.float().to(device)
            label = label.long().to(device)

            # 计算邻接矩阵 adj1 和 adj2
            adj1 = compute_adjacent_matrix(emp_feat.numpy(), config['n_neighbors'])
            adj2 = compute_adjacent_matrix(spe_feat.numpy(), config['n_neighbors'])
            adj1 = torch.from_numpy(adj1).float().to(device)
            adj2 = torch.from_numpy(adj2).float().to(device)

            # 调用 model 的 embs 方法进行前向传播，得到嵌入向量 emb
            emb = model.embs(emp_feat, spe_feat, adj1, adj2)
            embs.append(emb.cpu().numpy())
            true.append(label.cpu().numpy())

    # 将所有嵌入向量连接成一个大的数组，保存所有的 emb
    embs = np.concatenate(embs)     # 保存所有emb
    # 将所有真实标签连接成一个大的数组
    true = np.concatenate(true)
    # 对嵌入向量进行聚类预测
    cluster_pred = cluster.fit_predict(embs)
    # 计算聚类精度和类别精度
    cluster_acc, class_acc, cluster_pred = cluster_accuracy(true, cluster_pred)
    cluster_acc.extend(class_acc)           # 为了方便返回把聚类精度和类别精度放一个list中
    # 将聚类精度和类别精度的结果添加到 acc_curve 中
    acc_curve.append(cluster_acc)

    # 返回 true 和 cluster_pred 是为了后续可视化聚类结果
    return cluster_acc, true, cluster_pred  # 返回true和pred是为了画图


def visualization(points, shape, true, pred, save_path=None):
    # 创建一个形状为 shape 的全零数组，用于存储 true 真实标签的颜色映射图
    fig_true = np.zeros((shape[0], shape[1], 3))
    # 创建一个形状为 shape 的全零数组，用于存储 pred 预测标签的颜色映射图
    fig_pred = np.zeros((shape[0], shape[1], 3))
    # 获取 true 真实标签的颜色映射，根据 true 中每个标签的值来映射为对应的颜色
    colors_true = [color_tabel[true[i]] for i in range(true.shape[0])]
    # 获取 pred 预测标签的颜色映射，根据 pred 中每个标签的值来映射为对应的颜色
    colors_pred = [color_tabel[pred[i]] for i in range(pred.shape[0])]
    for i in range(points.shape[0]):
        p = points[i]
        # 将 true 真实标签对应的颜色映射值填充到 fig_true 的对应位置
        fig_true[p[0], p[1], :] = colors_true[i]
        # 将 pred 预测标签对应的颜色映射值填充到 fig_pred 的对应位置
        fig_pred[p[0], p[1], :] = colors_pred[i]

    # 创建一个大小为 10x7 的新图形
    plt.figure(figsize=(10, 7))
    # 创建子图，1 行 2 列的子图，当前为第一个子图
    ax1 = plt.subplot(1, 2, 1)
    # 在第一个子图中显示真实标签的颜色映射图
    ax1.imshow(fig_true)
    # 设置第一个子图的标题为 'true'
    ax1.set_title('true')
    # 创建子图，1 行 2 列的子图，当前为第二个子图
    ax2 = plt.subplot(1, 2, 2)
    # 在第二个子图中显示预测标签的颜色映射图
    ax2.imshow(fig_pred)
    # 设置第二个子图的标题为 'predict'
    ax2.set_title('predict')
    # 如果提供了保存路径 save_path，则保存图形到该路径
    if save_path is not None:
        # 将图形保存到指定路径 save_path
        plt.savefig(save_path)
        print('### Save figure in {}. ###'.format(save_path))
    # 显示图形
    plt.show()
    

def main(config, dataset_name, pretrained=None):
    device = torch.device(config['device'])

    # data 是全集
    train_data, num_classes = split_dataset(config, dataset_name, shuffle=False)
    dataloader_train = DataLoader(train_data, shuffle=False, batch_size=config['batch_size'])   # 按batch来

    model = DualGCN(config, num_classes)
    model.to(device)

   # [define cluster methods]
    REG_Coef_, NEIGHBORING_, RO_ = 1e2, 25, 0.4
    REG_Coef_K, NEIGHBORING_K, RO_K, GAMMA = 1e2, 25, 0.8, 10
    gcsc = GCSC(num_classes, REG_Coef_, NEIGHBORING_, RO_)
    gcsck = GCSCKernel(num_classes, REG_Coef_K, NEIGHBORING_K, 'rbf', GAMMA, RO_K)
    kmeans = KMeans(num_classes, random_state=0, n_init=10)  #n_init是整型，需要设置参数
    # [end define]

    print('==> train cluster model: {} | eval cluster model: {}'.format(config['train_cluster'], config['eval_cluster']))
    loss_func = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    loss_curve = [ ]                # 保存一下训练时的loss曲线
    acc_curve  = [ ]
    tr_start = time.time()          # 训练开始的时间戳

    if pretrained is not None:      # 使用预训练模型就直接测试完返回结果并可视化
        print('Loading pretrained model.')
        model.load_state_dict(torch.load(pretrained))
        print('==> [test]')
        cluster_acc, true, cluster_pred = cluster_eval(config, model, eval(config['eval_cluster']), dataloader_train, acc_curve)
        loss_curve = [-1]
        visualization(train_data.points, train_data.gt.shape[:2], true, cluster_pred, 
                      config['result_path'] + '/' + dataset_name + '_fig.png')
        return loss_curve, acc_curve, time.time() - tr_start

    # 正常训练
    for epoch in range(config['num_epoches']):
        ep_start = time.time()      # 该epoch开始的时间戳

        # [train start]
        print('==> Epoch: %03d[train]' % epoch)
        train_loss = train(config, model, dataloader_train, optimizer, loss_curve, None)
        # [train end]

        # [eval start]
        print('==> Epoch: %03d[eval]' % epoch)
        cluster_acc, __, __ = cluster_eval(config, model, eval(config['train_cluster']), dataloader_train, acc_curve)
        # [eval end]
        print("==> Epoch: %03d: train_loss=%.5f, OA=%.5f, Kappa=%.5f, NMI=%.5f, time=%ds"
            % (epoch, train_loss, cluster_acc[0], cluster_acc[1], cluster_acc[2], time.time() - ep_start))
        
    if config['save_model']:    # 保存训练结果
        save_path = config['save_path'] + '/' + dataset_name + '_model.pkl'
        torch.save(model.state_dict(), save_path)
        print('### Save model in {}. ###'.format(save_path))
    
    return loss_curve, acc_curve, time.time() - tr_start

def report_csv(loss_curve, acc_curve, save_path):
    n_class = len(acc_curve[0]) - 3
    epoches = len(acc_curve)
    columns = ['loss', 'OA', 'Kappa', 'NMI'] + [str(i) for i in range(n_class)]
    index = ['Epoch ' + str(i) for i in range(epoches)]
    data = np.concatenate((np.array(loss_curve).reshape(-1, 1), np.array(acc_curve)), axis=1)
    df = pd.DataFrame(data, index=index, columns=columns)
    df.to_csv(save_path)


if __name__ == '__main__':
    config = get_config('./config.json')
    # 获取命令行参数的个数和列表，存储在 argc 和 argv 变量中
    argc, argv = len(sys.argv), sys.argv        # 对命令行操作的支持
    dataset = 'Indian_pines'
    pretrained = None
    # 如果命令行参数个数为 2，说明提供了一个数据集名称
    if argc == 2:                               # 参数为1不管，参数大于2忽略后面的参数
        # 将第一个命令行参数作为数据集名称赋值给 dataset 变量
        dataset = argv[1]
    # 如果命令行参数个数为 3，说明提供了数据集名称和预训练模型的索引参数
    elif argc == 3:
        dataset = argv[1]
        pretrained = config['save_path'] + '/' + dataset + '_' + argv[2]
    
    if pretrained is None:      # 如果为None进行训练
        # 调用 main 函数进行模型训练，得到 loss 曲线和聚类精度曲线
        loss_curve, acc_curve, __ = main(config, dataset)
        # 获取当前时间戳，用于生成结果文件的名称
        dt = datetime.fromtimestamp(time.time())
        # 将 loss 曲线和聚类精度曲线保存为 csv 文件
        report_csv(loss_curve, acc_curve, config['result_path'] + '/' + dataset + '_{}.csv'.format(dt.strftime('%Y_%m_%d_%H_%M_%S')))
    # 使用预训练模型进行测试和可视化
    pretrained = config['save_path'] + '/' + dataset + '_model.pkl'  # 获取预训练模型的路径
    __, acc_curve, __ = main(config, dataset, pretrained)  # 调用 main 函数进行模型测试，得到聚类精度曲线
    # 输出聚类精度结果中的第一个元素，即 OA（Overall Accuracy）、Kappa、NMI（Normalized Mutual Information）和 CA（Class Accuracy）
    print(acc_curve[0])     # OA, Kappa, NMI, CA

