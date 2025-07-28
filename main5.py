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
from model.dualgcn1 import DualGCN
from utils.evaluator import cluster_accuracy
from utils.utils import compute_adjacent_matrix
from model.gcsc import GCSC
from model.gcsck import GCSCKernel

# 定义颜色表，用于可视化
color_tabel = np.array([[255, 254, 137], [3, 28, 241], [255, 89, 1], [5, 255, 133],
                        [255, 2, 251], [89, 1, 255], [3, 171, 255], [12, 255, 7],
                        [172, 175, 84], [160, 78, 158], [101, 173, 255], [60, 91, 112],
                        [104, 192, 63], [139, 69, 46], [119, 255, 172], [254, 255, 3]])
# 将颜色表归一化到0-1之间
color_tabel = minmax_scale(color_tabel, feature_range=(0, 1))


# 从给定的配置文件路径中加载配置信息
def get_config(config_path):
    f = open(config_path, 'r')
    return json.load(f)


# 训练函数
def train(config, model: DualGCN, dataloader_train,adj2, optimizer, loss_curve: list, loss_func=None):
    '''
    ## 训练函数
    ---
    loss_curve: loss曲线
    loss_func: 如果提供了就用这个求loss,否则用默认loss求
    '''
    device = config['device']
    train_loss = 0.0

    model.train()
    for emp_feat, spe_feat, label in dataloader_train:  # for循环只执行一次（一张大图）

        spe_feat = spe_feat.float().to(device)
        # 将标签转移到设备上，并转换为long类型
        label = label.long().to(device)

        # 计算模型的输出
        emb2, diff2 = model(spe_feat,adj2)  # 2个返回值都是对比学习中间结果
        # 如果没有提供损失函数，就使用模型的默认损失函数计算损失
        if loss_func is None:
            loss = model.loss_func(emb2, diff2)
        else:
            # 否则，使用提供的损失函数计算损失
            loss = loss_func(emb2, diff2)

        optimizer.zero_grad()
        # ##
        # 更新动量编码器
        model.update_momentum_encoder()

        # 反向传播计算梯度
        loss.backward()
        # 使用优化器进行参数更新
        optimizer.step()

        # 累加训练损失
        train_loss += loss.item()
    # 将训练损失添加到损失曲线中
    loss_curve.append(train_loss)
    # 返回训练损失
    return train_loss


# 对聚类模型进行评估
def cluster_eval(config, model: DualGCN, cluster, dataloader_train, adj2, acc_curve: list):
    device = config['device']
    with torch.no_grad():
        # 将模型切换到评估模式，避免在评估时计算梯度
        model.eval()
        # 遍历训练数据，这里只会循环一次，因为dataloader_train是一张大图的数据
        for emp_feat, spe_feat, label in dataloader_train:  # 只会循环一次
            # emp_feat = emp_feat.float().to(device)
            spe_feat = spe_feat.float().to(device)
            label = label.long().to(device)
            # 获取模型的嵌入表示
            embs = model.embs(spe_feat,adj2)   # 这个地方是加入了注意力机制，通过注意力机制获取的新的图嵌入表示
    # 将嵌入表示转移到CPU，并转换为numpy数组
    embs = embs.cpu().numpy()

    # 将标签转移到CPU，并转换为numpy数组
    true = label.cpu().numpy()

    # 使用聚类模型对嵌入表示进行聚类
    cluster_pred = cluster.fit_predict(embs)

    # 计算聚类的精度和类别的精度
    cluster_acc, class_acc, cluster_pred = cluster_accuracy(true, cluster_pred)

    # 将聚类精度和类别精度合并到一个列表中
    cluster_acc.extend(class_acc)  # 为了方便返回把聚类精度和类别精度放一个list中
    # 将精度结果添加到acc_curve中，用于后续绘制精度曲线
    acc_curve.append(cluster_acc)

    # 返回聚类精度、真实标签和聚类预测结果，用于可视化
    return cluster_acc, true, cluster_pred  # 返回true和pred是为了画图


# 可视化函数，用于绘制真实标签和聚类预测结果的可视化图像
def visualization(points, shape, true, pred, save_path=None):
    fig_true = np.zeros((shape[0], shape[1], 3))
    fig_pred = np.zeros((shape[0], shape[1], 3))
    colors_true = [color_tabel[true[i]] for i in range(true.shape[0])]
    colors_pred = [color_tabel[pred[i]] for i in range(pred.shape[0])]
    for i in range(points.shape[0]):
        p = points[i]
        fig_true[p[0], p[1], :] = colors_true[i]
        fig_pred[p[0], p[1], :] = colors_pred[i]

    # 创建图像窗口
    plt.figure(figsize=(10, 7))
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(fig_true)
    ax1.set_title('true')
    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(fig_pred)
    ax2.set_title('predict')
    # 如果提供了保存路径，则保存图像到指定路径，并打印保存信息
    if save_path is not None:
        plt.savefig(save_path)
        print('### Save figure in {}. ###'.format(save_path))
    # 显示图像
    plt.show()


# 主函数，用于模型训练和评估
def main(config, dataset_name, pretrained=None):

    # 获取设备信息
    device = torch.device(config['device'])

    # data 是全集，# 加载数据集并获取类别数量
    train_data, num_classes = split_dataset(config, dataset_name, shuffle=False)
    dataloader_train = DataLoader(train_data, shuffle=False, batch_size=len(train_data))  # 全图一起
    # 如果已经生成了邻接矩阵且不需要重新生成，则直接加载邻接矩阵//如果有直接加载, 没有就生成一下adj
    if os.path.exists(config['save_path'] + '/' + dataset_name + '_adj1.npy') and not config['regenerate_mat']:
        print('Loading matrix.')
        # adj1 = np.load(config['save_path'] + '/' + dataset_name + '_adj1.npy')
        adj2 = np.load(config['save_path'] + '/' + dataset_name + '_adj2.npy')
    else:
        # 否则，重新生成邻接矩阵
        print('Generating matrix.')
        for emp_feat, spe_feat, __ in dataloader_train:
            # adj1 = compute_adjacent_matrix(emp_feat.numpy(), config['n_neighbors'])
            adj2 = compute_adjacent_matrix(spe_feat.numpy(), config['n_neighbors'])
            # np.save(config['save_path'] + '/' + dataset_name + '_adj1.npy', adj1)
            np.save(config['save_path'] + '/' + dataset_name + '_adj2.npy', adj2)
    # adj1 = torch.from_numpy(adj1).float().to(device)
    adj2 = torch.from_numpy(adj2).float().to(device)
    # 创建DualGCN模型，并将其切换到设备上
    model = DualGCN(config, num_classes)
    model.to(device)
    print(model)

    # [define cluster methods]
    REG_Coef_, NEIGHBORING_, RO_ = 1e2, 25, 0.4
    REG_Coef_K, NEIGHBORING_K, RO_K, GAMMA = 1e2, 25, 0.8, 10

    # 初始化聚类模型和优化器
    gcsc = GCSC(num_classes, REG_Coef_, NEIGHBORING_, RO_)
    gcsck = GCSCKernel(num_classes, REG_Coef_K, NEIGHBORING_K, 'rbf', GAMMA, RO_K)
    kmeans = KMeans(num_classes, random_state=0, n_init=10)
    # [end define]

    # 打印当前使用的聚类模型
    print(
        '==> train cluster model: {} | eval cluster model: {}'.format(config['train_cluster'], config['eval_cluster']))
    # 初始化损失函数和优化器
    loss_func = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    loss_curve = []  # 保存一下训练时的loss曲线
    acc_curve = []
    tr_start = time.time()  # 训练开始的时间戳

    # 如果提供了预训练模型，则直接测试并可视化结果
    if pretrained is not None:  # 使用预训练模型就直接测试完返回结果并可视化
        print('Loading pretrained model.')
        model.load_state_dict(torch.load(pretrained))
        print('==> [test]')
        cluster_acc, true, cluster_pred = cluster_eval(config, model, eval(config['eval_cluster']), dataloader_train,
                                                       adj2, acc_curve)
        loss_curve = [-1]
        visualization(train_data.points, train_data.gt.shape[:2], true, cluster_pred,
                      config['result_path'] + '/' + dataset_name + '_fig.png')
        return loss_curve, acc_curve, time.time() - tr_start

    # 正常训练
    for epoch in range(config['num_epoches']):
        ep_start = time.time()  # 该epoch开始的时间戳

        # [train start]训练模型
        print('==> Epoch: %03d[train]' % epoch)
        train_loss = train(config, model, dataloader_train, adj2, optimizer, loss_curve, None)
        # [train end]

        # [eval start]评估模型
        print('==> Epoch: %03d[eval]' % epoch)
        cluster_acc, __, __ = cluster_eval(config, model, eval(config['train_cluster']), dataloader_train,adj2,
                                           acc_curve)

        # [eval end]打印当前训练的结果
        print("==> Epoch: %03d: train_loss=%.5f, OA=%.5f, Kappa=%.5f, NMI=%.5f, time=%ds"
              % (epoch, train_loss, cluster_acc[0], cluster_acc[1], cluster_acc[2], time.time() - ep_start))

    if config['save_model']:  # 保存训练结果
        save_path = config['save_path'] + '/' + dataset_name + '_model.pkl'
        torch.save(model.state_dict(), save_path)
        print('### Save model in {}. ###'.format(save_path))

    return loss_curve, acc_curve, time.time() - tr_start


# 将训练结果保存为CSV文件
def report_csv(loss_curve, acc_curve, save_path):
    n_class = len(acc_curve[0]) - 3
    epoches = len(acc_curve)
    columns = ['loss', 'OA', 'Kappa', 'NMI'] + [str(i) for i in range(n_class)]
    index = ['Epoch ' + str(i) for i in range(epoches)]
    data = np.concatenate((np.array(loss_curve).reshape(-1, 1), np.array(acc_curve)), axis=1)
    df = pd.DataFrame(data, index=index, columns=columns)
    df.to_csv(save_path)


if __name__ == '__main__':
    # 获取配置信息
    config = get_config('./config.json')

    # 读取命令行参数，获取数据集名称和预训练模型路径
    argc, argv = len(sys.argv), sys.argv  # 对命令行操作的支持
    dataset = 'Indian_pines'
    pretrained = None
    if argc == 2:  # 参数为1不管，参数大于2忽略后面的参数
        dataset = argv[1]
    elif argc == 3:
        dataset = argv[1]
        pretrained = config['save_path'] + '/' + dataset + '_' + argv[2]
    # 如果未提供预训练模型，则进行模型训练
    if pretrained is None:  # 如果为None进行训练
        loss_curve, acc_curve, __ = main(config, dataset)
        dt = datetime.fromtimestamp(time.time())
        report_csv(loss_curve, acc_curve,
                   config['result_path'] + '/' + dataset + '_{}.csv'.format(dt.strftime('%Y_%m_%d_%H_%M_%S')))
    # 如果提供了预训练模型，则直接测试预训练模型并打印OA的值
    pretrained = config['save_path'] + '/' + dataset + '_model.pkl'
    __, acc_curve, __ = main(config, dataset, pretrained)
    print(acc_curve[0])  # OA, Kappa, NMI, CA