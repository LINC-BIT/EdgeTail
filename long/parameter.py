


Resampling =0#是否重采样
Resampling_way = '平方根重采样' #重采样方法

Augmentation =4#是否数据增强
Augmentation_way = 'CMO'

Loss = 0  #是否优化损失函数
Loss_way = 'FocalLoss'

Logit = 0  #是否进行logit调整
Logit_way= 'loss_modification'

Pretrain = 0  #是否进行预训练
Pretrain_way = 'pretrained_resnet_cifar10.pth'
#Pretrain_way = 'pretrained_vit_cifar10.pth'

samplingstrategy = sampling_strategy = {
    0: 5000,
    1: 4500,
    2: 4000,
    3: 3500,
    4: 3000,
    5: 2500,
    6: 2000,
    7: 1500,
    8: 1000,
    9: 500
}#'auto'#'minority'#过采样因子/欠采样因子
oversample_factor = 1.2  # 可以调整为所需的值
undersample_factor = 1.0 # 可以调整为所需的值
alpha = 3  #增强强度
Augmentation_probalitlity = 1  #增强概率

EPOCH = 20 # 遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
BATCH_SIZE = 64  # 批处理尺寸(batch_size)
LR = 0.0072  # 学习率
start_data_aug=1
end_data_aug=1
mixup_prob=1.0
beta=1.0
weighted_alpha=0.8

cls_num=10 