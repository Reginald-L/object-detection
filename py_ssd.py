import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
from math import sqrt as sqrt
from itertools import product as product
import argparse
from d2l import torch as d2l
import os

parser = argparse.ArgumentParser()
parser.add_argument('--resume', type=bool, default=False, help='是否从上个节点继续训练')
parser.add_argument('--epochs', type=int, default=20, help='训练epochs')
parser.add_argument('--batch_size', type=int, default=16, help='批量大小')
parser.add_argument('--lr', type=float, default=0.1, help='学习率')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight_decay for SGD')
parser.add_argument('--img_size', type=int, default=300, help='输入的图像大小')
parser.add_argument('--num_class', type=int, default=1, help='分类类别个数')
parser.add_argument('--save_checkpoint_path', type=str, default='checkpoints', help='checkpoint文件保存路径')
parser.add_argument('--gpu', type=bool, default=True, help='是否使用GPU')

opt = parser.parse_args()

cuda = True if opt.gpu and torch.cuda.is_available() else False

print(f'在 -- {"GPU" if cuda else "CPU"} -- 上运行')

# 使用初始化特征提取层
def init_weight(m):
    '''
    all the newly added convolutional layers with the ”xavier”
    所有新增的卷积层全部采用 'xavier' 来初始化
    '''
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

# L2_Norm
class L2Norm(nn.Module):
    '''
    The scale of features from different layers may be quite different, making it difficult to directly combine them for prediction.
    '''
    def __init__(self, n_channels, scale):
        super().__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out

class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self):
        super(PriorBox, self).__init__()
        self.image_size = 300
        self.variance = [0.1]
        self.feature_maps = [38,19,10,5,3,1]
        self.min_sizes = [30,60,111,162,213,264]
        self.max_sizes = [60,111,162,213,264,315]
        self.steps = [8,16,32,64,100,300]
        self.aspect_ratios = [[2],[2,3],[2,3],[2,3],[2],[2]]
        self.clip = True

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            # 遍历所有的cell, i相当于行索引, j相当于列索引
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]: # 计算每种aspect_ratios对应的锚框的宽高
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)] # 1:2
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)] # 2:1
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        # 一共生成8732个框 (8732, 4)
        return output

def get_IoU(boxes1, boxes2):
    print(f'get_IoU boxes1.shape = {boxes1.shape} \tboxes2.shape = {boxes2.shape}')
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))

    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    '''
    获取两个矩形相交部分的左上角和右下角坐标.
    给定两个矩形分别为: 
        boxes1 = [[2, 2, 5, 5]] 
        boxes2 = [[3, 2, 6, 4]]
    1. 获取两个矩形的左上角的坐标的最大值: 注意是获取每个维度上的最大值与最小值, 而不是从两个点里选最大的点或者最小的点
        即从 [[2, 2]] [[3, 2]] 中分别获取两个维度上的最大值, 即得到 [3, 2]
    2. 获取两个矩形右下角的坐标的最小值
        即从 [[5, 5]] [[6, 4]] 中分别获取两个维度上的最小值, 即得到 [5, 4]
    3. 两个坐标围成的区域即为两个矩形的交集矩形
        点(3, 2) 和 点(5, 4)所围成的矩形就是交集的面积
    '''
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    # 两个点的差值不能小于0.
    # clamp(input, min, max) 将input的值压缩在 [min, max]的范围内, 若不指定min和max, 则没有上下边界
    # inters中包含两个值[[x, y]], 分别表示相交矩阵的长和宽
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # 交集矩阵的面积
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    # 并集的面积: 矩形A + 矩形B - 交集矩形面积
    union_areas = areas1[:, None] + areas2 - inter_areas
    IoU = inter_areas / union_areas
    return IoU

def match_anchor_to_bbox(ground_truth, anchors, device=None, iou_threshold=0.5):
    """
    将最接近的真实边界框分配给锚框
    ground_truth: [1: 4]
    anchors: [8732, 4]
    """
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # 位于第i行和第j列的元素x_ij是锚框i和真实边界框j的IoU
    jaccard = get_IoU(anchors, ground_truth)
    # 对于每个锚框，分配的真实边界框的张量
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long, device=device)
    # 根据阈值，决定是否分配真实边界框
    # 返回每一行的最大值及其列索引
    max_ious, indices = torch.max(jaccard, dim=1) 
    # max_ious = tensor([0.05, 0.14, 0.57, 0.21, 0.75])       
    # indices = tensor([0, 0, 1, 1, 1])
    # 获取IOU大于阈值的锚框索引, 即行索引
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    # 获取对应的列索引
    box_j = indices[max_ious >= iou_threshold] 
    # 将对应的真实边界框的索引, 即列索引赋值给对应的锚框
    anchors_bbox_map[anc_i] = box_j
    # 
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes): # 2
        max_idx = torch.argmax(jaccard)
        box_idx = (max_idx % num_gt_boxes).long() # 最大IOU对应的列索引
        anc_idx = (max_idx / num_gt_boxes).long() # 最大IOU对应的行索引
        anchors_bbox_map[anc_idx] = box_idx # 将最大的IOU对应的GT索引赋值给对应的锚框
        jaccard[:, box_idx] = col_discard # 将该IOU所在的列去除掉
        jaccard[anc_idx, :] = row_discard # 将该IOU所在的行去除掉
    return anchors_bbox_map

def box_corner_to_center(boxes):
    """
    从（左上，右下）转换到（中间，宽度，高度）
    为计算偏移量offset做准备
    """
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), axis=-1)
    return boxes

def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """对锚框偏移量的转换"""
    c_anc = box_corner_to_center(anchors)
    c_assigned_bb = box_corner_to_center(assigned_bb)
    # 偏移x  10*(xb - xa) / wa
    # 偏移y  5 * log(wb / wa)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = torch.cat([offset_xy, offset_wh], axis=1)
    return offset

def multibox_target(anchors, labels):
    """
    使用真实边界框标记锚框
    anchors: [8732, 4]
    labels: [batch_size, 1, 5]
    """
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :].cuda() if cuda else labels[i, :, :] # [1, 5]
        anchors_bbox_map = match_anchor_to_bbox(label[:, 1:], anchors, device)
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(1, 4)
        # 将类标签和分配的边界框坐标初始化为零
        class_labels = torch.zeros(num_anchors, dtype=torch.long,
                                   device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32,
                                  device=device)
        # 使用真实边界框来标记锚框的类别。
        # 如果一个锚框没有被分配，标记其为背景（值为零）
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true] # 获取到和锚框匹配的GT索引
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # 偏移量转换
        # 这里乘以bbox_mask是因为我们并不需要关注背景是怎么偏移的, 而将所有的关注点集中于预测为某个类别的锚框
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)

'''
input_size: 300 * 300 * 3
'''
class SSD_300(nn.Module):
    def __init__(self, struc:list, feature_struc:list):
        super().__init__()
        self.net = nn.Sequential()
        self.relu = nn.ReLU()
        self.priorbox = PriorBox()
        self.defaultbox = torch.autograd.Variable(self.priorbox.forward(), volatile=True)
        # 定义VGG块
        def vgg_block(n, in_channels, out_channels, last, cm=False):
            layers = []
            for i in range(n):
                layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1))
                layers.append(self.relu)
                in_channels = out_channels
            if not last:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=cm))
            return nn.Sequential(*layers)
        # 加载VGG-16特征层, block1, 2, 3, 4, 5
        for item in struc:
            self.net.append(vgg_block(*item))
        
        # 修改pool 5, FC6 -> CONV6, FC7 -> CONV7
        def modify_add_layers():
            layers = []
            # 修改block 5的池化层, kernel_size从2变成3, stride=1, padding=1
            # (19, 19, 512) -> (19, 19, 512)
            pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            layers.append(pool5)
            # 添加 conv6 block: 512 -> 1024, k=3, p=6, dilation=6
            # (19, 19, 512) -> (19, 19, 1024)
            conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
            layers.append(conv6)
            layers.append(self.relu)
            # 添加 conv7 block: 1024 -> 1024, k=1
            # (19, 19, 1024) -> (19 - 1)
            conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
            layers.append(conv7)
            layers.append(self.relu)
            return nn.Sequential(*layers)

        self.net.append(modify_add_layers())

        # 添加特征提取层
        def add_extras(cfg, i, size=300):
            # Extra layers added to VGG for feature scaling
            layers = []
            in_channels = i
            flag = False
            for k, v in enumerate(cfg):
                if in_channels != 'S':
                    if v == 'S':
                        layers += [nn.Conv2d(in_channels, cfg[k + 1], kernel_size=(1, 3)[flag], stride=2, padding=1)]
                    else:
                        layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
                    flag = not flag
                in_channels = v
            return nn.Sequential(*layers).apply(init_weight)

        self.net.append(add_extras(feature_struc, 1024))
        # L2归一化 conv4_3
        self.l2_norm = L2Norm(512, 20)

        # classification layers
        self.cls_layers = nn.Sequential(
            # 输入通道为特征层的输出通道, 输出通道为 (每个cell锚框数 * 类别数), kernel_size=3, padding=1
            nn.Conv2d(512, 4 * (opt.num_class + 1), kernel_size=3, padding=1),
            nn.Conv2d(1024, 6 * (opt.num_class + 1), kernel_size=3, padding=1),
            nn.Conv2d(512, 6 * (opt.num_class + 1), kernel_size=3, padding=1),
            nn.Conv2d(256, 6 * (opt.num_class + 1), kernel_size=3, padding=1),
            nn.Conv2d(256, 4 * (opt.num_class + 1), kernel_size=3, padding=1),
            nn.Conv2d(256, 4 * (opt.num_class + 1), kernel_size=3, padding=1)
        )

        # regression layers
        self.loc_layers = nn.Sequential(
            # 输入通道为特征层的输出通道, 输出通道为 (每个cell锚框数 * 4), kernel_size=3, padding=1. 4表示4个坐标值
            nn.Conv2d(512, 4 * 4, kernel_size=3, padding=1),
            nn.Conv2d(1024, 6 * 4, kernel_size=3, padding=1),
            nn.Conv2d(512, 6 * 4, kernel_size=3, padding=1),
            nn.Conv2d(256, 6 * 4, kernel_size=3, padding=1),
            nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1),
            nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1)
        )

    def forward(self, x):
        features = []
        loc = []
        cls = []
        for block in range(7):
            # 提取conv4_3和它前面的所有层, conv4_3包含在block3, 第三个conv
            if block == 3:
                sub_net = self.net[block]
                for index, layer in enumerate(sub_net):
                    x = layer(x)
                    if index == 4:
                        # 进行L2归一化
                        s = self.l2_norm(x)
                        # 添加到特征集合中
                        features.append(s)
            elif block == 5:
                # 提取FC7
                sub_net = self.net[block]
                for index, layer in enumerate(sub_net):
                    x = layer(x)
                    if index == 3: # FC7
                        features.append(x)
            elif block == 6:
                # 提取 conv8_2, conv9_2, conv10_2, conv11_2
                sub_net = self.net[block]
                for index, layer in enumerate(sub_net):
                    x = F.relu(layer(x))
                    if index in [1, 3, 5, 7]:
                        features.append(x)
            else:
                x = self.net[block](x)

        # 对特征进行分类和回归
        '''
        多特征分类或者回归时, 由于每个特征图的尺度不一样, 有大有小, 而且每个特征图指定的锚框个数也不一样, (B, C, W, H) 
        所以除了批量维度以外, 剩下的三个维度都不一样.
        所以需要把他们统一到一致的格式, 方便后续的计算等操作
        首先先将通道维度放在最后, 然后将维度转换为⼆维的 (批量⼤⼩，⾼×宽×通道数) 最后在维度1上进行连接
        '''
        for (x, l, c) in zip(features, self.loc_layers, self.cls_layers):
            '''
            torch.Size([1, 16, 38, 38]) torch.Size([1, 24, 19, 19]) torch.Size([1, 24, 10, 10]) 
            torch.Size([1, 24, 5, 5])   torch.Size([1, 16, 3, 3])   torch.Size([1, 16, 1, 1])
            '''
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            cls.append(c(x).permute(0, 2, 3, 1).contiguous())
        
        # 将维度转换为二维的(B, W*H*C)
        loc = torch.cat([lr.view(lr.shape[0], -1) for lr in loc], dim=1) # loc = torch.Size([1, 34928]) 
        cls = torch.cat([cr.view(cr.shape[0], -1) for cr in cls], dim=1) # cls = torch.Size([1, 17464])
        # print(f'loc = {loc.view(loc.shape[0], -1, 4).shape} \ncls = {cls.view(cls.shape[0], -1, opt.num_class + 1).shape}')

        loc = loc.view(loc.shape[0], -1)
        cls = cls.view(cls.shape[0], -1, opt.num_class + 1)

        return tuple((self.defaultbox.cuda() if cuda else self.defaultbox, 
                      loc.cuda() if cuda else loc, 
                      cls.cuda() if cuda else cls))

def show_imgs(X, y, row, col):
    '''
    显示一个批量的图像
    X的shape为 [B, C, W, H]
    '''
    X = X.permute(0, 2, 3, 1) / 255
    axes = d2l.show_images(X, row, col, scale=2)
    for ax, label in zip(axes, y):
        d2l.show_bboxes(ax, [label[0][1:5] * 256], colors=['w'])
    d2l.plt.show()

def get_loss(pred_class, true_class, pred_loc, true_loc, bbox_masks, cross_entroy, L1_loss):
    '''
    计算损失函数
    损失函数分为两个部分 class_loss, 类别损失和偏移量损失 offset_loss
    bbox_masks旨在确保背景(负样例)不参与损失的计算
    pred_class: [16, 8732, 2]
    true_class: [16, 8732]
    pred_loc: [16, 34928]
    true_loc: [16, 34928]]
    '''
    batch_size, num_classes = pred_class.shape[0], pred_class.shape[2]
    # 计算class_loss
    class_loss = cross_entroy(pred_class.reshape(-1, num_classes), true_class.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    offset_loss = L1_loss(pred_loc * bbox_masks, true_loc * bbox_masks).mean(dim=1)
    return class_loss + offset_loss

def transforme_features(features):
    trans = torchvision.transforms.Compose([
        torchvision.transforms.Resize((opt.img_size, opt.img_size))
    ])
    return trans(features)

def train(net, optimiser, cross_entropy, L1_loss):

    best_model = 200
    start_epoch = -1
    if opt.resume:
        checkpoint_dict = torch.load(opt.save_checkpoint_path, 'best_model.pth')
        net.load_state_dict(checkpoint_dict['net'])
        optimiser.load_state_dict(checkpoint_dict['optimiser'])
        start_epoch = checkpoint_dict['epoch']
        best_model = checkpoint_dict['best_model']

    net = net.cuda() if cuda else net
    num_epochs = opt.epochs
    timer = d2l.Timer()
    for epoch in range(start_epoch+1, num_epochs):
        net.train()
        for features, target in train_iter:
            timer.start()
            optimiser.zero_grad()
            X, Y = transforme_features(features).cuda(), target.cuda()
            # 生成多尺度的锚框，为每个锚框预测类别和偏移量
            anchors, bbox_preds, cls_preds = net(X)
            # 为每个锚框标注类别和偏移量
            bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, Y)
            # 根据类别和偏移量的预测和标注值计算损失函数
            l = get_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks, cross_entropy, L1_loss)
            l.mean().backward()
            optimiser.step()
        print(f'epoch = {epoch + 1} \ttrain_loss = {float(l.sum()):.6f}')

        if l.sum() <= best_model:
            print(f'保存模型')
            checkpoint = {
                'net': net.state_dict(),
                'optimiser': optimiser.state_dict(),
                'epoch': epoch,
                'best_model': l.sum()
            }
            torch.save(checkpoint, f'{opt.save_checkpoint_path}/best_model.pth')
            best_model = l.sum()

    print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on {"gpu"}')


'''
conv4 3:  default box with scale 0.1 on conv4 3
conv7 (fc7), 
conv8 2, 
conv9 2, 
conv10 2,
conv11 2
'''

if __name__ == '__main__':
    '''
    [n, in_channels, out_channels, is_last, ceil_mode]
    n: 卷积层的个数
    (in_size + 2 * padding - dilation * (k - 1) -1 / stride + 1
    302 - 2 - 1 + 1
    '''
    struc = [   
        # block 1_1, (300, 300, 64) 
        # block 1_2, (300, 300, 64) 
        # MaxPool, (150, 150, 64)
        [2, 3, 64, False, False], 
        # block 2_1, (150, 150, 128)
        # block 2_2, (150, 150, 128)
        # MaxPool: (75, 75, 128)
        [2, 64, 128, False, False], 
        # block 3_1, (75, 75, 256)
        # block 3_2, (75, 75, 256)
        # block 3_3, (75, 75, 256)
        # MaxPool: (38, 38, 256)
        [3, 128, 256, False, True], 
        # block 4_1, (38, 38, 512)
        # block 4_2, (38, 38, 512)
        # block 4_3, (38, 38, 512) ==> feature map
        # MaxPool: (19, 19, 512)
        [3, 256, 512, False, False],
        # block 5_1, (19, 19, 512) 
        # block 5_2, (19, 19, 512)
        # block 5_3, (19, 19, 512)
        [3, 512, 512, True]
    ]
    '''
    特征提取层, 和原论文中保持一致
    其中 S 表示这个卷积层的stride为2, 否则为1
    列表中是一些输出通道
    特征图不断减小, 是为了检测不同Scales的目标, Multi-Scale feature maps
    '''
    feature_struc = [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256]

    # 构造模型
    ssd = SSD_300(struc, feature_struc)

    # 加载训练数据集
    train_iter, _ = d2l.load_data_bananas(opt.batch_size)
    # 显示一个批量的数据样本
    show_imgs(*next(iter(train_iter)), 4, 4)
    # 训练
    optimiser = torch.optim.SGD(ssd.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    cross_entory = nn.CrossEntropyLoss(reduction='none')
    l1_loss = nn.L1Loss(reduction='none')
    # 创建目录
    print(f'创建目录: {opt.save_checkpoint_path}')
    os.makedirs(opt.save_checkpoint_path, exist_ok=True)
    train(ssd, optimiser, cross_entory, l1_loss)


    # # 读取一张照片
    # img = Image.open('imgs/catdog.png')
    # trans = torchvision.transforms.Compose([
    #     torchvision.transforms.Resize((300, 300)),
    #     torchvision.transforms.ToTensor()
    # ])

    # img_tensor = trans(img).unsqueeze(0)
    # ssd = SSD_300(struc, feature_struc)
    # out = ssd(img_tensor)
    # print(f'out[0].shape = {out[0].shape}')
    # print(len(out[0])) # 8732 一共得到8732个先验框
    # print(out[1].shape) # torch.Size([1, 8732, 4])
    # print(out[2].shape) # torch.Size([1, 8732, 2])
