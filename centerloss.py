import torch
import torch.nn as nn
import scipy.spatial

class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    
    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        '''
                    1.计算distmat = sum(x^2)+sum(c^2)-2xc = 欧式距离的平方
                    2.把label变成labels=[0, 0, ..., label, 0, ..., 0]
                    3.mask = [[0, 0, ..., 1, 0, ..., 0],
                            [0, 0, ..., 1, 0, ..., 0],
                            [0, 0, ..., 1, 0, ..., 0],
                            [0, 0, ..., 1, 0, ..., 0]]
                    4.dist = 将distmat和mask相乘，计算出对应于原中心的距离
                    5.取均值
                '''
        batch_size = x.size(0)
        '''
        batch_size 4
        num_classes 200
        self.feat_dim 200
        distmat torch.Size([4, 200])
        classes torch.Size([200])
        labels torch.Size([4, 200])
        mask torch.Size([4, 200])
        '''
        # print("batch_size",batch_size)
        # print("num_classes",self.num_classes)
        # print("self.feat_dim",self.feat_dim)

        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        # print("distmat",distmat.size())
        # β = 1, α = -2
        # 1 * distmat - 2 *（X @ centers.t()）
        distmat.addmm_(1, -2, x, self.centers.t())
        # x1x2 + y1y2

        # 创建0-199的数组
        classes = torch.arange(self.num_classes).long()

        if self.use_gpu: 
            classes = classes.cuda()

        # la = labels.unsqueeze(1)
        # print("labels",labels.size())
        # print("la",la.size())

        # unsqueeze()在第二维增加一个维度
        # labels = torch.Size([4])   labels.unsqueeze(1) = torch.Size([4, 1])
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)

        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        # print("dist",dist.size())
        # dist torch.Size([4, 200])

        # sum/batch_size 求均值
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        # print("loss",loss)
        return loss