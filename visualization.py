import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
from torchvision import models, transforms
import torch
# import timm

import os
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
# import skimage.data
# import skimage.io
# import skimage.transform
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
from PIL import Image
import cv2
from model_448_sum import san
class SaveConvFeatures():

    def __init__(self, m):  # module to hook
        self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.data

    def remove(self):
        self.hook.remove()


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
t = transforms.Compose([transforms.Resize((512, 512)),  # 128, 128
                        transforms.CenterCrop(448),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])


img_file = r"img/5.jpg"
img = Image.open(img_file)
# Image.fromarray(cv.cvtColor(Img,cv2.COLOR_BGR2RGB))
print(img.size)
img = t(img).unsqueeze(0).to(device)

# custom_model = models.resnet50(pretrained=True).cuda()

custom_model = san(sa_type=1, layers=[2, 1, 2, 4, 1], kernels=[3, 7, 7, 7, 7], num_classes=200).cuda()
checkpoint = torch.load('model/ours.pkl')
custom_model.load_state_dict(checkpoint)
custom_model.eval()
# custom_model = timm.create_model('resnest50d', pretrained=True)
# custom_model.layer4是自己需要查看特征输出的卷积层
# for name, param in custom_model.named_parameters():
#     print(name)
# exit()
hook_ref = SaveConvFeatures(custom_model.layer4)
with torch.no_grad():
    custom_model.forward_share(img)

conv_features = hook_ref.features  # [1,2048,7,7]

print('特征图输出维度：', conv_features)  # 其实得到特征图之后可以自己编写绘图程序
hook_ref.remove()


therd_size = 256
def make_dirs(path):
    if os.path.exists(path) is False:
        os.makedirs(path)
def shows(img_src, conv_features):
    features = conv_features[0]
    iter_range = features.shape[0]
    for i in range(iter_range):
        # plt.imshow(x[0].data.numpy()[0,i,:,:],cmap='jet')
        feature = features.data.cpu().numpy()
        feature_img = feature[i, :, :]
        feature_img = np.asarray(feature_img * 255, dtype=np.uint8)

        dst_path = os.path.join(img_src, "layer4")

        make_dirs(dst_path)
        feature_img = cv2.applyColorMap(feature_img, cv2.COLORMAP_JET)
        print("feature_img",feature_img.shape)

        if feature_img.shape[0] < therd_size:
            img = Image.open(img_file).convert('RGB')
            tmp_file = os.path.join(dst_path, str(i) + '_' + str(therd_size) + '.png')
            superimg_file = os.path.join(dst_path, str(i) + '_' + str(therd_size)+"_superimg" + '.png')
            tmp_img = feature_img.copy()
            # print("tmp_img",tmp_img.shape)
            # print("img.size[0]",img.size)
            tmp_img = cv2.resize(tmp_img, (700, 700), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(tmp_file, tmp_img)
            superimg = tmp_img * 0.6 + np.array(img)[:, :, ::-1]

            cv2.imwrite(superimg_file, superimg)
            # plt.imshow(superimg)

        dst_file = os.path.join(dst_path, str(i) + '.png')
        cv2.imwrite(dst_file, feature_img)




def show_feature_map(img_src, conv_features):
    '''可视化卷积层特征图输出
    img_src:源图像文件路径
    conv_feature:得到的卷积输出,[b, c, h, w]
    '''
    img = Image.open(img_file).convert('RGB')
    height, width = img.size
    heat = conv_features.squeeze(0)  # 降维操作,尺寸变为(2048,7,7)
    heat_mean = torch.mean(heat, dim=0)  # 对各卷积层(2048)求平均值,尺寸变为(7,7)
    heatmap = heat_mean.cpu().numpy()  # 转换为numpy数组
    heatmap /= np.max(heatmap)  # minmax归一化处理
    heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))  # 变换heatmap图像尺寸,使之与原图匹配,方便后续可视化
    heatmap = np.uint8(255 * heatmap)  # 像素值缩放至(0,255)之间,uint8类型,这也是前面需要做归一化的原因,否则像素值会溢出255(也就是8位颜色通道)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 颜色变换
    plt.imshow(heatmap)
    plt.show()
    # heatmap = np.array(Image.fromarray(heatmap).convert('L'))
    superimg = heatmap * 0.4 + np.array(img)[:, :, ::-1]  # 图像叠加，注意翻转通道，cv用的是bgr
    cv2.imwrite('img/superimg.jpg', superimg)  # 保存结果
    # 可视化叠加至源图像的结果
    img_ = np.array(Image.open('img/superimg.jpg').convert('RGB'))
    plt.imshow(img_)
    plt.show()

if __name__ == '__main__':
    # show_feature_map(img_file, conv_features)
    shows("./v_san_img5", conv_features)