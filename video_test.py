import argparse
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import CubDataset, CubTextDataset
from model_448 import san
from retrieval import *
from validate import validate

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
post =[3]

def arg_parse():
    # CUDA_VISIBLE_DEVICES=0,1 python test.py --data_path='/path/dataset/' --snapshot='./model/rankingloss/model.pkl' --feature='./feature'
    parser = argparse.ArgumentParser(description='PyTorch HSE Deployment')
    parser.add_argument('--gpu', default=0, type=int, help='GPU nums to use')
    parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('--batch_size', default=1, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--data_path', default='/home/FGCrossNet', type=str, help='path to dataset')
    parser.add_argument('--snapshot', default='./model/san10_pair_r/epoch_63_0.3405.pkl', type=str, help='path to latest checkpoint')
    parser.add_argument('--feature', default='./feature', type=str, help='path to feature')
    parser.add_argument('--crop_size', default=448, type=int, help='crop size')
    parser.add_argument('--scale_size', default=512, type=int, help='the size of the rescale image')

    args = parser.parse_args()
    return args


def print_args(args):
    print("==========================================")
    print("==========       CONFIG      =============")
    print("==========================================")
    for arg, content in args.__dict__.items():
        print("{}:{}".format(arg, content))
    print("\n")


def main():
    args = arg_parse()
    print_args(args)

    print("==> Creating dataloader...")

    data_dir = args.data_path

    test_list2 = 'out1.txt'
    test_loader2 = get_test_set(data_dir, test_list2, args)



    out_feature_dir2 = os.path.join(args.feature, 'video')



    mkdir(out_feature_dir2)


    print("==> Loading the modelwork ...")
    # model = san(sa_type=1, layers=[3, 3, 4, 6, 3], kernels=[3, 7, 7, 7, 7], num_classes=200)
    model = san(sa_type=0, layers=[2, 1, 2, 4, 1], kernels=[3, 7, 7, 7, 7], num_classes=200)
    model = model.cuda()

    '''
    if args.gpu is not None:
        model = nn.DataParallel(model, device_ids=range(args.gpu))
        model = model.cuda()
        cudnn.benchmark = True
    '''

    if args.snapshot:
        if os.path.isfile(args.snapshot):
            print("==> loading checkpoint '{}'".format(args.snapshot))
            checkpoint = torch.load(args.snapshot)
            model.load_state_dict(checkpoint)
            print("==> loaded checkpoint '{}'".format(args.snapshot))
        else:
            print("==> no checkpoint found at '{}'".format(args.snapshot))
            exit()

    model.eval()
    # model = model.module




    print("Video Features ...")
    vid = extra(model, test_loader2, out_feature_dir2, args, flag='v')

    # compute_mAP(img, vid, aud, txt)


def mkdir(out_feature_dir):
    if not os.path.exists(out_feature_dir):
        os.makedirs(out_feature_dir)


def extra(model, test_loader, out_feature_dir, args, flag):
    size = args.batch_size
    num = 0
    # if (flag == 'v'):
    #     size = 1
    #     f = np.zeros((len(test_loader), 200))
    # else:
    p = 0
    m=0
    f = np.zeros((len(test_loader) * size, 200))
    fla =True
    for i, (input, target) in enumerate(test_loader):

        target = target.cuda()
        with torch.no_grad():
            input_var = torch.autograd.Variable(input).cuda()

        out = model.forward_share(input_var)
        # print("output",out.size())

        p += 1
        if p == post[m]:
            output = torch.cat((out, output), 0)
            p = 0
            m += 1
            # print("output",output.size())

            output = torch.mean(output, 0).reshape(1,200)
            # print("output",output)

            output = F.softmax(output, dim=1).detach().cpu().numpy()
            print("output",output)
            num += output.shape[0]
            print(m)
            if ((m-1) == len(post) - 1):
                f[(m-1) * size:num, :] = output
            else:
                f[(m-1) * size:(m) * size, :] = output
            fla = True
        else:
            if fla:
                output = out
                fla = False
            else:
                output = torch.cat((out,output),0)
                # print("output",output.size())

    np.savetxt(out_feature_dir + '/features_te.txt', f[:num, :])
    return out_feature_dir + '/features_te.txt'


def get_test_set(data_dir, test_list, args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    crop_size = args.crop_size
    scale_size = args.scale_size
    test_data_transform = transforms.Compose([
        transforms.Resize((scale_size, scale_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize,
    ])
    test_set = CubDataset(data_dir, test_list, test_data_transform)
    # print("test_set",test_set)
    test_loader = DataLoader(dataset=test_set, num_workers=args.workers, batch_size=args.batch_size, shuffle=False,
                             pin_memory=True)

    return test_loader


def get_text_set(data_dir, test_list, args, split):
    data_set = CubTextDataset(data_dir, test_list, split)
    # print("data_set",data_set)
    data_loader = DataLoader(dataset=data_set, num_workers=args.workers, batch_size=args.batch_size, shuffle=False,
                             pin_memory=True)

    return data_loader


if __name__ == "__main__":
    main()
