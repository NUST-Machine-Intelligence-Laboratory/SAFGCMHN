import os
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import CubDataset, CubTextDataset
from model_448_sum import san
from centerloss import CenterLoss
from train import *
from validate import *
from log_save import *
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
def arg_parse():
    parser = argparse.ArgumentParser(description='PyTorch HSE Deployment')
    parser.add_argument('--gpu', default=1, type=int, help='GPU nums to use')
    parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')

    parser.add_argument('--snapshot', default='./trained_img/san10_patchwise.pth', type=str, metavar='PATH',
                        help='path to latest checkpoint')

    parser.add_argument('--batch_size', default=3, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--data_path', default='/home/FGCrossNet', type=str, help='path to dataset')
    parser.add_argument('--model_path', default='./model/san10_patch_decay_consine_dot+sum+_classification/', type=str, help='path to model')
    parser.add_argument('--crop_size', default=448, type=int, help='crop size')
    parser.add_argument('--scale_size', default=512, type=int, help='the size of the rescale image')
    parser.add_argument('--print_freq', default=500, type=int, metavar='N', help='print frequency')
    parser.add_argument('--eval_epoch', default=1, type=int, help='every eval_epoch we will evaluate')
    parser.add_argument('--eval_epoch_thershold', default=2, type=int, help='eval_epoch_thershold')
    # parser.add_argument('--loss_choose', default='c', type=str, help='choose loss(c:centerloss, r:rankingloss)')

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
    sys.stdout = Logger(sys.stdout)  # 将输出记录到log
    args = arg_parse()
    print_args(args)
    print("model_san10_weightdecay=0.0001_patch")
    print("==> Creating dataloader...")

    data_dir = args.data_path
    train_list = './list/image/train.txt'
    train_set = get_train_set(data_dir, train_list, args)
    train_list1 = './list/video/train.txt'
    train_set1 = get_train_set(data_dir, train_list1, args)
    train_list2 = './list/audio/train.txt'
    train_set2 = get_train_set(data_dir, train_list2, args)
    train_list3 = './list/text/train.txt'
    train_set3 = get_text_set(data_dir, train_list3, args, 'train')

    test_list = './list/image/test.txt'
    test_set = get_test_set(data_dir, test_list, args)
    test_list1 = './list/video/test.txt'
    test_set1 = get_test_set(data_dir, test_list1, args)
    test_list2 = './list/audio/test.txt'
    test_set2 = get_test_set(data_dir, test_list2, args)
    test_list3 = './list/text/test.txt'
    test_set3 = get_text_set(data_dir, test_list3, args, 'test')

    # test_loader = DataLoader(dataset=test_set, num_workers=args.workers, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    # test_loader1 = DataLoader(dataset=test_set1, num_workers=args.workers, batch_size=args.batch_size, shuffle=False)
    # test_loader2 = DataLoader(dataset=test_set2, num_workers=args.workers, batch_size=args.batch_size, shuffle=False)
    test_loader3 = DataLoader(dataset=test_set3, num_workers=args.workers, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    print("==> Loading the network ...")
    # model = san(sa_type=1, layers=[3, 3, 4, 6, 3], kernels=[3, 7, 7, 7, 7], num_classes=200)
    model = san(sa_type=1, layers=[2, 1, 2, 4, 1], kernels=[3, 7, 7, 7, 7], num_classes=200)
    
    # center_loss = CenterLoss(200, 200, True)  # 损失函数改进,第一个200是类别，第二个200是特征维度

    if args.gpu is not None:
        model = model.cuda()
        cudnn.benchmark = True

    if os.path.isfile(args.snapshot):
        print("==> loading checkpoint '{}'".format(args.snapshot))
        # pretrained
               
        checkpoint = torch.load(args.snapshot)
        # print("checkpoint",checkpoint)
        model_dict = model.state_dict()
        restore_param = {}
        for k, v in checkpoint.items():
            if k in {"state_dict"}:
                # print(k)
                restore_param = {x[7:]: y for x, y in v.items() if x[7:] in model_dict}
        for x, y in restore_param.items():
            print(x)
        model_dict.update(restore_param)
        # print("model_dict",model_dict)
        model.load_state_dict(model_dict)
        '''
        
        # run r
        checkpoint = torch.load(args.snapshot)
        model_dict = model.state_dict()
        restore_param = {k: v for k, v in checkpoint.items() if k in model_dict}
        model_dict.update(restore_param)
        # print("model_dict",model_dict)
        for x,y in model_dict.items():
            print(x)
        model.load_state_dict(model_dict)
        '''   
        # torch.save(model.state_dict(), "model.pkl", _use_new_zipfile_serialization=False)
        print("==> loaded checkpoint '{}'".format(args.snapshot))
    else:
        print("==> no checkpoint found at '{}'".format(args.snapshot))
        # exit()
    criterion = nn.CrossEntropyLoss()  # 损失函数

    params = list(model.parameters())
    # params = list(model.parameters()) + list(center_loss.parameters())
    # optimizer = optim.SGD(filter(lambda p: p.requires_grad,params),
    #                       lr=0.001, momentum=0.9, weight_decay=0.0001)  # 优化器
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, params),
                          lr=0.001, momentum=0.9, weight_decay=0.0001)
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    # for name, param in center_loss.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    # 学习率测试 学习率过大，loss为nan
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)  # 学习率动态下降
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    savepath = args.model_path
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    for epoch in range(args.epochs):
        scheduler.step()
        print("lr", optimizer.param_groups[0]['lr'])
        train_loader = DataLoader(dataset=train_set, num_workers=args.workers, batch_size=args.batch_size,
                                  shuffle=True, pin_memory=True)
        train_loader1 = DataLoader(dataset=train_set1, num_workers=args.workers, batch_size=args.batch_size,
                                   shuffle=True, pin_memory=True)
        train_loader2 = DataLoader(dataset=train_set2, num_workers=args.workers, batch_size=args.batch_size,
                                   shuffle=True, pin_memory=True)
        train_loader3 = DataLoader(dataset=train_set3, num_workers=args.workers, batch_size=args.batch_size,
                                   shuffle=True, pin_memory=True)

        # train(train_loader, train_loader1, train_loader2, train_loader3, args, model, criterion, center_loss, optimizer,
        #       epoch, args.epochs)
        train(train_loader, train_loader1, train_loader2, train_loader3, args, model, criterion, optimizer,
              epoch, args.epochs)
        # train_img(train_loader,args, model, criterion, center_loss, optimizer, epoch, args.epochs)
        print('-' * 20)
        # print("Image Acc:")
        # image_acc = validate(test_loader, model, args, False)
        # print("Video Acc:")
        # video_acc = validate(test_loader1, model, args, False)
        # print("Audio Acc:")
        # audio_acc = validate(test_loader2, model, args, False)
        print("Text Acc:")
        text_acc = validate(test_loader3, model, args, True)

        save_model_path = savepath + 'epoch_' + str(epoch) + '_' + str(text_acc) + '.pkl'
        torch.save(model.state_dict(), save_model_path)


def get_train_set(data_dir, train_list, args):
    # print("data_dir", data_dir)
    # print("train_list", train_list)
    # 归一化
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    crop_size = args.crop_size
    scale_size = args.scale_size
    train_data_transform = transforms.Compose([
        transforms.Resize((scale_size, scale_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize,
    ])

    train_set = CubDataset(data_dir, train_list, train_data_transform)
    # train_loader = DataLoader(dataset=train_set, num_workers=args.workers, batch_size=args.batch_size, shuffle=False)

    return train_set


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
    # test_loader = DataLoader(dataset=test_set, num_workers=args.workers, batch_size=args.batch_size, shuffle=False)
    return test_set


def get_text_set(data_dir, test_list, args, split):
    data_set = CubTextDataset(data_dir, test_list, split)
    # data_loader = DataLoader(dataset=data_set, num_workers=args.workers, batch_size=args.batch_size, shuffle=False)
    return data_set


if __name__ == "__main__":
    main()
