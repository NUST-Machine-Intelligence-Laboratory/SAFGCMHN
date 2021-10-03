import time
from torch.autograd import Variable
from util import AverageMeter
from rankingloss import *


def train(train_loader, train_loader1, train_loader2, train_loader3, args, model, criterion,  optimizer,
          epoch, num_epochs):

    print(len(train_loader),len(train_loader1), len(train_loader2), len(train_loader3))
    # 29 432 30 20
    since = time.time()

    # 管理变量Loss的更新
    running_loss0 = AverageMeter()
    running_loss1 = AverageMeter()
    running_loss2 = AverageMeter()
    running_loss3 = AverageMeter()
    running_loss4 = AverageMeter()
    # running_loss5 = AverageMeter()
    # running_loss6 = AverageMeter()
    running_loss = AverageMeter()



    # 设置为训练模式
    model.train()
    count = 0
    image_acc = 0
    text_acc = 0
    video_acc = 0
    audio_acc = 0


    # 迭代读取数据的循环函数
    for (i, (input, target)), (j, (input1, target1)), (k, (input2, target2)), (p, (input3, target3)) in zip(
            enumerate(train_loader), enumerate(train_loader1), enumerate(train_loader2), enumerate(train_loader3)):

        # input torch.Size([batch_size, 3, 448, 448])
        # print("input.size()",input.size())
        # 将读取的数据从Tensor转换成，前向传播需要的Variable格式
        input_var = Variable(input.cuda())
        input_var1 = Variable(input1.cuda())
        input_var2 = Variable(input2.cuda())
        input_var3 = Variable(input3.cuda())


        # target 类别
        targets = torch.cat((target, target1, target2, target3), 0)# 按行拼接
        targets = Variable(targets.cuda())

        target_var = Variable(target.cuda())
        target_var1 = Variable(target1.cuda())
        target_var2 = Variable(target2.cuda())
        target_var3 = Variable(target3.cuda())

        '''
            input_var torch.Size([1, 3, 448, 448])
            input_var1 torch.Size([1, 3, 448, 448])
            input_var2 torch.Size([1, 3, 448, 448])
            input_var3 torch.Size([1, 448])
        '''
        # 前向传播

        outputs = model(input_var, input_var1, input_var2, input_var3)



        feature = outputs
        # outputs torch.Size([4*batch_size, 200])
        # print(outputs.size())
        # print("outputs.size(0)",outputs.size(0))
        size = int(outputs.size(0) / 4)
        # print("size",size)
        img = outputs.narrow(0, 0, target_var.size(0))
        # print("img",img.size())
        vid = outputs.narrow(0, target_var.size(0), target_var1.size(0))
        aud = outputs.narrow(0, target_var.size(0)+target_var1.size(0), target_var2.size(0))
        txt = outputs.narrow(0, target_var.size(0)+target_var1.size(0)+target_var2.size(0), target_var3.size(0))
        if img.size(0) != size or vid.size(0) != size or aud.size(0) != size or txt.size(0) != size:
            print("img", img.size())
            print("vid", vid.size())
            print("aud", aud.size())
            print("txt", txt.size())
            print("img_var",target_var.size(0))
            print("vid_var", target_var1.size(0))
            print("aud_var", target_var2.size(0))
            print("txt_var", target_var3.size(0))

        _, predict1 = torch.max(img, 1)  # 0是按列找，1是按行找
        _, predict2 = torch.max(vid, 1)  # 0是按列找，1是按行找
        _, predict3 = torch.max(txt, 1)  # 0是按列找，1是按行找
        _, predict4 = torch.max(aud, 1)  # 0是按列找，1是按行找
        image_acc += torch.sum(torch.squeeze(predict1.float() == target_var.float())).item() / float(
            target_var.size()[0])
        video_acc += torch.sum(torch.squeeze(predict2.float() == target_var1.float())).item() / float(
            target_var1.size()[0])
        audio_acc += torch.sum(torch.squeeze(predict4.float() == target_var2.float())).item() / float(
            target_var2.size()[0])
        text_acc += torch.sum(torch.squeeze(predict3.float() == target_var3.float())).item() / float(
            target_var3.size()[0])

        # 损失函数计算,输入前向传播结果和实际类别的索引对象
        loss0 = criterion(img, target_var)
        # print("loss0",loss0)
        loss1 = criterion(vid, target_var1)
        loss2 = criterion(aud, target_var2)
        loss3 = criterion(txt, target_var3)
        # 分类约束loss4
        loss4 = loss0 + loss1 + loss2 + loss3

        # 中心约束loss5,alpha为0.001,
        # loss5 = center_loss(feature, targets) * 0.001
        # print("loss5",loss5)
        # 排序约束loss6
        #if (args.loss_choose == 'r'):
        #    loss6, _ = ranking_loss(targets, feature, margin=1, margin2=0.5, squared=False)

        #    loss6 = loss6 * 0.1

        #else:
        #    loss6 = 0.0

        # 损失值loss = 分类约束 + 中心约束 + 排序约束
        #loss = loss4 + loss5 + loss6
        loss = loss4
        # 更新loss
        batchsize = input_var.size(0)
        running_loss0.update(loss0.item(), batchsize)
        running_loss1.update(loss1.item(), batchsize)
        running_loss2.update(loss2.item(), batchsize)
        running_loss3.update(loss3.item(), batchsize)
        running_loss4.update(loss4.item(), batchsize)
        # running_loss5.update(loss5.item(), batchsize)
        # if (args.loss_choose == 'r'):
        #     running_loss6.update(loss6.item(), batchsize)
        running_loss.update(loss.item(), batchsize)

        # 梯度初始化为零
        optimizer.zero_grad()

        # 反向传播，求梯度
        loss.backward()

        #for param in center_loss.parameters():
            # param torch.Size([200, 200])
            # 倍数（1. / alpha），以消除alpha对更新中心的影响
        #    param.grad.data *= (1. / 0.001)

        # 更新所有参数
        optimizer.step()
        count += 1
        if (i % args.print_freq == 0):

            print('-' * 20)
            print('Epoch [{0}/{1}][{2}/{3}]'.format(epoch, num_epochs, i, len(train_loader)))
            print('Image Loss: {loss.avg:.5f}'.format(loss=running_loss0))
            print('Video Loss: {loss.avg:.5f}'.format(loss=running_loss1))
            print('Audio Loss: {loss.avg:.5f}'.format(loss=running_loss2))
            print('Text Loss: {loss.avg:.5f}'.format(loss=running_loss3))
            print('AllMedia Loss: {loss.avg:.5f}'.format(loss=running_loss4))
            #print('Center Loss: {loss.avg:.5f}'.format(loss=running_loss5))
            # print('separate Loss: {loss.avg:.5f}'.format(loss=running_loss7))
            #if (args.loss_choose == 'r'):
            #    print('Ranking Loss: {loss.avg:.5f}'.format(loss=running_loss6))
            print('All Loss: {loss.avg:.5f}'.format(loss=running_loss))
            # log.save_train_info(epoch, i, len(train_loader), running_loss)

    print("训练第%d个epoch:" % epoch)
    print("image:", image_acc / len(train_loader3))
    print("text:", text_acc / len(train_loader3))
    print("video:", video_acc / len(train_loader3))
    print("audio:", audio_acc / len(train_loader3))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60), "训练了%d个batch" % count)
