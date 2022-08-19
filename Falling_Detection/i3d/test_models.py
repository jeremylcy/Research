import argparse
import time
import os
from turtle import position
import cv2
from PIL import Image
from sklearn.metrics import confusion_matrix
import sys

from dataset import I3DDataSet
from models import i3d
from transforms import *

import torch
import torchvision.transforms as transform

if __name__ == '__main__':
    # options
    #print(torch.cuda.device_count())
    parser = argparse.ArgumentParser(description="Standard video-level testing")
    parser.add_argument('dataset', type=str, choices=['ucf101', 'hmdb51', 'kinetics'])
    parser.add_argument('modality', type=str, choices=['RGB', 'Flow'])
    parser.add_argument('root_path', type=str)
    parser.add_argument('test_list', type=str)
    parser.add_argument('weights', type=str)
    parser.add_argument('--arch', type=str, default='i3d_resnet50')
    parser.add_argument('--save_scores', type=str, default=None)
    parser.add_argument('--max_num', type=int, default=-1)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--clip_length', default=250, type=int, metavar='N',
                        help='length of sequential frames (default: 64)')
    parser.add_argument('--dropout', type=float, default=0.7)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--gpus', nargs='+', type=int, default=None)
    parser.add_argument('--flow_prefix', type=str, default='flow_')

    args = parser.parse_args()

    if args.dataset == 'ucf101':
        num_classes = 101
    elif args.dataset == 'hmdb51':
        num_classes = 51
    elif args.dataset == 'kinetics':
        num_classes = 400
    else:
        raise ValueError('Unknown dataset ' + args.dataset)

    model = getattr(i3d, args.arch)(modality=args.modality, num_classes=num_classes,
                                    dropout_ratio=args.dropout)

    checkpoint = torch.load(args.weights)
    print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))

    base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
    model.load_state_dict(base_dict)

    # Data loading code
    crop_size = args.input_size
    scale_size = args.input_size * 256 // 224
    input_mean = [0.485, 0.456, 0.406]
    input_std = [0.229, 0.224, 0.225]
    if args.modality == 'Flow':
        input_mean = [0.5]
        input_std = [np.mean(input_std)]

    data_loader = torch.utils.data.DataLoader(
        I3DDataSet(args.root_path, args.test_list, clip_length=args.clip_length, modality=args.modality,
                image_tmpl="{:00d}.jpg" if args.modality == "RGB" else args.flow_prefix + "{}_{:05d}.jpg",
                transform=torchvision.transforms.Compose([
                    GroupScale(scale_size),
                    GroupCenterCrop(crop_size),
                    ToNumpyNDArray(),
                    ToTorchFormatTensor(),
                    GroupNormalize(input_mean, input_std),
                ]),
                test_mode=True),
        batch_size=1, shuffle=False,
        num_workers=args.workers * 2, pin_memory=True)
    
    if args.gpus is not None:
        devices = [args.gpus[i] for i in range(args.workers)]
    else:
        devices = list(range(args.workers))

    model = torch.nn.DataParallel(model.cuda())
    model.eval()

    data_gen = enumerate(data_loader)
    print (data_gen)
    total_num = len(data_loader.dataset)
    output, out1 = [], []

    def eval_video(i, data, label):
        with torch.no_grad():
        #i, data, label = video_data
        #print(i)
        #print(data)
        #print(label[0])
        #x = torch.tensor(data, device='cuda')
        #x = data.clone().detach()
        #print(x.cpu().numpy())
            #print(data)
            rst = model(data).data.cpu().numpy().copy()
            return i, rst, label[0]

    proc_start_time = time.time()
    max_num = args.max_num if args.max_num > 0 else len(data_loader.dataset)
    #negative, positive = 0, 0
    failed, passed = 0, 0
    
    for i, (data, label) in data_gen:
        if i >= max_num:
            break
        #torch.cuda.empty_cache()
        out1.append(label[0].item())
        rst = eval_video(i, data, label)
        #y = torch.tensor(label[0], device='cuda').item()
        #if y == 1:
            #negative += 1 
        output.append(rst[1:])
        #elif y == 2:
            #positive += 1 
        #output2.append(rst[1:])
        #torch.cuda.empty_cache()
        cnt_time = time.time() - proc_start_time
        if i == 0 and i == total_num:
            print('{} videos done, total {}/{}, average {} sec/video'.format(i, i + 1,total_num,float(cnt_time) / (i + 1)))
        elif i % 10 == 0:
            print('{} videos done, total {}/{}, average {} sec/video'.format(i, i + 1,total_num,float(cnt_time) / (i + 1)))
    #print("negative: " + str(negative))
    #print("positive: " + str(positive))
    video_pred = [np.argmax(x[0]) for x in output]
    jj, tt, ff = 0, 0, 0
    for x in output:
        video = np.argmax(x[0])
        score = x[0]
        #print(((score[0][1] + score[0][2]) / 2))
        if score[0][1] > score[0][2]:
            scores = (((score[0][1] -  score[0][2]) / score[0][1]) * 100)
        elif score[0][2] > score[0][1]:
            scores = (((score[0][2] - score[0][1])/ score[0][2]) * 100)
        if out1[jj] == video:
            re = "True"
            tt += 1
        else:
            re = "False"
            ff += 1
        #print(str(out1[jj]) + " % " + str(video))
        jj += 1
        #print(video_pred)

    print("failed: " + str(ff))
    print("success: " + str(tt))
    print(str((tt/(tt+ff))*100) + "%")

    video_labels = [x[1] for x in output]
    cf = confusion_matrix(video_labels, video_pred, normalize='all').astype(float)
    
    cls_cnt = cf.sum(axis=1)
    print(cls_cnt)
    cls_hit = np.diag(cf)
    print(cls_hit)

    cls_acc = cls_hit / cls_cnt

    print('Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))

    """
    if args.save_scores is not None:

        # reorder before saving
        name_list = [x.strip().split()[0] for x in open(args.test_list)]

        order_dict = {e: i for i, e in enumerate(sorted(name_list))}

        reorder_output = [None] * len(output)
        reorder_label = [None] * len(output)

        for i in range(len(output)):
            idx = order_dict[name_list[i]]
            reorder_output[idx] = output[i]
            reorder_label[idx] = video_labels[i]

        np.savez(args.save_scores, scores=output, labels=video_pred)
    
    data = np.load("D:/i3d-pytorch-master/data/ucfTrainTestlist/1.npz", allow_pickle=True)
    print(data.files)
    row = data.files
    np.set_printoptions(threshold=np.inf)
    print(data.files[0])
    sys.stdout=open("D:/i3d-pytorch-master/data/ucfTrainTestlist/1.csv","w")
    for i in row:
        print("--------------------------")
        print(data[i])
    sys.stdout.close()

    t = []
    vid_path = input("Vid Path ?")
    vid = cv2.VideoCapture(vid_path)
    ret, frame = vid.read()
    while ret:
        if not ret:
            break
        frame = torch.from_numpy(frame)
        rst = eval_video(frame)
        t.append[rst]
        ret, frame = vid.read()

    vid_pred = np.argmax(t[0])
    print(vid_pred)
    """