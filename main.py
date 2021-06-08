import os
import random
import time
import argparse
import numpy as np
from tqdm import tqdm
from ptflops import get_model_complexity_info
from adabelief_pytorch import AdaBelief

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR

from data_local_loader import CustomDataset, data_loader
from model import FewShotClassifier
from loss import LabelSmoothingLoss

try:
    import nsml
    from nsml import DATASET_PATH, IS_ON_NSML

except:
    IS_ON_NSML=False
    DATASET_PATH = '../1-3-DATA-fin'

def _infer(model, root_path, test_loader=None):
    if test_loader is None:
        test_loader = data_loader(root=root_path, phase='test', batch_size=1)

    lv0 = [0, 23, 39, 57, 65, 69, 82, 85, 86, 90, 93, 100]; lv1 = [1, 2, 3, 4, 5, 6, 11, 17, 24, 25, 26, 27, 28, 29, 35, 40, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 63, 66, 67, 68, 70, 74, 78, 83, 84, 87, 88, 89, 91, 92, 94, 95, 96, 97, 98, 99, 101, 102, 103, 104, 105, 106]; lv2 = [7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 30, 31, 32, 33, 34, 36, 37, 38, 41, 42, 43, 45, 48, 55, 56, 59, 61, 62, 64, 71, 72, 73, 75, 76, 77, 79, 80, 81]

    model.eval()
    ret_id = []
    ret_cls = []
    for data_id, image in tqdm(test_loader):
        image = image.cuda()
        data_id = data_id[0].item()
        fc0, fc1, fc2 = model(image)

        fc0 = fc0.squeeze().detach().cpu().numpy()
        res_cls0 = lv0[int(np.argmax(fc0))]

        fc1 = fc1.squeeze().detach().cpu().numpy()
        res_cls1 = lv1[int(np.argmax(fc1))]

        fc2 = fc2.squeeze().detach().cpu().numpy()
        res_cls2 = lv2[int(np.argmax(fc2))]

        ret_cls.append([res_cls0, res_cls1, res_cls2])
        ret_id.append(data_id)

    return [ret_id, ret_cls]

def bind_nsml(model, optimizer, scheduler):
    def save(dir_name, *args, **kwargs):
        os.makedirs(dir_name, exist_ok=True)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        torch.save(state, os.path.join(dir_name, 'model.pth'))

    def load(dir_name, *args, **kwargs):
        state = torch.load(os.path.join(dir_name, 'model.pth'))
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        print('loaded')

    def infer(root_path, top_k=1):
        return _infer(model, root_path)

    nsml.bind(save=save, load=load, infer=infer, use_nsml_legacy=False)

grad_history = []
def _get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm 

if __name__ == '__main__':
    # mode argument
    args = argparse.ArgumentParser()
    args.add_argument("--pretrained", type=str, default='tf_efficientnet_b4_ns')
    args.add_argument("--num_classes", type=int, default=107)
    args.add_argument("--batch_size", type=int, default=256)
    args.add_argument("--train_split", type=float, default=0.95)
    args.add_argument("--lr", type=float, default=0.001)
    args.add_argument("--cuda", type=bool, default=True)
    args.add_argument("--num_steps", type=int, default=2000)
    args.add_argument("--print_iter", type=int, default=50)
    args.add_argument("--smoothing", type=float, default=0.1)
    args.add_argument("--autoclip", default=False, action='store_true')

    # reserved for nsml
    args.add_argument("--mode", type=str, default="train")
    args.add_argument("--iteration", type=str, default='0')
    args.add_argument("--pause", type=int, default=0)

    config = args.parse_args()

    # get configurations
    pretrained = config.pretrained
    num_classes = config.num_classes
    base_lr = config.lr
    cuda = config.cuda and torch.cuda.is_available()
    num_steps = config.num_steps
    print_iter = config.print_iter
    mode = config.mode
    train_split = config.train_split
    batch_size = config.batch_size
    smoothing = config.smoothing

    # seed
    seed = 1905
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # init model
    model = FewShotClassifier(pretrained)

    input_resolution = 250
    macs, params = get_model_complexity_info(model, (3, input_resolution, input_resolution), print_per_layer_stat=False, as_strings=False)
    macs /= 10 ** 9
    params /= 10 ** 6
    print('{:<30}  {} GMac'.format('Computational complexity: ', macs))
    print('{:<30}  {} M'.format('Number of parameters: ', params))

    criterion = LabelSmoothingLoss(smoothing)

    if cuda:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model).cuda()
        else:
            model = model.cuda()

    optimizer = AdaBelief(model.parameters(), lr=base_lr, eps=1e-8, weight_decay=1e-2, betas=(0.9,0.999), weight_decouple = True, rectify = False, print_change_log = False)
    scheduler = MultiStepLR(optimizer, [1100], gamma=0.1)

    if IS_ON_NSML:
        bind_nsml(model, optimizer, scheduler)

        if config.pause:
            nsml.paused(scope=locals())
    
    lv0 = [0, 23, 39, 57, 65, 69, 82, 85, 86, 90, 93, 100]; lv1 = [1, 2, 3, 4, 5, 6, 11, 17, 24, 25, 26, 27, 28, 29, 35, 40, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 63, 66, 67, 68, 70, 74, 78, 83, 84, 87, 88, 89, 91, 92, 94, 95, 96, 97, 98, 99, 101, 102, 103, 104, 105, 106]; lv2 = [7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 30, 31, 32, 33, 34, 36, 37, 38, 41, 42, 43, 45, 48, 55, 56, 59, 61, 62, 64, 71, 72, 73, 75, 76, 77, 79, 80, 81]

    if mode == 'train':
        print('@@@ dataloader init')
        train_dataset = CustomDataset(is_train=True, root=DATASET_PATH, split=train_split)
        val_dataset = CustomDataset(is_train=False, root=DATASET_PATH, split=train_split)
        wrong_note = [False for _ in range(len(train_dataset))]
        
        train_loader = data_loader(dataset=train_dataset, phase='train', batch_size=batch_size)
        if len(val_dataset) > 0:
            val_loader = data_loader(dataset=val_dataset, root=DATASET_PATH, phase='test', batch_size=batch_size)

        print('length of train_loader :', len(train_loader))

        global_iter = int(config.iteration)
        print(f'@@@ training with {torch.cuda.device_count()} GPU(s)')
        for epoch in range(99999999):
            print('@ Epoch', epoch)
            print('length of train_dataset :', len(train_dataset))
            print('length of train_loader :', len(train_loader))
            for iter_, data in enumerate(train_loader):
                model.train()
                time_ = time.time()
                global_iter += 1
                idx, x, label_0, label_1, label_2 = data
                if cuda:
                    idx = idx.cuda(); x = x.cuda(); label_0 = label_0.cuda().reshape(-1); label_1 = label_1.cuda().reshape(-1); label_2 = label_2.cuda().reshape(-1)
                
                B = x.shape[0]
                C0 = 12; C1 = 53; C2 = 42
                
                mask_0 = (label_0.reshape(B) != -100)
                mask_1 = (label_1.reshape(B) != -100)
                mask_2 = (label_2.reshape(B) != -100)
                pred0, pred1, pred2 = model(x)
                loss0 = criterion(pred0[mask_0], label_0[mask_0])
                loss1 = criterion(pred1[mask_1], label_1[mask_1])
                loss2 = criterion(pred2[mask_2], label_2[mask_2])
                loss = (loss0 + loss1 + loss2) / 3

                optimizer.zero_grad()
                loss.backward()

                if config.autoclip:
                    obs_grad_norm = _get_grad_norm(model)
                    grad_history.append(obs_grad_norm)
                    clip_value = np.percentile(grad_history, 10)
                    nn.utils.clip_grad_norm_(model.parameters(), clip_value)

                optimizer.step()
                if IS_ON_NSML:
                    report_dict = dict()
                    report_dict["train__loss"] = float(loss.item())
                    report_dict["train__lr"] = optimizer.param_groups[0]["lr"]
                    nsml.report(step=global_iter, **report_dict)
                scheduler.step()
                
                if global_iter == 1:
                    t = torch.cuda.get_device_properties(0).total_memory
                    r = torch.cuda.memory_reserved(0) 
                    a = torch.cuda.memory_allocated(0)
                    print(f'Total : {t/1.049e+6}, Reserved : {r/1.049e+6}, Allocated : {a/1.049e+6}')
                if global_iter % print_iter == 0:
                    print('[{:6d}] '.format(global_iter), end='')
                    print('Train {:.4f} ({:.4f}-{:.4f}-{:.4f}) / '.format(loss.item(), loss0.item(), loss1.item(), loss2.item()), end='')

                    # validation
                    if len(val_dataset) > 0:
                        with torch.no_grad():
                            model.eval()
                            score_sum = 0
                            scorei_sum = [0, 0, 0]
                            cnti = [0, 0, 0]
                            val_loss_sum = val_loss0_sum = val_loss1_sum = val_loss2_sum = 0.0
                            for iter_, data in enumerate(val_loader):
                                _, x, label_0, label_1, label_2 = data
                                labels = torch.stack([label_0.reshape(-1), label_1.reshape(-1), label_2.reshape(-1)], dim=-1)
                                if cuda:
                                    x = x.cuda(); label_0 = label_0.cuda()[:,0]; label_1 = label_1.cuda()[:,0]; label_2 = label_2.cuda()[:,0]

                                B = x.shape[0]
                                C0 = 12; C1 = 53; C2 = 42

                                mask_0 = (label_0.reshape(B) != -100)
                                mask_1 = (label_1.reshape(B) != -100)
                                mask_2 = (label_2.reshape(B) != -100)
                                pred0, pred1, pred2 = model(x)

                                loss0 = criterion(pred0[mask_0], label_0[mask_0])
                                loss1 = criterion(pred1[mask_1], label_1[mask_1])
                                loss2 = criterion(pred2[mask_2], label_2[mask_2])
                                loss = (loss0 + loss1 + loss2) / 3.0

                                val_loss_sum += loss.item()
                                val_loss0_sum += loss0.item()
                                val_loss1_sum += loss1.item()
                                val_loss2_sum += loss2.item()

                                # B * 3
                                prediction = torch.zeros(B, 3).to(pred0.device)
                                for i in range(B):
                                    res_cls0 = int(torch.argmax(pred0[i]))
                                    prediction[i][0] = res_cls0
                                    res_cls1 = int(torch.argmax(pred1[i]))
                                    prediction[i][1] = res_cls1
                                    res_cls2 = int(torch.argmax(pred2[i]))
                                    prediction[i][2] = res_cls2
                                for i in range(B):
                                    correct = total = 0
                                    for j in range(3):
                                        if labels[i][j] >= 0:
                                            total += 1
                                            if int(prediction[i][j]) == int(labels[i][j]):
                                                correct += 1
                                                scorei_sum[j] += 1
                                            cnti[j] += 1
                                        else:
                                            break
                                    score_sum += correct / total

                            if IS_ON_NSML:
                                report_dict = dict()
                                report_dict["val__loss"] = float(val_loss_sum / cnti[0])
                                report_dict["val__score"] = float(score_sum / cnti[0] * 100)
                                nsml.report(step=global_iter, **report_dict)
                            print('Val {:.4f} ({:.4f}-{:.4f}-{:.4f}) / '.format(val_loss_sum / len(val_loader), val_loss0_sum / len(val_loader), val_loss1_sum / len(val_loader), val_loss2_sum / len(val_loader)), end='')
                            print('Acc {:.2f} ({:.2f}-{:.2f}-{:.2f}) / '.format(score_sum / cnti[0] * 100, scorei_sum[0] / cnti[0] * 100, scorei_sum[1] / cnti[1] * 100, scorei_sum[2] / cnti[2] * 100), end='')
                            print('{:.2f}sec.'.format(time.time() - time_))
                    if IS_ON_NSML:
                        nsml.save(str(global_iter))
                if global_iter >= num_steps:
                    exit(0)
