import os
import numpy as np
import torch
import argparse
import warnings
import random
from model import PFSNet, Enhancer
from datasets.dataloader import Crowd
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import datetime
from os.path import join as pjn
import torchvision
import torch.nn.functional as F

warnings.filterwarnings('ignore')

# define the GPU id to be used
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def clip(input):
    mx = torch.Tensor([2.2489, 2.4286, 2.6400]).reshape(1,3,1,1).cuda()
    mn = torch.Tensor([-2.1179, -2.0357, -1.8044]).reshape(1,3,1,1).cuda()
    output = torch.relu(input - mn) + mn
    output = -torch.relu(-output + mx) + mx
    return output 

class SPP(torch.nn.Module):
    def __init__(self):
        super(SPP, self).__init__()        
        self.aap2 = torch.nn.AdaptiveAvgPool2d(2)
        self.aap4 = torch.nn.AdaptiveAvgPool2d(4)
        self.aap8 = torch.nn.AdaptiveAvgPool2d(8)
    
    def forward(self, x):
        x1 = self.aap2(x).reshape(x.size(0), x.size(1)*4)
        x2 = self.aap4(x).reshape(x.size(0), x.size(1)*16)
        x3 = self.aap8(x).reshape(x.size(0), x.size(1)*64)
        x = torch.cat([x1, x2, x3], dim=1)
        return x

def get_args_parser():
    # define the argparse for the script
    parser = argparse.ArgumentParser('Inference setting', add_help=False)
    parser.add_argument('--model_path', type=str, help='path of pre-trained model')
    parser.add_argument('--data_path', type=str, help='root path of the dataset')
    parser.add_argument('--save_path', type=str, default='/root/checkpoints', help='root path of checkpoints') 

    parser.add_argument('--batch_size', type=int, default=8, help='batch size in training')
    parser.add_argument('--log_para', type=int, default=1000, help='magnify the target density map')
    parser.add_argument('--step', type=str, default='train', help='train or test')
    parser.add_argument('--crop_size', type=int, default=128, help='crop size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=1000, help='epoch')
    parser.add_argument('--wdecay', type=float, default=1e-4)
    parser.add_argument('--debug', type=str, default='no', help='yes')
    parser.add_argument('--seed', type=int, default=100)

    parser.add_argument('--final_counting_loss_ratio', type=float, default=1.)
    parser.add_argument('--level_counting_loss_ratio', type=float, default=1.)
    parser.add_argument('--consistency_loss_ratio', type=float, default=1)
    
    parser.add_argument('--enhancer', type=str, default='no', help='working word: enhancer/yes/y')
    parser.add_argument('--repeat', type=int, default=5, help='enhancer augmentation repeat factor')
    
    parser.add_argument('--reduction', type=int, default=4, help='reduction rate')
    parser.add_argument('--multi_gpu', type=str, default='no', help='working work: yes')
    
    return parser

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.cur_val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    # update the moving average
    def update(self, cur_val):
        self.cur_val = cur_val
        self.sum += cur_val
        self.count += 1
        self.avg = self.sum / self.count

def main(args):
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False 
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.debug != 'yes':
        name = args.data_path.split('/')[-1].split('_')[1]
        dt = datetime.datetime.today()
        dt = dt.strftime('%Y-%m-%d-%H:%M:%S')
    best_mae = 1e+20
    best_mae_mse = 1e+20

    model = PFSNet(pretrained=True, args=args).cuda()
    
    if args.multi_gpu == 'yes':
        print("MULTI GPU")
        model = torch.nn.DataParallel(model)
        
    if args.enhancer == 'yes':
        print(" ENHANCER ")
        enhancer = Enhancer().cuda()
        model.load_state_dict(torch.load(args.model_path)['model_state_dict'])
        

    # load the trained model
    train_set = Crowd(args.data_path, crop_size=args.crop_size, enhancer=args.enhancer, repeat=args.repeat, method='train', log_para=args.log_para)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=4, shuffle=True, drop_last=False)
    
    val_set = Crowd(args.data_path, method='val')
    val_loader = DataLoader(val_set, batch_size=1, num_workers=4, shuffle=False, drop_last=False)
    
    spp = SPP()
    l2loss = torch.nn.MSELoss(reduction='mean')
    
    if args.enhancer == 'yes':   
        optimizer = torch.optim.Adam(list(enhancer.parameters()), lr=args.learning_rate, weight_decay=args.wdecay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.wdecay)
    
    for epoch in tqdm(range(1, args.epochs+1)):
        if args.enhancer == 'yes':
            enhancer.train()
            model.eval()
        else:
            model.train()
        
        total_losses = AverageMeter()
        final_counting_losses = AverageMeter()
        level_counting_losses = AverageMeter()
        consistency_losses = AverageMeter()
        
        for i, sample in enumerate(train_loader):
            total_loss = 0
            iter_batch_size = sample[0].size(0)
            img = sample[0].reshape(-1, 3, args.crop_size, args.crop_size)
            target = sample[1].reshape(-1, args.crop_size, args.crop_size)
            optimizer.zero_grad()
            
            img = img.cuda()
            target = target.cuda()
            if args.enhancer == 'yes':
                img_ = img/5
                value, gamma = enhancer(img_)
                image = clip(value+img_*gamma)
                
                s = spp(image).reshape(int(iter_batch_size*4), int(args.repeat), -1)
                consistency_loss = s.std(dim=1).mean()
                total_loss += consistency_loss*args.consistency_loss_ratio
                consistency_losses.update(consistency_loss.cpu().detach())
                
                img = (image*5).reshape(int(iter_batch_size*4), int(args.repeat), 3, args.crop_size, args.crop_size).mean(dim=1)
                
            pred_map, x1_density, x2_density, x3_density, x4_density, x5_density = model(img)
            
            if args.level_counting_loss_ratio > 0:
                level_counting_loss = (l2loss(x1_density.squeeze(1), target) + l2loss(x2_density.squeeze(1), target) + l2loss(x3_density.squeeze(1), target) + l2loss(x4_density.squeeze(1), target) + l2loss(x5_density.squeeze(1), target))/5
                total_loss += level_counting_loss * args.level_counting_loss_ratio 
                level_counting_losses.update(level_counting_loss.cpu().detach())
            else:
                level_counting_loss = 0
                level_counting_losses.update(level_counting_loss)

            final_counting_loss = l2loss(pred_map.squeeze(1), target)
            total_loss += args.final_counting_loss_ratio*final_counting_loss
            final_counting_losses.update(final_counting_loss.cpu().detach())
                
            total_losses.update(total_loss.cpu().detach())
            total_loss.backward()
            
            optimizer.step()

            if args.enhancer == 'yes':
                print('Epoch %d total_loss: %.3f cons_loss: %.3f final_count_loss: %.3f level_count_loss: %.3f' %(epoch, consistency_losses.avg, total_losses.avg, final_counting_losses.avg, level_counting_losses.avg))
            else:
                print('Epoch %d total_loss: %.3f final_count_loss: %.3f level_count_loss: %.3f' %(epoch, total_losses.avg, final_counting_losses.avg, level_counting_losses.avg))
                
        if epoch % 1 == 0:
            with torch.no_grad():
                model.eval()
                if args.enhancer == 'yes':
                    enhancer.eval()
                maes = AverageMeter()
                maes1 = AverageMeter()
                maes2 = AverageMeter()
                maes3 = AverageMeter()
                maes4 = AverageMeter()
                maes5 = AverageMeter()
                maes_avg = AverageMeter()
                mses = AverageMeter()
                # iterate over the dataset
                for vi, sample in enumerate(val_loader, 0):
                    img = sample[0]
                    gt_map = sample[1]
                    img = img.cuda()

                    gt_map = gt_map.type(torch.FloatTensor).unsqueeze(0).numpy()
                    if args.enhancer == 'yes':
                        img_ = img/5
                        value, gamma = enhancer(img_)
                        img = clip((value + img_*gamma)*5)

                    # get the predicted density map
                    pred_map, x1_density, x2_density, x3_density, x4_density, x5_density = model(img)
                    
                    pred_map = pred_map.data.cpu().numpy()
                    pred_map1 = x1_density.data.cpu().numpy()
                    pred_map2 = x2_density.data.cpu().numpy()
                    pred_map3 = x3_density.data.cpu().numpy()
                    pred_map4 = x4_density.data.cpu().numpy()
                    pred_map5 = x5_density.data.cpu().numpy()

                    # evaluation over the batch
                    for i_img in range(pred_map.shape[0]):
                        pred_cnt = np.sum(pred_map[i_img]) / args.log_para
                        gt_count = np.sum(gt_map[i_img])
                        pred_cnt1 = np.sum(pred_map1[i_img]) / args.log_para
                        pred_cnt2 = np.sum(pred_map2[i_img]) / args.log_para
                        pred_cnt3 = np.sum(pred_map3[i_img]) / args.log_para
                        pred_cnt4 = np.sum(pred_map4[i_img]) / args.log_para
                        pred_cnt5 = np.sum(pred_map5[i_img]) / args.log_para
                        maes1.update(abs(gt_count - pred_cnt1))
                        maes2.update(abs(gt_count - pred_cnt2))
                        maes3.update(abs(gt_count - pred_cnt3))
                        maes4.update(abs(gt_count - pred_cnt4))
                        maes5.update(abs(gt_count - pred_cnt5))
                    
                        maes.update(abs(gt_count - pred_cnt))
                        mses.update((gt_count - pred_cnt) * (gt_count - pred_cnt))
                
                # calculation mae and mre
                mae = maes.avg
                mse = np.sqrt(mses.avg)
                
                if args.debug != 'yes':
                    if mae < best_mae:
                        best_mae = mae
                        best_mae_mse = mse
                        nm = f'epoch_{epoch:04d}.pth'
                        path = pjn(args.save_path, dt)
                        if not os.path.isdir(pjn(path)):
                            os.mkdir(pjn(path))
                        if args.enhancer == 'yes':
                            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'enhancer_state_dict': enhancer.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, pjn(path, nm))
                        else:
                            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, pjn(path, nm))
                    # print the results
                print('Epoch %d    [mae %.3f mse %.3f]' % (epoch, mae, mse))
                

def test(args):
    model = PFSNet(pretrained=True, args=args).cuda()
    if args.multi_gpu == 'yes':
        print("MULTI GPU")
        model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path)['model_state_dict'])
    
    if args.enhancer == 'yes':
        print(" ENHANCER ")
        enhancer = Enhancer().cuda()
        enhancer.load_state_dict(torch.load(args.model_path)['enhancer_state_dict'])
    print('successfully load model from', args.model_path)
    
    test_set = Crowd(args.data_path, method='test')
    test_loader = DataLoader(test_set, batch_size=1, num_workers=4, shuffle=False, drop_last=False)

    with torch.no_grad():
        model.eval()
        if args.enhancer == 'yes':
            enhancer.eval()
        maes = AverageMeter()
        maes1 = AverageMeter()
        maes2 = AverageMeter()
        maes3 = AverageMeter()
        maes4 = AverageMeter()
        maes5 = AverageMeter()
        mses = AverageMeter()
        
        # iterate over the dataset
        for vi, data in enumerate(test_loader, 0):
            img, gt_map = data
            img = img.cuda()
            gt_map = gt_map.type(torch.FloatTensor).unsqueeze(0).cuda()
            if args.enhancer == 'yes':
                img_ = img/5
                value, gamma = enhancer(img_)
                img = clip((value + img_*gamma)*5)

            # get the predicted density map
            pred_map, x1_density, x2_density, x3_density, x4_density, x5_density, *l = model(img)

            pred_map = pred_map.data.cpu().numpy()
            gt_map = gt_map.data.cpu().numpy()
            pred_map1 = x1_density.data.cpu().numpy()
            pred_map2 = x2_density.data.cpu().numpy()
            pred_map3 = x3_density.data.cpu().numpy()
            pred_map4 = x4_density.data.cpu().numpy()
            pred_map5 = x5_density.data.cpu().numpy()

            # evaluation over the batch
            for i_img in range(pred_map.shape[0]):
                pred_cnt = np.sum(pred_map[i_img]) / args.log_para
                gt_count = np.sum(gt_map[i_img])
                pred_cnt1 = np.sum(pred_map1[i_img]) / args.log_para
                pred_cnt2 = np.sum(pred_map2[i_img]) / args.log_para
                pred_cnt3 = np.sum(pred_map3[i_img]) / args.log_para
                pred_cnt4 = np.sum(pred_map4[i_img]) / args.log_para
                pred_cnt5 = np.sum(pred_map5[i_img]) / args.log_para
                maes1.update(abs(gt_count - pred_cnt1))
                maes2.update(abs(gt_count - pred_cnt2))
                maes3.update(abs(gt_count - pred_cnt3))
                maes4.update(abs(gt_count - pred_cnt4))
                maes5.update(abs(gt_count - pred_cnt5))

                maes.update(abs(gt_count - pred_cnt))
                mses.update((gt_count - pred_cnt) * (gt_count - pred_cnt))    
        # calculation mae and mre
        mae = maes.avg
        mse = np.sqrt(mses.avg)
        
        # print the results
        print('=' * 50)
        print('    ' + '-' * 20)
        print('    [mae %.3f mse %.3f]' % (mae, mse))
        print('    [mae1 %.3f mae2 %.3f mae3 %.3f mae4 %.3f mae5 %.3f]' % (maes1.avg, maes2.avg, maes3.avg, maes4.avg, maes5.avg))
        print('    ' + '-' * 20)
        print('=' * 50) 
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser('PFSNet inference', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.step == 'test':
        test(args)
    else:
        main(args)
