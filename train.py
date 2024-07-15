import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import os.path as osp
import utils
from utils import AverageMeter
import MLdataset
import argparse
import time
from model_new import get_model
import evaluation
import torch
import numpy as np
import copy
from myloss import Loss, vade_trick
from torch import nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CosineAnnealingLR



# from constructGraph import getMvKNNGraph, getIncMvKNNGraph

    

def view_mixup(data_list,label,v_mask,l_mask):
    new_data = []
    new_labels = label.unsqueeze(0).repeat(len(data_list),1,1)
    num = data_list[0].shape[0]
    for i,view_data in enumerate(data_list):
        indices = torch.randperm(num).cuda()
        new_data.append(torch.index_select(view_data, 0, indices))
        new_labels[i] = torch.index_select(new_labels[i],0,indices)
        v_mask[:,i] = torch.index_select(v_mask[:,i],0,indices)
        
    label = new_labels.sum(dim=0)
    label = torch.masked_fill(label,label>0,1)
    l_mask = torch.masked_fill(l_mask,label>0,1)
    return new_data,label,v_mask,l_mask


def train(loader, model, loss_model, opt, sche, epoch,dep_graph,last_preds,logger):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    mce= nn.MultiLabelSoftMarginLoss()
    model.train()
    end = time.time()
    All_preds = torch.tensor([]).cuda()
    for i, (data, label, inc_V_ind, inc_L_ind) in enumerate(loader):
        data_time.update(time.time() - end)
        data=[v_data.to('cuda:0') for v_data in data]

        label = label.to('cuda:0')
        inc_V_ind = inc_V_ind.float().to('cuda:0')
        inc_L_ind = inc_L_ind.float().to('cuda:0')


        z_sample, uniview_mu_list, uniview_sca_list, fusion_z_mu, fusion_z_sca, xr_list, label_emb_sample, pred, label_emb,label_emb_var,p_xr_list = model(data,mask=inc_V_ind )

        All_preds = torch.cat([All_preds,pred],dim=0)
        

        
        if epoch<args.pre_epochs:
            loss_mse_viewspec = 0
            loss_CL_views = 0
            loss_list=[]

            loss = loss_CL_views
            assert torch.sum(torch.isnan(loss)).item() == 0
            assert torch.sum(torch.isinf(loss)).item() == 0
        else:
            
            loss_CL = loss_model.weighted_BCE_loss(pred,label,inc_L_ind)

            z_c_loss = loss_model.z_c_loss_new(z_sample, label, label_emb_sample,inc_L_ind)
            cohr_loss = loss_model.corherent_loss(uniview_mu_list, uniview_sca_list,fusion_z_mu, fusion_z_sca,mask=inc_V_ind)

            loss_mse = 0
            for v in range(len(data)):
                loss_mse += loss_model.weighted_wmse_loss(data[v],xr_list[v],inc_V_ind[:,v],reduction='mean')

            assert torch.sum(torch.isnan(loss_mse)).item() == 0
            loss = loss_CL + loss_mse *args.alpha + z_c_loss*args.beta + cohr_loss *args.sigma 
        # loss = loss_CL
        opt.zero_grad()
        loss.backward()
        if isinstance(sche,CosineAnnealingWarmRestarts):
            sche.step(epoch + i / len(loader))
        
        opt.step()
        # print(model.classifier.parameters().grad)
        losses.update(loss.item())
        batch_time.update(time.time()- end)
        end = time.time()
    if isinstance(sche,StepLR):
        sche.step()
    logger.info('Epoch:[{0}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'Loss {losses.avg:.3f}'.format(
                        epoch,   batch_time=batch_time,
                        data_time=data_time, losses=losses))
    return losses,model,All_preds,label_emb_sample

def test(loader, model, loss_model, epoch,logger):
    batch_time = AverageMeter()
    losses = AverageMeter()
    total_labels = []
    total_preds = []
    model.eval()
    end = time.time()
    for i, (data, label, inc_V_ind, inc_L_ind) in enumerate(loader):
        # data_time.update(time.time() - end)
        data=[v_data.to('cuda:0') for v_data in data]
        
        # pred,_,_ = model(data,mask=torch.ones_like(inc_V_ind).to('cuda:0'))
        z_sample, uniview_mu_list, uniview_sca_list, fusion_z_mu, fusion_z_sca, xr_list, label_embedding_sample, qc_z, label_emb,label_emb_var, xr_list_views = model(data,mask=inc_V_ind.to('cuda:0'))
        # qc_x = vade_trick(fusion_z_mu, model.mix_prior, model.mix_mu, model.mix_sca)
        pred = qc_z
        pred = pred.cpu()
        total_labels = np.concatenate((total_labels,label.numpy()),axis=0) if len(total_labels)>0 else label.numpy()
        total_preds = np.concatenate((total_preds,pred.detach().numpy()),axis=0) if len(total_preds)>0 else pred.detach().numpy()
        
        loss=loss_model.weighted_BCE_loss(pred,label,inc_L_ind)
        
        losses.update(loss.item())
        batch_time.update(time.time()- end)
        end = time.time()
    total_labels=np.array(total_labels)
    total_preds=np.array(total_preds)

    evaluation_results=evaluation.do_metric(total_preds,total_labels)
    logger.info('Epoch:[{0}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Loss {losses.avg:.3f}\t'
                  'AP {ap:.3f}\t'
                  'HL {hl:.3f}\t'
                  'RL {rl:.3f}\t'
                  'AUC {auc:.3f}\t'.format(
                        epoch,   batch_time=batch_time,
                        losses = losses,
                        ap=evaluation_results[0], 
                        hl=evaluation_results[1],
                        rl=evaluation_results[2],
                        auc=evaluation_results[3]
                        ))
    return evaluation_results


def main(args,file_path):
    data_path = osp.join(args.root_dir, args.dataset, args.dataset+'_six_view.mat')
    fold_data_path = osp.join(args.root_dir, args.dataset, args.dataset+'_six_view_MaskRatios_' + str(
                                args.mask_view_ratio) + '_LabelMaskRatio_' +
                                str(args.mask_label_ratio) + '_TraindataRatio_' + 
                                str(args.training_sample_ratio) + '.mat')
    
    folds_num = args.folds_num
    folds_results = [AverageMeter() for i in range(9)]
    if args.logs:
        logfile = osp.join(args.logs_dir,args.name+args.dataset+'_V_' + str(
                                    args.mask_view_ratio) + '_L_' +
                                    str(args.mask_label_ratio) + '_T_' + 
                                    str(args.training_sample_ratio) + '_'+str(args.alpha)+'_'+str(args.beta)+'.txt')
    else:
        logfile=None
    logger = utils.setLogger(logfile)
    device = torch.device('cuda:0')
    label_Inp_list = []
    for fold_idx in range(folds_num):
        fold_idx=fold_idx
        train_dataloder,train_dataset = MLdataset.getIncDataloader(data_path, fold_data_path,training_ratio=args.training_sample_ratio,fold_idx=fold_idx,mode='train',batch_size=args.batch_size,shuffle = True,num_workers=4)
        test_dataloder,test_dataset = MLdataset.getIncDataloader(data_path, fold_data_path,training_ratio=args.training_sample_ratio,val_ratio=0.15,fold_idx=fold_idx,mode='test',batch_size=args.batch_size,num_workers=4)
        val_dataloder,val_dataset = MLdataset.getIncDataloader(data_path, fold_data_path,training_ratio=args.training_sample_ratio,fold_idx=fold_idx,mode='val',batch_size=args.batch_size,num_workers=4)
        d_list = train_dataset.d_list
        classes_num = train_dataset.classes_num     
        labels = torch.tensor(train_dataset.cur_labels).float().to('cuda:0')
        dep_graph = torch.matmul(labels.T,labels)
        dep_graph = dep_graph/(torch.diag(dep_graph).unsqueeze(1)+1e-10)
        # dep_graph[dep_graph<=args.sigma]=0.
        dep_graph.fill_diagonal_(fill_value=0.)
        pri_c = train_dataset.cur_labels.sum(axis=0)/train_dataset.cur_labels.shape[0]
        pri_c = torch.tensor(pri_c).cuda()
        model=get_model(d_list,num_classes=classes_num,z_dim=args.z_dim,adj=dep_graph,rand_seed=0)
        # print(model)
        loss_model = Loss()
        # crit = nn.BCELoss()
        # optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)
        optimizer = Adam(model.parameters(), lr=args.lr)
        # optimizer = Adam([{"params": model.VAE.parameters(), 'lr': args.lr},
                                                    # {"params": model.mix_mu, 'lr': args.lr}, 
                                                    # {"params": model.mix_sca, 'lr': args.lr},
                                                    # {"params": model.mix_prior, 'lr': args.lr},
                                                    # ])
        # scheduler = StepLR(optimizer, step_size=5, gamma=0.85)
        # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=4, T_mult=2)
        scheduler = None
        

        logger.info('train_data_num:'+str(len(train_dataset))+'  test_data_num:'+str(len(test_dataset))+'   fold_idx:'+str(fold_idx))
        print(args)
        static_res = 0
        epoch_results = [AverageMeter() for i in range(9)]
        total_losses = AverageMeter()
        train_losses_last = AverageMeter()
        best_epoch=0
        best_model_dict = {'model':model.state_dict(),'epoch':0}
        for epoch in range(args.epochs):
            # tt=time.time()
            if epoch==0:
                All_preds = None
            train_losses,model,All_preds,label_emb_sample = train(train_dataloder,model,loss_model,optimizer,scheduler,epoch,dep_graph,All_preds,logger)
            # print("traintime:",time.time()-tt)
            label_InP = label_emb_sample.mm(label_emb_sample.t())
            # test_results = test(test_dataloder,model,loss_model,epoch,dep_graph,logger)
            # tt=time.time()
            
            if epoch>=args.pre_epochs:
                val_results = test(val_dataloder,model,loss_model,epoch,logger)
                # print("testtime:",time.time()-tt)
                # for i,re in enumerate(epoch_results):
                    # re.update(test_results[i])
                
                if val_results[0]*0.25+val_results[2]*0.25+val_results[3]*0.25>=static_res:
                    static_res = val_results[0]*0.25+val_results[2]*0.25+val_results[3]*0.25
                    best_model_dict['model'] = copy.deepcopy(model.state_dict())
                    best_model_dict['epoch'] = epoch
                    best_epoch=epoch
                train_losses_last = train_losses
                total_losses.update(train_losses.sum)
        model.load_state_dict(best_model_dict['model'])
        print("epoch",best_model_dict['epoch'])
        test_results = test(test_dataloder,model,loss_model,epoch,logger)
        label_Inp_list.append(label_InP.cpu().detach().numpy())
        logger.info('final: fold_idx:{} best_epoch:{}\t best:ap:{:.4}\t HL:{:.4}\t RL:{:.4}\t AUC_me:{:.4}\n'.format(fold_idx,best_epoch,test_results[0],test_results[1],
            test_results[2],test_results[3]))

        for i in range(9):
            folds_results[i].update(test_results[i])
        if args.save_curve:
            np.save(osp.join(args.curve_dir,args.dataset+'_V_'+str(args.mask_view_ratio)+'_L_'+str(args.mask_label_ratio))+'_'+str(fold_idx)+'.npy', np.array(list(zip(epoch_results[0].vals,train_losses.vals))))
    np.save(f"mid_res/label_InP_{args.dataset}.npy",np.array(label_Inp_list))
    file_handle = open(file_path, mode='a')
    if os.path.getsize(file_path) == 0:
        file_handle.write(
            'AP 1-HL 1-RL AUCme 1-oneE 1-Cov macAUC macro_f1 micro_f1 lr alpha beta gamma sigma\n')
    # generate string-result of 9 metrics and two parameters
    res_list = [str(round(res.avg,3))+'+'+str(round(res.std,3)) for res in folds_results]
    res_list.extend([str(args.lr),str(args.alpha),str(args.beta),str(args.gamma),str(args.sigma)])
    res_str = ' '.join(res_list)
    file_handle.write(res_str)
    file_handle.write('\n')
    file_handle.close()
        

def filterparam(file_path,index):
    params = []
    if os.path.exists(file_path):
        file_handle = open(file_path, mode='r')
        lines = file_handle.readlines()
        lines = lines[1:] if len(lines)>1 else []
        params = [[float(line.split(' ')[idx]) for idx in index] for line in lines ]
    return params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # misc
    working_dir = osp.dirname(osp.abspath(__file__)) 
    parser.add_argument('--logs-dir', type=str, metavar='PATH', 
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--logs', default=False, type=bool)
    parser.add_argument('--records-dir', type=str, metavar='PATH', 
                        default=osp.join(working_dir, 'final_records'))
    parser.add_argument('--file-path', type=str, metavar='PATH', 
                        default='')
    parser.add_argument('--root-dir', type=str, metavar='PATH', 
                        default='data/')
    parser.add_argument('--dataset', type=str, default='')#mirflickr corel5k pascal07 iaprtc12 espgame
    parser.add_argument('--datasets', type=list, default=['corel5k'])
    parser.add_argument('--mask-view-ratio', type=float, default=0.5)
    parser.add_argument('--mask-label-ratio', type=float, default=0.5)
    parser.add_argument('--training-sample-ratio', type=float, default=0.7)
    parser.add_argument('--folds-num', default=10, type=int)
    parser.add_argument('--weights-dir', type=str, metavar='PATH', 
                        default=osp.join(working_dir, 'weights'))
    parser.add_argument('--curve-dir', type=str, metavar='PATH', 
                        default=osp.join(working_dir, 'curves'))
    parser.add_argument('--save-curve', default=False, type=bool)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--workers', default=8, type=int)
    
    parser.add_argument('--name', type=str, default='final_')
    # Optimization args
    parser.add_argument('--lr', type=float, default=1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=200) #200 for corel5k  100 for iaprtc12 50 for pascal07 100 for espgame
    parser.add_argument('--pre_epochs', type=int, default=0)
    # Training args
    parser.add_argument('--z_dim', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--alpha', type=float, default=1e-1)
    parser.add_argument('--beta', type=float, default=1e-1)
    parser.add_argument('--gamma', type=float, default=1e-1)
    parser.add_argument('--sigma', type=float, default=0.)

    
    args = parser.parse_args()
    
    if args.logs:
        if not os.path.exists(args.logs_dir):
            os.makedirs(args.logs_dir)
    if args.save_curve:
        if not os.path.exists(args.curve_dir):
            os.makedirs(args.curve_dir)
    if True:
        if not os.path.exists(args.records_dir):
            os.makedirs(args.records_dir)
    lr_list = [1e-3]
    alpha_list = [1e0]# 
    beta_list = [1e-3]#1e-3 for corel5k and mirflickr, 1e0 for pascal07, 1e-1 for espgame, 1e0 for iaprtc12
    gamma_list = [0]
    sigma_list = [1e0]#1e0for others ,1e-1 for mirflickr
    if args.lr >= 0.01:
        args.momentumkl = 0.90
    for lr in lr_list:
        args.lr = lr
        for alpha in alpha_list:
            args.alpha = alpha
            for beta in beta_list:
                args.beta = beta
                for gamma in gamma_list:
                    args.gamma = gamma
                    for sigma in sigma_list:
                        args.sigma = sigma
                        for dataset in args.datasets:
                            args.dataset = dataset
                            file_path = osp.join(args.records_dir,args.name+args.dataset+'_VM_' + str(
                                            args.mask_view_ratio) + '_LM_' +
                                            str(args.mask_label_ratio) + '_T_' + 
                                            str(args.training_sample_ratio) + '.txt')
                            args.file_path = file_path
                            existed_params = filterparam(file_path,[-5,-4,-3,-2,-1])
                            if [args.lr,args.alpha,args.beta,args.gamma,args.sigma] in existed_params:
                                print('existed param! beta:{}'.format(args.beta))
                                # continue
                            main(args,file_path)