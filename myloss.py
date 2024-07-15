import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
def gaussian_reparameterization_var(means, var, times=1):
    std = torch.sqrt(var)
    res = torch.zeros_like(means).to(means.device)
    for t in range(times):
        epi = std.data.new(std.size()).normal_()
        res += epi * std + means
    return res/times
def W_distance(p_mu,p_v,q_mu,q_v,eps=1e-9):
    p_var = p_v+eps
    q_var = q_v+eps
    res = torch.sum((p_mu-q_mu)**2,dim=-1)+p_var.sum(-1)+q_var.sum(-1)- 2*((p_var*q_var)**0.5).sum(-1)
    return res
def gaussian_log_var(x, mu, var, eps=1e-14):
    return -0.5 * (torch.log(torch.tensor(2.0 * np.pi)) + torch.log(var) + torch.pow(x - mu, 2) / var)

def kl_div_var(q_mu, q_var, p_mu, p_var, eps=1e-12):
    
    return 0.5 * (torch.log((p_var+eps) / (q_var+eps)) + (q_var+eps) / (p_var+eps) + torch.pow(q_mu - p_mu, 2) / (p_var+eps) - 1)

def cosdis(x1,x2):
    return (1-torch.cosine_similarity(x1,x2,dim=-1))/2

def vade_trick(mc_sample, gmm_pi, gmm_mu, gmm_sca):
    
        log_pz_c = torch.mean(gaussian_log_var(mc_sample.unsqueeze(1), gmm_mu.unsqueeze(0), gmm_sca.unsqueeze(0)), dim=-1)

        log_pc = torch.log(gmm_pi.unsqueeze(0))
        log_pc_z = log_pc + log_pz_c
        pc_z = torch.exp(log_pc_z) + 1e-10
        normalize_pc_z = pc_z / torch.sum(pc_z, dim=1, keepdim=True)
        return normalize_pc_z

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
    



    def z_c_loss_new(self,z_mu, label,  c_mu,inc_L_ind):
        label_inc = label.mul(inc_L_ind)
        
        sample_label_emb = (label_inc.matmul(c_mu))/(label_inc.sum(-1)+1e-9).unsqueeze(-1)
        loss = ((z_mu-sample_label_emb)**2)
        # print(loss.mean())
        return loss.mean()






    def corherent_loss(self,uniview_dist_mu, uniview_dist_sca, aggregate_mu, aggregate_sca, mask=None):
        if mask is None:
            mask = torch.ones_like((aggregate_mu.shape[0],len(uniview_dist_mu))).to(aggregate_mu.device)

        z_tc_loss = []
        # mask_stack = torch.concatenate(mask, dim=1)
        norm = torch.sum(mask, dim=1)
        weight = F.softmax(torch.stack(uniview_dist_sca,dim=1),dim=1)
        for v in range(len(uniview_dist_mu)):
            # zv_tc_loss = torch.mean(kl_div_var(aggregate_mu, aggregate_sca, uniview_dist_mu[v], uniview_dist_sca[v])*weight[:,v,:],
            #                     dim=1)
            zv_tc_loss = torch.mean(kl_div_var(aggregate_mu, aggregate_sca, uniview_dist_mu[v], uniview_dist_sca[v]),
                                dim=1)
            exist_loss = zv_tc_loss * mask[:,v]
            z_tc_loss.append(exist_loss)
        z_tc_loss = torch.stack(z_tc_loss,dim=1)

        sample_ave_tc_term_loss = torch.sum(z_tc_loss) / mask.sum()

        return sample_ave_tc_term_loss

    def weighted_BCE_loss(self,pred,label,inc_L_ind,reduction='mean'):
        assert torch.sum(torch.isnan(torch.log(pred))).item() == 0
        assert torch.sum(torch.isnan(torch.log(1 - pred + 1e-5))).item() == 0
        res=torch.abs((label.mul(torch.log(pred + 1e-5)) \
                                                + (1-label).mul(torch.log(1 - pred + 1e-5))).mul(inc_L_ind))
        assert torch.sum(torch.isnan(res)).item() == 0
        assert torch.sum(torch.isinf(res)).item() == 0

        if reduction=='mean':
            return torch.sum(res)/torch.sum(inc_L_ind)
        elif reduction=='sum':
            return torch.sum(res)
        elif reduction=='none':
            return res
            
    def weighted_wmse_loss(self,input, target, weight, reduction='mean'):
        ret = (torch.diag(weight).mm(target - input)) ** 2
        if torch.sum(torch.isnan(ret)).item()>0:
            print(ret)
        if reduction == 'mean':
            return torch.mean(ret)
        elif reduction=='sum':
            return torch.sum(ret)
        elif reduction=='none':
            return ret


    
    