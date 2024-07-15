import torch
# from utils.expert import weight_sum_var, ivw_aggregate_var
import numpy as np
import torch.nn as nn

def gaussian_reparameterization_var(means, var, times=1):
    std = torch.sqrt(var)
    res = torch.zeros_like(means).to(means.device)
    for t in range(times):
        epi = std.data.new(std.size()).normal_()
        res += epi * std + means
    return res/times
def fill_with_label(label_embedding,label,x_embedding,inc_V_ind):
    fea = label.matmul(label_embedding)/(label.sum(dim=1,keepdim=True)+1e-8)
    new_x =  x_embedding*inc_V_ind.T.unsqueeze(-1) + fea.unsqueeze(0)*(1-inc_V_ind.T.unsqueeze(-1))
    return new_x
class MLP(nn.Module):
    def __init__(self, in_dim,  out_dim,hidden_dim:list=[512,1024,1024,1024,512], act =nn.GELU,norm=nn.BatchNorm1d,dropout_rate=0.,final_act=True,final_norm=True):
        super(MLP, self).__init__()
        self.act = act
        self.norm = norm
        # init layers
        self.mlps =[]
        layers = []
        
        if len(hidden_dim)>0:
            layers.append(nn.Linear(in_dim, hidden_dim[0]))
            # layers.append(nn.Dropout(dropout_rate))
            layers.append(self.norm(hidden_dim[0]))
            layers.append(self.act())
            self.mlps.append(nn.Sequential(*layers))
            layers = []
            ##hidden layer
            for i in range(len(hidden_dim)-1):
                layers.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))
                # layers.append(nn.Dropout(dropout_rate))
                layers.append(self.norm(hidden_dim[i+1]))
                layers.append(self.act())
                self.mlps.append(nn.Sequential(*layers))
                layers = []
            ##output layer
            layers.append(nn.Linear(hidden_dim[-1], out_dim))
            if final_norm:
                layers.append(self.norm(out_dim))
            # layers.append(nn.Dropout(dropout_rate))
            if final_act:
                layers.append(self.act())
            self.mlps.append(nn.Sequential(*layers))
            layers = []
        else:
            layers.append(nn.Linear(in_dim, out_dim))
            if final_norm:
                layers.append(self.norm(out_dim))
            if final_act:
                layers.append(self.act())
            self.mlps.append(nn.Sequential(*layers))
        self.mlps = nn.ModuleList(self.mlps)
    def forward(self, x):
        for layers in self.mlps:
            x = layers(x)
            # x = x + y
        return x
class sharedQz_inference_mlp(nn.Module):
    def __init__(self, in_dim, out_dim,hidden_dim=[1024]):
        super(sharedQz_inference_mlp, self).__init__()
        self.transfer_act = nn.ReLU
        self.mlp = MLP(in_dim, out_dim,hidden_dim=hidden_dim)
        self.z_loc = nn.Linear(out_dim, out_dim)
        self.z_sca = nn.Sequential(nn.Linear(out_dim, out_dim), nn.Softplus())
        # self.qzv_inference = nn.Sequential(*self.qz_layer)

    def forward(self, x):
        hidden_features = self.mlp(x)
        z_mu = self.z_loc(hidden_features)
        z_sca = self.z_sca(hidden_features)
        # class_feature  = self.z
        return z_mu, z_sca
    
class inference_mlp(nn.Module):
    def __init__(self, in_dim, out_dim,hidden_dim=[1024]):
        super(inference_mlp, self).__init__()
        self.transfer_act = nn.ReLU
        self.mlp = MLP(in_dim, out_dim,hidden_dim=hidden_dim)

        # self.qzv_inference = nn.Sequential(*self.qz_layer)

    def forward(self, x):
        hidden_features = self.mlp(x)
        # class_feature  = self.z
        return hidden_features
    
class Px_generation_mlp(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=[512]):
        super(Px_generation_mlp, self).__init__()
        self.mlp = MLP(in_dim, out_dim,hidden_dim=hidden_dim,final_act=False,final_norm=False)
        # self.transfer_act = nn.ReLU
        # self.px_layer = mlp_layers_creation(self.z_dim, self.x_dim, self.layers, self.transfer_act)
        # self.px_z = nn.Sequential(*self.px_layer)

    def forward(self, z):
        xr = self.mlp(z)
        return xr
    
class VAE(nn.Module):
    def __init__(self, d_list,z_dim,class_num):
        super(VAE, self).__init__()
        self.x_dim_list = d_list
        self.k = class_num
        self.z_dim = z_dim
        self.num_views = len(d_list)
        

        # self.switch_layers = switch_layers(z_dim,self.num_views)

        
        self.z_inference = []
        for v in range(self.num_views):
            self.z_inference.append(inference_mlp(self.x_dim_list[v], self.z_dim))
        self.qz_inference = nn.ModuleList(self.z_inference)
        self.qz_inference_header = sharedQz_inference_mlp(self.z_dim, self.z_dim)
        self.x_generation = []
        for v in range(self.num_views):
            self.x_generation.append(Px_generation_mlp(self.z_dim,self.x_dim_list[v]))
        self.px_generation = nn.ModuleList(self.x_generation)
        self.px_generation2 = nn.ModuleList(self.x_generation)
    def inference_z(self, x_list):
        uniview_mu_list = []
        uniview_sca_list = []
        for v in range(self.num_views):
            if torch.sum(torch.isnan(x_list[v])).item() > 0:
                print("zzz:nan")
                pass
            fea = self.qz_inference[v](x_list[v])
            if torch.sum(torch.isnan(fea)).item() > 0:
                print("zz:nan")
                pass
            z_mu_v, z_sca_v = self.qz_inference_header(fea)
            if torch.sum(torch.isnan(z_mu_v)).item() > 0:
                print("zzmu:nan")
                pass
            if torch.sum(torch.isnan(z_sca_v)).item() > 0:
                print("zzvar:nan")
                pass
            uniview_mu_list.append(z_mu_v)
            uniview_sca_list.append(z_sca_v)





        # mu = torch.stack(z_mu)
        # sca = torch.stack(z_sca)
        
        ###POE aggregation
        # fusion_mu, fusion_sca = self.poe_aggregate(mu, sca, mask)

        ###weighted fusion
        # z = []
        # for v in range(self.num_views):
        #     z.append(gaussian_reparameterization_var(uniview_mu_list[v], uniview_sca_list[v],times=1))
        # z = torch.stack(z,dim=1) #[n v d]
        # z = z.mul(mask.unsqueeze(-1)).sum(1)
        # z = z / (mask.sum(1).unsqueeze(-1)+1e-8)

        return uniview_mu_list, uniview_sca_list
    
    def generation_x(self, z):
        
        xr_dist = []
        for v in range(self.num_views):
            xrs_loc = self.px_generation[v](z)
            xr_dist.append(xrs_loc)
        return xr_dist
    
    def generation_x_p(self, z):
        
        xr_dist = []
        for v in range(self.num_views):
            xrs_loc = self.px_generation2[v](z)
            xr_dist.append(xrs_loc)
        return xr_dist
    
    def poe_aggregate(self, mu, var, mask=None, eps=1e-5):
        if mask is None:
            mask_matrix = torch.ones_like(mu).to(mu.device)
        else:
            mask_matrix = mask.transpose(0,1).unsqueeze(-1)
        # mask_matrix = torch.stack(mask, dim=0)
        mask_matrix_new = torch.cat([torch.ones([1,mask_matrix.shape[1],mask_matrix.shape[2]]).cuda(),mask_matrix],dim=0)
        p_z_mu = torch.zeros([1,mu.shape[1],mu.shape[2]]).cuda()
        p_z_var = torch.ones([1,mu.shape[1],mu.shape[2]]).cuda()
        mu_new = torch.cat([p_z_mu,mu],dim=0)
        var_new = torch.cat([p_z_var,var],dim=0)
        exist_mu = mu_new * mask_matrix_new
        
        T = 1. / (var_new+eps)
        if torch.sum(torch.isnan(exist_mu)).item()>0:
            print('.')
        if torch.sum(torch.isinf(T)).item()>0:
            print('.')
        exist_T = T * mask_matrix_new
        aggregate_T = torch.sum(exist_T, dim=0)
        aggregate_var = 1. / (aggregate_T + eps)
        # if torch.sum(torch.isnan(aggregate_var)).item()>0:
        #     print('.')
        aggregate_mu = torch.sum(exist_mu * exist_T, dim=0) / (aggregate_T + eps)
        if torch.sum(torch.isnan(aggregate_mu)).item()>0:
            print(',')
        return aggregate_mu, aggregate_var
    
    def moe_aggregate(self, mu, var, mask=None, eps=1e-5):
        if mask is None:
            mask_matrix = torch.ones_like(mu).to(mu.device)
        else:
            mask_matrix = mask.transpose(0,1).unsqueeze(-1)
        exist_mu = mu * mask_matrix
        exist_var = var * mask_matrix
        aggregate_var = exist_var.sum(dim=0)
        aggregate_mu = exist_mu.sum(dim=0)
        return aggregate_mu,aggregate_var
    
    def forward(self, x_list, mask=None):
        uniview_mu_list, uniview_sca_list = self.inference_z(x_list)
        z_mu = torch.stack(uniview_mu_list,dim=0) # [v n d]
        z_sca = torch.stack(uniview_sca_list,dim=0) # [v n d]
        if torch.sum(torch.isnan(z_mu)).item() > 0:
            print("z:nan")
            pass
        
        # if self.training:
        #     z_mu = fill_with_label(label_embedding_mu,label,z_mu,mask)
        #     z_sca = fill_with_label(label_embedding_var,label,z_sca,mask)
        fusion_mu, fusion_sca = self.poe_aggregate(z_mu, z_sca, mask)
        

        if torch.sum(torch.isnan(fusion_mu)).item() > 0:
            pass
        assert torch.sum(fusion_sca<0).item() == 0
        z_sample = gaussian_reparameterization_var(fusion_mu, fusion_sca,times=10)
        if torch.sum(torch.isnan(z_sample)).item() > 0:
            print("z:nan")
            pass
        xr_list = self.generation_x(z_sample)
        # z_sample_list = []
        # for i,mu in enumerate(uniview_mu_list):
        #     z_sample_list.append(gaussian_reparameterization_var(uniview_mu_list[i],uniview_sca_list[i]))
            # z_sample_list.append((mu))
        # xr_list_views = self.generation_x_s1(z_sample_list)
        # c_z_sample = self.gaussian_rep_function(fusion_z_mu, fusion_z_sca)
        return z_sample, uniview_mu_list, uniview_sca_list, fusion_mu, fusion_sca, xr_list
