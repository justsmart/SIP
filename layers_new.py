import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import math
class MLP(nn.Module):
    # 100/1000..., 512, []
    def __init__(self, in_features, out_features, hidden_features=[], batchNorm=False,
                 nonlinearity='leaky_relu', negative_slope=0.1,
                 with_output_nonlineartity=True):
        super(MLP, self).__init__()
        self.nonlinearity = nonlinearity
        self.negative_slope = negative_slope
        # 创建容器
        self.fcs = nn.ModuleList()
        if hidden_features:
            in_dims = [in_features] + hidden_features
            out_dims = hidden_features + [out_features]
            for i in range(len(in_dims)):
                self.fcs.append(nn.Linear(in_dims[i], out_dims[i]))
                if with_output_nonlineartity or i < len(hidden_features):
                    if batchNorm:
                        self.fcs.append(nn.BatchNorm1d(out_dims[i], track_running_stats=True))
                    if nonlinearity == 'relu':
                        self.fcs.append(nn.ReLU(inplace=True))
                    elif nonlinearity == 'leaky_relu':
                        self.fcs.append(nn.LeakyReLU(negative_slope, inplace=True))
                    else:
                        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))
        # 上面没有用，只用了这个
        else:
            self.fcs.append(nn.Linear(in_features, out_features))
            if with_output_nonlineartity:
                # 一直都是false
                if batchNorm:
                    self.fcs.append(nn.BatchNorm1d(out_features, track_running_stats=True))
                if nonlinearity == 'relu':
                    self.fcs.append(nn.ReLU(inplace=True))
                elif nonlinearity == 'leaky_relu':
                    self.fcs.append(nn.LeakyReLU(negative_slope, inplace=True))
                else:
                    raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        for l in self.fcs:
            if l.__class__.__name__ == 'Linear':
                nn.init.kaiming_uniform_(l.weight, a=self.negative_slope,
                                         nonlinearity=self.nonlinearity)
                if self.nonlinearity == 'leaky_relu' or self.nonlinearity == 'relu':
                    nn.init.uniform_(l.bias, 0, 0.1)
                else:
                    nn.init.constant_(l.bias, 0.0)
            elif l.__class__.__name__ == 'BatchNorm1d':
                l.reset_parameters()
    
    def forward(self, input):
        for l in self.fcs:
            input = l(input)
        return input

class GINLayer(nn.Module):
    def __init__(self, mlp, eps=0.0, train_eps=True, residual=True):
        super(GINLayer, self).__init__()
        self.mlp = mlp
        self.initial_eps = eps
        self.residual = residual
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.mlp.reset_parameters()
        self.eps.data.fill_(self.initial_eps)
    
    def forward(self, input, adj):
        res = input
        
        # Aggregating neighborhood information
        neighs = torch.matmul(adj, res)

        # Reweighting the center node representation
        res = (1 + self.eps) * res + neighs
        
        # Updating node representations
        res = self.mlp(res)
        
        # Residual connection
        if self.residual:
            output = res + input
        else:
            output = res
        
        return output
    
class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
    
class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.2, alpha=0.2, nheads=8):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        # return F.log_softmax(x, dim=1)
        return x
    
class GIN(nn.Module):
    def __init__(self, num_layers, in_features, out_features, hidden_features=[],
                 eps=0.0, train_eps=True, residual=True, batchNorm=True,
                 nonlinearity='leaky_relu', negative_slope=0.1):
        super(GIN, self).__init__()
        
        self.GINLayers = nn.ModuleList()
        
        if in_features != out_features:
            first_layer_res = False
        else:
            first_layer_res = True
        self.GINLayers.append(GINLayer(MLP(in_features, out_features, hidden_features, batchNorm,
                                           nonlinearity, negative_slope),
                                       eps, train_eps, first_layer_res))
        for i in range(num_layers-1):
            self.GINLayers.append(GINLayer(MLP(out_features, out_features, hidden_features, batchNorm,
                                                                nonlinearity, negative_slope),
                                           eps, train_eps, residual))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for l in self.GINLayers:
            l.reset_parameters()
    
    def forward(self, input, adj):
        for l in self.GINLayers:
            input = l(input, adj)
        return input
class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        
        self.eps = eps
    
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm
def attention(q, k, v, d_k, mask=None, dropout=None):
    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k) #scores shape is [bs heads view view]
    # print('scores',scores.shape)
    if mask is not None:
        mask = mask.float()
        mask = mask.unsqueeze(-1).matmul(mask.unsqueeze(-2))#mask shape is [bs 1 view view]
        # mask = mask.unsqueeze(1) #mask shape is [bs 1 1 view]
        scores = scores.masked_fill(mask == 0, -1e9)
    
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    return output
class FDModel(nn.Module):
    # in_features_x是特征数，是那个六维数组
    def __init__(self, d_list, in_features_y, hidden_features, out_features,beta=0.2,
                 in_layers1=1, out_layers=1, batchNorm=False,
                 nonlinearity='leaky_relu', negative_slope=0.1):
        super(FDModel, self).__init__()
        
        hidden_list = [hidden_features] * (in_layers1-1)

        # 这个hidden_features实际上是定义中的out_features, 所以是512, 上面的hidden_features和out_features都是512
        # in_features_x是特征数，是那个六维数组
        # 100/1000..., 512, []
        self.MLP_list = nn.ModuleList([MLP(dim_pre_view, hidden_features, hidden_list,
                       batchNorm, nonlinearity, negative_slope) for dim_pre_view in d_list])
        

        
        self.MLP_list2 = nn.ModuleList([MLP(hidden_features, out_features, hidden_list,
                       batchNorm, nonlinearity, negative_slope) for dim_pre_view in d_list])
        self.NN2 = nn.Linear(in_features_y, hidden_features)
        self.beta = beta
        assert self.beta >0
        hidden_list = [hidden_features] * (out_layers-1)
        # self.Q = MLP(hidden_features, out_features, hidden_list,
        #                batchNorm, nonlinearity, negative_slope)
        # self.K = MLP(hidden_features, out_features, hidden_list,
        #                batchNorm, nonlinearity, negative_slope)
        # self.V = MLP(hidden_features, out_features, hidden_list,
        #                batchNorm, nonlinearity, negative_slope)
        self.Q = nn.Linear(hidden_features, out_features)
        self.K = nn.Linear(hidden_features, out_features)
        self.V = nn.Linear(hidden_features, out_features)
        self.out = nn.Linear(hidden_features, out_features)
        self.norm = Norm(hidden_features)
        self.reset_parameters()
        
    def reset_parameters(self):
        for module in self.MLP_list:
            module.reset_parameters()
        for module in self.MLP_list2:
            module.reset_parameters()
        nn.init.kaiming_uniform_(self.NN2.weight, nonlinearity='sigmoid')
        nn.init.constant_(self.NN2.bias, 0.0)

    
    def forward(self, x, y, mask=None):
        # x = self.NN1(x) # b1 x h
        
        # 利用循环对x中的每个元素执行NN1操作
        x_processed = []
        for i in range(len(x)):
            x_i = self.MLP_list[i](x[i])
            x_processed.append(x_i)
        x_processed = torch.stack(x_processed,dim=0) # m n d
        # x_processed = self.norm(x_processed)
        d = x_processed.shape[-1]
        # x_q = self.Q(x_processed)
        x_q = x_processed
        # x_q = self.norm(x_q)
        x_q = F.normalize(x_q,dim=-1)
        # x_k = self.K(x_processed)
        # x_v = self.V(x_processed)
        # new_x = attention(x_q.transpose(0,1),x_k.transpose(0,1),x_v.transpose(0,1),d,mask)
        x_att_score = x_q.matmul(x_q.transpose(1,2)) # m n n 
        # x_att_score = torch.exp(x_att_score)
        x_att_score = (x_att_score/self.beta).exp()
        if mask is not None:
            mask_12 = torch.matmul(mask.float().t().unsqueeze(dim=-1),mask.float().t().unsqueeze(dim=1)) # m n n
            
            x_att_score = x_att_score.mul(mask_12)
            # x_att_score = x_att_score.mul((1-(torch.zeros_like(x_att_score)+torch.eye(mask.shape[0],device = mask.device).unsqueeze(0))))
            # x_att_score = x_att_score.sum(dim=0)/(mask_12.sum(dim=0)+1e-9)  #  n n 
            x_att_score = x_att_score.max(dim=0).values
            
            x_att_score = x_att_score.fill_diagonal_(0.)
            # x_att_score = x_att_score.masked_fill(mask_12.sum(dim=0)==0,-1e9)
            # print(x_att_score[:5,:5],x_att_score.max(dim=-1))
            
            # x_att_score = x_att_score.unsqueeze(0)
            # energy = (x_att_score/1.).exp().sum(dim=-1).log()
            # confi = (torch.log(x_att_score+1e-9)*self.beta).max(dim=-1).values
            confi = (torch.log(x_att_score+1e-9)*self.beta).unsqueeze(0).mul(mask.t().unsqueeze(1))
            confi = confi.max(dim=-1).values
            # print(confi.shape)
            # confi = confi.sum(dim=-1)/(mask.t().sum(dim=-1,keepdim=True)+1e-9)
            # x_att_score = F.softmax(x_att_score,dim=-1).detach()
            x_att_score = x_att_score/(x_att_score.sum(dim=-1,keepdim=True)+1e-9)
            x_att_score = x_att_score.unsqueeze(0).mul(mask.t().unsqueeze(1)) # m n n 
        new_x = x_att_score.matmul(x_processed)
        # new_x[mask.t()==0,:] = new_x[mask.t()==0,:]
        # new_x = self.out(new_x)
        new_x = new_x*(1-mask.transpose(0,1).unsqueeze(dim=-1)) * 1  + x_processed*mask.transpose(0,1).unsqueeze(dim=-1)
        # print(new_x.shape)
        y_n = self.NN2(y).sigmoid_() # b2 x h

        # y_n = self.NN2(y) # b2 x h
        Z = []
        for i in range(new_x.shape[0]):
            z_i = new_x[i].unsqueeze(1) * y_n.unsqueeze(0)
            # z_i = self.MLP_list2[i](x_processed_i.unsqueeze(1) * y_n.unsqueeze(0))
            Z.append(z_i)
        return Z, torch.clamp(confi,min=0.,max=1.)
        # return Z, torch.clamp(confi,min=1,max=0)
        # return Z, confi