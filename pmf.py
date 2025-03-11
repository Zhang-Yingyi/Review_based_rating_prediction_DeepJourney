import torch
from torch import nn
import torch.nn.functional as F 

class PMF(nn.Module):
    def __init__(self, config ,user_num, item_num):
        super().__init__()
        self.lam_u = config.lam_u
        self.lam_v = config.lam_v
        self.user_embedding = nn.Embedding(user_num, config.id_embd_num)
        self.item_embedding = nn.Embedding(item_num, config.id_embd_num)

        self.b_users = nn.Parameter(torch.randn(user_num, 1))
        self.b_items = nn.Parameter(torch.randn(item_num, 1))
        self.b_0 = nn.Parameter(torch.randn(1))
        self.init_weight()

    def init_weight(self):
        nn.init.uniform_(self.b_users, a=0, b=0.1)
        nn.init.uniform_(self.b_items, a=0, b=0.1)

    
    def forward(self, user_review, item_review, uid, iid):
    
        u_id_embd = self.user_embedding(uid) # [bs,16]
        i_id_embd = self.item_embedding(iid) # [bs,16]
        
        # rating_matrix = get_ratingMatrix(uid, iid)
        
        # non_zero_mask = (rating_matrix != -1).type(torch.FloatTensor)
        # predicted = torch.sigmoid(torch.mm(u_id_embd, i_id_embd.t()))
        # predicted = torch.diag(predicted)

        predicted = torch.sum(u_id_embd * i_id_embd, dim = 1, keepdim = True) + self.b_users[uid] + self.b_items[iid] + self.b_0 #[bs,1]
        # predicted = torch.sum(predicted, dim=1, keepdim=True)
       
        # diff = (rating_matrix - predicted)**2
        # prediction_error = torch.sum(diff*non_zero_mask)

        # u_regularization = self.lam_u * torch.sum(u_id_embd.norm(dim=1))
        # v_regularization = self.lam_v * torch.sum(i_id_embd.norm(dim=1))
        
        return predicted.squeeze()
