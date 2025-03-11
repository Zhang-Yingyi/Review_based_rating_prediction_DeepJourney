import torch
from torch import nn
import torch.nn.functional as F

class ANR(nn.Module):

    def __init__(self, config, word_emb,user_num, item_num):
        super(ANR, self).__init__()
        self.aspect_num = config.aspect_num
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(word_emb))
        self.user_aspect_a1 = nn.Linear(config.word_embd_dim,config.aspect_hidden_size,bias = False)
        self.user_aspect_a2 = nn.Linear(config.word_embd_dim,config.aspect_hidden_size,bias = False)
        self.user_aspect_a3 = nn.Linear(config.word_embd_dim,config.aspect_hidden_size,bias = False)
        self.user_aspect_a4 = nn.Linear(config.word_embd_dim,config.aspect_hidden_size,bias = False)
        self.user_aspect_a5 = nn.Linear(config.word_embd_dim,config.aspect_hidden_size,bias = False)

        self.item_aspect_a1 = nn.Linear(config.word_embd_dim,config.aspect_hidden_size,bias = False)
        self.item_aspect_a2 = nn.Linear(config.word_embd_dim,config.aspect_hidden_size,bias = False)
        self.item_aspect_a3 = nn.Linear(config.word_embd_dim,config.aspect_hidden_size,bias = False)
        self.item_aspect_a4 = nn.Linear(config.word_embd_dim,config.aspect_hidden_size,bias = False)
        self.item_aspect_a5 = nn.Linear(config.word_embd_dim,config.aspect_hidden_size,bias = False)

        self.window_size = config.window_size
        self.aspect_vec = nn.Parameter(torch.randn(5, config.window_size * config.aspect_hidden_size, 1))
        self.W_s = nn.Parameter(torch.randn(config.window_size *config.aspect_hidden_size, config.window_size *config.aspect_hidden_size))
        self.W_x = nn.Parameter(torch.randn(config.window_size *config.aspect_hidden_size, config.affinity_hidden_size))
        self.W_y = nn.Parameter(torch.randn(config.window_size *config.aspect_hidden_size, config.affinity_hidden_size))
        self.v_x = nn.Parameter(torch.randn(config.affinity_hidden_size,1))
        self.v_y = nn.Parameter(torch.randn(config.affinity_hidden_size,1))

        self.b_users = nn.Parameter(torch.randn(user_num, 1))
        self.b_items = nn.Parameter(torch.randn(item_num, 1))
        self.b_0 = nn.Parameter(torch.randn(1))
        self.init_weight()

    def init_weight(self):
        nn.init.uniform_(self.b_users, a=0, b=0.1)
        nn.init.uniform_(self.b_items, a=0, b=0.1)

    def forward(self, user_review, item_review,uid,iid):  # input shape(batch_size, review_count, review_length)
        # new_batch_size = user_review.shape[0] * user_review.shape[1]
        # user_review = user_review.reshape(new_batch_size, -1)
        # item_review = item_review.reshape(new_batch_size, -1)
        

        M_u = self.embedding(user_review) # [bs,300,300]
        M_i = self.embedding(item_review) # [bs,300,300]

        # user level
        M_u_a1 = self.user_aspect_a1(M_u)
        M_u_a2 = self.user_aspect_a2(M_u)
        M_u_a3 = self.user_aspect_a3(M_u)
        M_u_a4 = self.user_aspect_a4(M_u)
        M_u_a5 = self.user_aspect_a5(M_u)

        z_u_a1 = M_u_a1.unfold(1,self.window_size,1).permute(0,1,3,2) #[bs,word_len-window_size+1,window_size,aspect_hidden_size]
        z_u_a1 = z_u_a1.reshape(-1,z_u_a1.shape[1],z_u_a1.shape[2]*z_u_a1.shape[3]) #[bs,word_len-window_size+1,window_size*aspect_hidden_size]
        att_u_a1 = F.softmax(torch.matmul(z_u_a1,self.aspect_vec[0]),dim = 1) #[bs,word_len-window_size+1,1]
        p_u_a1 = torch.matmul(att_u_a1.permute(0,2,1),z_u_a1)

        z_u_a2 = M_u_a2.unfold(1,self.window_size,1).permute(0,1,3,2)
        z_u_a2 = z_u_a2.reshape(-1,z_u_a2.shape[1],z_u_a2.shape[2]*z_u_a2.shape[3])
        att_u_a2 = F.softmax(torch.matmul(z_u_a2,self.aspect_vec[1]),dim = 1) 
        p_u_a2 = torch.matmul(att_u_a2.permute(0,2,1),z_u_a2)

        z_u_a3 = M_u_a3.unfold(1,self.window_size,1).permute(0,1,3,2)
        z_u_a3 = z_u_a3.reshape(-1,z_u_a3.shape[1],z_u_a3.shape[2]*z_u_a3.shape[3])
        att_u_a3 = F.softmax(torch.matmul(z_u_a3,self.aspect_vec[2]),dim = 1) 
        p_u_a3 = torch.matmul(att_u_a3.permute(0,2,1),z_u_a3)

        z_u_a4 = M_u_a4.unfold(1,self.window_size,1).permute(0,1,3,2)
        z_u_a4 = z_u_a4.reshape(-1,z_u_a4.shape[1],z_u_a4.shape[2]*z_u_a4.shape[3])
        att_u_a4 = F.softmax(torch.matmul(z_u_a4,self.aspect_vec[3]),dim = 1) 
        p_u_a4 = torch.matmul(att_u_a4.permute(0,2,1),z_u_a4)

        z_u_a5 = M_u_a5.unfold(1,self.window_size,1).permute(0,1,3,2)
        z_u_a5 = z_u_a5.reshape(-1,z_u_a5.shape[1],z_u_a5.shape[2]*z_u_a5.shape[3])
        att_u_a5 = F.softmax(torch.matmul(z_u_a5,self.aspect_vec[4]),dim = 1) 
        p_u_a5 = torch.matmul(att_u_a5.permute(0,2,1),z_u_a5)

        P_u = torch.cat([p_u_a1,p_u_a2,p_u_a3,p_u_a4,p_u_a5],dim = 1) # [bs, K, window_size*aspect_hidden_size]

        # item level
        M_i_a1 = self.item_aspect_a1(M_i)
        M_i_a2 = self.item_aspect_a2(M_i)
        M_i_a3 = self.item_aspect_a3(M_i)
        M_i_a4 = self.item_aspect_a4(M_i)
        M_i_a5 = self.item_aspect_a5(M_i)

        z_i_a1 = M_i_a1.unfold(1,self.window_size,1).permute(0,1,3,2) #[bs,word_len-window_size+1,window_size,word_embd_dim]
        z_i_a1 = z_i_a1.reshape(-1,z_i_a1.shape[1],z_i_a1.shape[2]*z_i_a1.shape[3]) #[bs,word_len-window_size+1,window_size*word_embd_dim]
        att_i_a1 = F.softmax(torch.matmul(z_i_a1,self.aspect_vec[0]),dim = 1)  #[bs,word_len-window_size+1,1]
        p_i_a1 =torch.matmul(att_i_a1.permute(0,2,1),z_i_a1)#[bs,1, window_size*aspect_hidden_size]

        z_i_a2 = M_i_a2.unfold(1,self.window_size,1).permute(0,1,3,2)
        z_i_a2 = z_i_a2.reshape(-1,z_i_a2.shape[1],z_i_a2.shape[2]*z_i_a2.shape[3])
        att_i_a2 = F.softmax(torch.matmul(z_i_a2,self.aspect_vec[1]),dim = 1) 
        p_i_a2 = torch.matmul(att_i_a2.permute(0,2,1),z_i_a2)

        z_i_a3 = M_i_a3.unfold(1,self.window_size,1).permute(0,1,3,2)
        z_i_a3 = z_i_a3.reshape(-1,z_i_a3.shape[1],z_i_a3.shape[2]*z_i_a3.shape[3])
        att_i_a3 = F.softmax(torch.matmul(z_i_a3,self.aspect_vec[2]),dim = 1) 
        p_i_a3 = torch.matmul(att_i_a3.permute(0,2,1),z_i_a3)

        z_i_a4 = M_i_a4.unfold(1,self.window_size,1).permute(0,1,3,2)
        z_i_a4 = z_i_a4.reshape(-1,z_i_a4.shape[1],z_i_a4.shape[2]*z_i_a4.shape[3])
        att_i_a4 = F.softmax(torch.matmul(z_i_a4,self.aspect_vec[3]),dim = 1) 
        p_i_a4 = torch.matmul(att_i_a4.permute(0,2,1),z_i_a4)

        z_i_a5 = M_i_a5.unfold(1,self.window_size,1).permute(0,1,3,2)
        z_i_a5 = z_i_a5.reshape(-1,z_i_a5.shape[1],z_i_a5.shape[2]*z_i_a5.shape[3])
        att_i_a5 = F.softmax(torch.matmul(z_i_a5,self.aspect_vec[4]),dim = 1) 
        p_i_a5 = torch.matmul(att_i_a5.permute(0,2,1),z_i_a5)

        Q_i = torch.cat([p_i_a1,p_i_a2,p_i_a3,p_i_a4,p_i_a5],dim = 1) # [bs, K, window_size*aspect_hidden_size]

        # Affinity matrix
        S = F.relu(torch.matmul(torch.matmul(P_u,self.W_s),Q_i.permute(0,2,1)))

        # H_u, H_i

        H_u = F.relu(torch.matmul(P_u,self.W_x) + torch.matmul(S.permute(0,2,1),torch.matmul(Q_i,self.W_y)))#[bs,K,h2]
        H_i = F.relu(torch.matmul(Q_i,self.W_y) + torch.matmul(S,torch.matmul(P_u,self.W_x) )) 
        beta_u = F.softmax(torch.matmul(H_u,self.v_x)).view(-1,self.aspect_num) #[bs,K]
        beta_i = F.softmax(torch.matmul(H_i,self.v_y)).view(-1,self.aspect_num)

        pred_a = torch.cat(
            [
                torch.matmul(p_u_a1,p_i_a1.permute(0,2,1)).view(-1, 1), #[bs,1]
                torch.matmul(p_u_a2,p_i_a2.permute(0,2,1)).view(-1, 1),
                torch.matmul(p_u_a3,p_i_a3.permute(0,2,1)).view(-1, 1),
                torch.matmul(p_u_a4,p_i_a4.permute(0,2,1)).view(-1, 1),
                torch.matmul(p_u_a5,p_i_a5.permute(0,2,1)).view(-1, 1),
            ], dim=1
        )#[bs,k]
        # print('beta_u:',beta_u.shape,'beta_i',beta_i.shape,'pred_a',pred_a.shape)
        prediction = torch.sum(beta_u * beta_i * pred_a,dim = 1, keepdim = True)+ self.b_users[uid]+self.b_items[iid]+ self.b_0 #[bs,1]
        # print(prediction.shape)
        return prediction.view(-1)
