import torch
from torch import nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        #d_model是每个词embedding后的维度
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term2 = torch.pow(torch.tensor(10000.0),torch.arange(0, d_model, 2).float()/d_model)
        div_term1 = torch.pow(torch.tensor(10000.0),torch.arange(1, d_model, 2).float()/d_model)
        #高级切片方式，即从0开始，两个步长取一个。即奇数和偶数位置赋值不一样。直观来看就是每一句话的
        pe[:, 0::2] = torch.sin(position * div_term2)
        pe[:, 1::2] = torch.cos(position * div_term1)
        #这里是为了与x的维度保持一致，释放了一个维度
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        pe = self.pe.tile(x.size(0),1,1)
        x = x + pe
        return x

class PositionalMultiHeadAttn(nn.Module):
    def __init__(self,config):
        super(PositionalMultiHeadAttn, self).__init__()
        self.attn_Q = nn.Linear(config.word_embd_dim, config.AHAG_attn_dim * config.AHAG_attn_head_num)
        self.attn_K = nn.Linear(config.word_embd_dim, config.AHAG_attn_dim * config.AHAG_attn_head_num)
        self.attn_V = nn.Linear(config.word_embd_dim, config.AHAG_attn_dim * config.AHAG_attn_head_num)
        self.attn_dim = config.AHAG_attn_dim
        self.head_num = config.AHAG_attn_head_num
        self.attn_final = nn.Linear(config.AHAG_attn_dim * config.AHAG_attn_head_num, config.word_embd_dim)
        self.dropout = nn.Dropout(config.dropout_prob)
        self.layer_norm = nn.LayerNorm(config.word_embd_dim)
        self.positional_encoding = PositionalEncoding(d_model = config.word_embd_dim,dropout = config.dropout_prob, max_len = config.word_len)
        self.final_layer_1 = nn.Linear(config.word_embd_dim, config.AHAG_FC_1)
        self.relu = nn.ReLU()
        self.final_layer_2 = nn.Linear(config.AHAG_FC_1, config.AHAG_FC_2)

    def forward(self, vec):
        bs = vec.shape[0]
        vec_q = self.attn_Q(vec) #[bs,300,attn_dim*head]
        vec_k = self.attn_K(vec)
        vec_v = self.attn_V(vec)
        
        vec_q = vec_q.view(bs * self.head_num, -1, self.attn_dim) #[bs*num_head,300,attn_dim]
        vec_k = vec_k.view(bs * self.head_num, -1, self.attn_dim)
        vec_v = vec_v.view(bs * self.head_num, -1, self.attn_dim)
        d = vec_v.shape[-1]

        attn_score = F.softmax(torch.matmul(vec_q,vec_k.permute(0,2,1))/math.sqrt(d)) # [bs*num_head,300,300]
        attn_out = torch.matmul(attn_score,vec_v) # [bs*num_head,300,attn_dim]
        attn_out = attn_out.view(bs, -1 , self.head_num * self.attn_dim) # [bs,300,attn_dim*num_head]
        attn_out = self.attn_final(attn_out) # [bs,300,300]
        attn_out = self.dropout(attn_out)
        attn_out = self.layer_norm(vec + attn_out)
        positional_attn_out = self.positional_encoding(attn_out)
        fc_output = self.relu(self.final_layer_1(positional_attn_out))
        fc_output = self.final_layer_2(fc_output)
        return fc_output

class HighOrderAttn(nn.Module):
    def __init__(self,config):
        super(HighOrderAttn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels = config.word_len,
                out_channels=config.AHAG_conv_num,
                kernel_size=config.AHAG_conv_kernel,
                padding=(config.AHAG_conv_kernel - 1) // 2), 
            nn.ReLU(),
            nn.Conv1d(
                in_channels = config.AHAG_conv_num,
                out_channels=config.word_len,
                kernel_size=config.AHAG_conv_kernel,
                padding=(config.AHAG_conv_kernel - 1) // 2), 
            nn.ReLU(),
            )

    def forward(self, vec):
        conv_vec =  self.conv(vec) 
        out = vec * conv_vec
        return out

class CNN(nn.Module):

    def __init__(self, config, word_dim):
        super(CNN, self).__init__()

        self.kernel_count = config.kernel_count
        self.review_count = config.review_count

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels = config.word_len,
                out_channels=config.AHAG_kernel_count,
                kernel_size=config.AHAG_kernel_size,
                padding=(config.AHAG_kernel_size - 1) // 2),  # out shape(new_batch_size, kernel_count, word_embd_dim)
            nn.Dropout(p=config.dropout_prob))

        self.linear = nn.Sequential(
            nn.Linear(config.AHAG_FC_2, config.id_embd_num),
            nn.Tanh(),
            nn.Dropout(p=config.dropout_prob))

    def forward(self, vec):  # input shape(new_batch_size, review_length, word2vec_dim)
        latent = self.conv(vec)  # out(new_batch_size, kernel_count, word_embd_dim) 
        latent = torch.mean(latent, dim = 1) # [bs,word_embd_dim]
        latent = self.linear(latent)
        return latent  # out shape(batch_size, cnn_out_dim)

class NFM(nn.Module):

    def __init__(self, config, p, k):
        super(NFM, self).__init__()
        self.linear = nn.Linear(config.id_embd_num*2, 1)
        self.v = nn.Parameter(torch.rand(p, k) / 10)
        self.f = nn.Sequential(
                nn.Linear(k, config.AHAG_NFM_MLP_dim[0]),
                nn.Tanh(),
                nn.Linear(config.AHAG_NFM_MLP_dim[0], config.AHAG_NFM_MLP_dim[1])
            )
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):  
        linear_part = self.linear(x)
        inter_part1 = torch.mm(x, self.v) ** 2
        inter_part2 = torch.mm(x ** 2, self.v ** 2)
        pair_interactions = inter_part1 - inter_part2
        out = linear_part + self.f(pair_interactions)
        return out

class AHAG(nn.Module):

    def __init__(self, config, word_emb,user_num, item_num):
        super(AHAG, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(word_emb))
        
        self.user_embedding = nn.Embedding(user_num, config.id_embd_num)
        self.item_embedding = nn.Embedding(item_num, config.id_embd_num)
        # pos self attn layer
        self.user_pos_attn = PositionalMultiHeadAttn(config)
        self.item_pos_attn = PositionalMultiHeadAttn(config)
        # high order attn layer
        self.user_high_attn = HighOrderAttn(config)
        self.item_high_attn = HighOrderAttn(config)
        #co-attn layer
        self.W_u = nn.Linear(config.AHAG_FC_2, config.AHAG_FC_2, bias = False)
        self.W_v = nn.Linear(config.AHAG_FC_2, config.AHAG_FC_2, bias = False)
        self.W_uv = nn.Linear(config.AHAG_FC_2, config.AHAG_FC_2, bias = False)
        self.W_s = nn.Linear(config.AHAG_FC_2 * 3, config.AHAG_co_attn_dim, bias = False)
        self.co_u_pool = nn.MaxPool2d((config.word_len,1))
        self.co_i_pool = nn.MaxPool2d((1,config.word_len))
        # gate layer
        self.gate_u = nn.Linear(config.id_embd_num, config.id_embd_num)
        self.gate_i = nn.Linear(config.id_embd_num, config.id_embd_num)
        self.tanh = nn.Tanh()

        self.cnn_u = CNN(config, word_dim=self.embedding.embedding_dim)
        self.cnn_i = CNN(config, word_dim=self.embedding.embedding_dim)
        # self.wu_p = nn.Linear(config.id_embd_num, config.id_embd_num, bias = False)
        # self.wu_f = nn.Linear(config.id_embd_num, config.id_embd_num, bias = False)
        self.wu_ff = nn.Linear(config.id_embd_num, config.id_embd_num)
        self.wi_ff = nn.Linear(config.id_embd_num, config.id_embd_num)
        self.item_gate_filter = nn.Linear(config.id_embd_num*2, config.id_embd_num)
        # predict layer
        self.nfm = NFM(config, config.id_embd_num * 2, 10)


    def init_weight(self):
        nn.init.uniform_(self.b_users, a=0, b=0.1)
        nn.init.uniform_(self.b_items, a=0, b=0.1)

    def forward(self, user_review, item_review,uid,iid):  # input shape(batch_size, review_count, review_length)
        # new_batch_size = user_review.shape[0] * user_review.shape[1]
        # user_review = user_review.reshape(new_batch_size, -1)
        # item_review = item_review.reshape(new_batch_size, -1)
        u_vec = self.embedding(user_review) # [bs,300,300]
        i_vec = self.embedding(item_review)
        bs,word_num,word_dim = u_vec.shape
        # PositionalEncoding
        u_vec_pos_attn = self.user_pos_attn(u_vec) # [bs,300,50]
        i_vec_pos_attn = self.item_pos_attn(u_vec)
        # HighOrderAttn
        u_vec_high = self.user_high_attn(u_vec_pos_attn) # [bs,300,50]
        i_vec_high = self.item_high_attn(u_vec_pos_attn)

        # co attn
        # The detail not contain in the paper, so we refer to "Bidirectional attention flow for machine comprehension (ICLR 2017)"
        u_co = self.W_u(u_vec_high).unsqueeze(1).tile(1,word_num,1,1) # [bs,1,300,50]->[bs,300,300,50]
        i_co = self.W_v(i_vec_high).unsqueeze(1).tile(1,word_num,1,1) # [bs,1,300,300]->[bs,300,300,50]
        co_ui = self.W_uv(u_co * i_co)# [bs,300,300,50]
        co_vec = torch.cat([u_co,i_co,co_ui], dim = 3) # [bs,300,300,150]
        A = self.W_s(co_vec).squeeze(3)# [bs,300,300,1]->[bs,300,300]
        g_u = self.co_u_pool(A).permute(0,2,1) # [bs,300,300]->[bs,300]
        g_i = self.co_i_pool(A)
        # print('A',A.shape,'g_u',g_u.shape,'g_i',g_i.shape,'u_vec_high',u_vec_high.shape)
        t_u = g_u * u_vec_high # [bs,300,50]
        t_i = g_i * i_vec_high
        user_latent = self.cnn_u(t_u) # [bs,16]
        item_latent = self.cnn_i(t_i)

        # id embd
        u_id_embd = self.user_embedding(uid) # [bs,16]
        i_id_embd = self.item_embedding(iid)

        #gate fusion
        su  = self.tanh(self.gate_u(user_latent + u_id_embd))
        su_f = su * user_latent + (1 - su) * u_id_embd
        si  = self.tanh(self.gate_u(item_latent + i_id_embd))
        si_f = si * item_latent + (1 - si) * i_id_embd
        su_ff = self.tanh(self.wu_ff(su_f))
        u_u = u_id_embd + su_ff
        si_ff = self.tanh(self.wi_ff(si_f))
        v_i = i_id_embd + si_ff

        # item filter
        v_i = v_i * self.tanh(self.item_gate_filter(torch.cat([u_u,v_i], dim = 1)))
        #final
        z = torch.cat([u_u,v_i], dim = 1)
        prediction = self.nfm(z)

        return prediction.view(-1)
