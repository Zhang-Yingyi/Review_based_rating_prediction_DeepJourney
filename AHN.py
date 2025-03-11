import torch
from torch import nn
import torch.nn.functional as F
class FactorizationMachine(nn.Module):

    def __init__(self, p, k):  # p=cnn_out_dim
        super().__init__()
        self.v = nn.Parameter(torch.rand(p, k) / 10)
        self.linear = nn.Linear(p, 1, bias=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        linear_part = self.linear(x)  # input shape(batch_size, cnn_out_dim), out shape(batch_size, 1)
        inter_part1 = torch.mm(x, self.v) ** 2
        inter_part2 = torch.mm(x ** 2, self.v ** 2)
        pair_interactions = torch.sum(inter_part1 - inter_part2, dim=1, keepdim=True)
        pair_interactions = self.dropout(pair_interactions)
        output = linear_part + 0.5 * pair_interactions
        return output  # out shape(batch_size, 1)


class AHN(nn.Module):
    def __init__(self, config, word_emb,user_num, item_num):
        super(AHN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(word_emb))
        self.bi_lstm_u_word = nn.LSTM(config.word_embd_dim,
                                config.AHN_bilstm_hidden_size,
                                num_layers=1,
                                bidirectional=True,
                                batch_first=True)
        self.bi_lstm_u_senten = nn.LSTM(config.AHN_bilstm_hidden_size*2,
                                config.AHN_bilstm_hidden_size,
                                num_layers=1,
                                bidirectional=True,
                                batch_first=True)
        self.bi_lstm_i_word = nn.LSTM(config.word_embd_dim,
                                config.AHN_bilstm_hidden_size,
                                num_layers=1,
                                bidirectional=True,
                                batch_first=True)
        self.bi_lstm_i_senten = nn.LSTM(config.AHN_bilstm_hidden_size*2,
                                config.AHN_bilstm_hidden_size,
                                num_layers=1,
                                bidirectional=True,
                                batch_first=True)
        self.att_linear_i_Q = nn.Linear(config.AHN_bilstm_hidden_size*2, config.AHN_item_attn_dim,bias=False)
        self.att_linear_i_K = nn.Linear(config.AHN_bilstm_hidden_size*2, config.AHN_item_attn_dim,bias=False)
        self.att_i_V = nn.Parameter(torch.randn(config.AHN_item_attn_dim,1)) # [50,1]

        self.co_att_linear_u = nn.Linear(config.AHN_bilstm_hidden_size*2, config.AHN_co_attn_dim)
        self.co_att_linear_i = nn.Linear(config.AHN_bilstm_hidden_size*2, config.AHN_co_attn_dim)
        self.M_s = nn.Parameter(torch.randn(config.AHN_dim_M,config.AHN_dim_M)) # [50,50]

        self.att_linear_i_Q_r = nn.Linear(config.AHN_bilstm_hidden_size*2, config.AHN_item_attn_dim,bias=False)
        self.att_linear_i_K_r = nn.Linear(config.AHN_bilstm_hidden_size*2, config.AHN_item_attn_dim,bias=False)
        self.att_i_V_r = nn.Parameter(torch.randn(config.AHN_item_attn_dim,1)) # [50,1]

        self.co_att_linear_u_r = nn.Linear(config.AHN_bilstm_hidden_size*2, config.AHN_co_attn_dim)
        self.co_att_linear_i_r = nn.Linear(config.AHN_bilstm_hidden_size*2, config.AHN_co_attn_dim)
        self.M_r = nn.Parameter(torch.randn(config.AHN_dim_M,config.AHN_dim_M)) # [50,50]


        self.user_embedding = nn.Embedding(user_num, config.id_embd_num)
        self.item_embedding = nn.Embedding(item_num, config.id_embd_num)

        self.fm = FactorizationMachine(config.AHN_bilstm_hidden_size * 2 * 2 + config.id_embd_num * 2, 10)

    # def forward(self, user_review, item_review,uid,iid, u_iid_seq, i_uid_seq):  # input shape(batch_size, review_count, review_length)
    def forward(self, user_review, item_review,uid,iid,u_iid_seq, i_uid_seq, has_img, user_mean_rating,food_label,service_label,price_label,ambience_label,price_differ,cate_differ,location_differ,hist_post_img):  # input shape(batch_size, review_count, review_length)
        # new_batch_size = user_review.shape[0] * user_review.shape[1]
        # user_review = user_review.reshape(new_batch_size, -1)
        # item_review = item_review.reshape(new_batch_size, -1)
        u_vec = self.embedding(user_review) # [bs,10,50,300]
        i_vec = self.embedding(item_review)
        bs,seq_num,word_num,embd_dim = u_vec.shape

        # user review encoding
        u_vec_s = u_vec.view(bs * seq_num,word_num,embd_dim)  # [bs*10,50,300]
        u_vec_s,(u_vec_hn,u_vec_cn) = self.bi_lstm_u_word(u_vec_s) #[bs*10,50,2*150]
        u_vec_s = F.max_pool2d(u_vec_s,(word_num,1)) #[bs*10,1,2*50]
        LSTM_dim = u_vec_s.shape[-1]
        u_vec_s = u_vec_s.view(bs,seq_num,LSTM_dim) #[bs,10,2*50]
        u_s = u_vec_s
        # u_s,(u_s_hn,u_s_cn) = self.bi_lstm_u_senten(u_vec_s) #[bs,10,2*50]
        # item review encoding
        i_vec_s = i_vec.view(bs * seq_num,word_num,embd_dim)
        i_vec_s,(i_vec_hn,i_vec_cn) = self.bi_lstm_i_word(i_vec_s)
        i_vec_s = F.max_pool2d(i_vec_s,(word_num,1))
        i_vec_s = i_vec_s.view(bs,seq_num,LSTM_dim)
        i_s = i_vec_s
        # i_s,(i_s_hn,i_s_cn) = self.bi_lstm_i_senten(i_vec_s)

        # item review level aggregation
        item_attn_Q = F.tanh(self.att_linear_i_Q(i_s)) #[bs,10,50]
        item_attn_K = F.sigmoid(self.att_linear_i_K(i_s)) #[bs,10,50]
        item_attn_V = torch.matmul(item_attn_Q*item_attn_K,self.att_i_V) #[bs,10,1]
        item_attn_score = F.softmax(item_attn_V,dim = 1) #[bs,10,1]
        item_agg_vec =  item_attn_score * i_s #[bs,10,2*50]

        # user item review level co attention
        co_att_u = self.co_att_linear_u(u_s) #[bs,10,50]
        co_att_i = self.co_att_linear_i(i_s).permute(0,2,1) #[bs,50,10]
        G = F.relu(torch.matmul(torch.matmul(co_att_u,self.M_s),co_att_i)) #[bs,10,10]
        a_u_i = F.max_pool1d(G * item_attn_score,seq_num) #[bs,10,1]
        user_agg_vec = F.softmax(a_u_i,1) * u_s #[bs,10,2*50]
        item_agg_vec_r =  torch.sum(item_agg_vec, dim = 1)
        user_agg_vec_r = torch.sum(user_agg_vec, dim = 1)

        # item review level aggregation
        # item_attn_Q_r = F.tanh(self.att_linear_i_Q_r(item_agg_vec)) #[bs,10,50]
        # item_attn_K_r = F.sigmoid(self.att_linear_i_K_r(i_s)) #[bs,10,50]
        # item_attn_V_r = torch.matmul(item_attn_Q_r*item_attn_K_r,self.att_i_V_r) #[bs,10,1]
        # item_attn_score_r = F.softmax(item_attn_V_r,dim = 1) #[bs,10,1]
        # item_agg_vec_r =  torch.sum(item_attn_score_r * i_s, dim = 1) #[bs,2*50]

        # user item review level co attention
        # co_att_u_r = self.co_att_linear_u_r(user_agg_vec) #[bs,10,50]
        # co_att_i_r = self.co_att_linear_i_r(item_agg_vec).permute(0,2,1) #[bs,50,10]
        # G = F.relu(torch.matmul(torch.matmul(co_att_u_r,self.M_r),co_att_i_r)) #[bs,10,10]
        # a_u_i = F.max_pool1d(G * item_attn_score,seq_num) #[bs,10,1]
        # user_agg_vec_r = torch.sum(F.softmax(a_u_i,1) * u_s, dim = 1) #[bs,2*50]

        # id embedding
        u_id_embd = self.user_embedding(uid)
        i_id_embd = self.item_embedding(iid)
        u_out = torch.cat([u_id_embd,user_agg_vec_r],dim = 1)#[bs,2*50+16]
        i_out = torch.cat([i_id_embd,item_agg_vec_r],dim = 1)#[bs,2*50+16]

        concat_latent = torch.cat([u_out,i_out],dim = 1)
        prediction = self.fm(concat_latent)
        prediction = torch.sum(prediction,dim = 1).float()
        return prediction.view(-1)
