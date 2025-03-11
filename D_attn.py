import torch
from torch import nn
import torch.nn.functional as F

class Dattn(nn.Module):

    def __init__(self, config, word_emb,user_num, item_num):
        super(Dattn, self).__init__()
        self.window_size = config.d_attn_window_size
        self.word_len = config.word_len
        self.word_embd_dim = config.word_embd_dim
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(word_emb))
        # l-attn u
        self.l_attn_weight_u = nn.Parameter(torch.randn(config.word_len, config.d_attn_window_size))
        self.l_attn_bias_u = nn.Parameter(torch.randn(config.word_len))
        self.L_attn_u_conv = nn.Sequential(
            nn.Conv2d(
                    in_channels = 1,
                    out_channels = config.l_attn_kernal_size,
                    kernel_size= config.l_attn_conv_kernal_size,
                    padding=[(config.l_attn_conv_kernal_size[0] - 1) // 2,0]
                    ),  # out shape (batch_size,narre_kernel_num, review_words_num, word2vec_dim-1)
                nn.Tanh(),
                # nn.MaxPool2d(kernel_size=(config.word_len,self.word_embd_dim )),
            )
        # g-attn u
        self.g_attn_weight_u = nn.Parameter(torch.randn(config.word_len, config.word_embd_dim))
        self.g_attn_bias_u = nn.Parameter(torch.randn(config.word_len))
        self.G_attn_u_conv = nn.ModuleList([nn.Conv2d(1, config.g_attn_kernal_size, (k, config.word_embd_dim)) for k in config.g_attn_conv_kernal_size])
        # out layer u
        self.out_u_1 = nn.Sequential(
                nn.Linear(config.word_len + config.g_attn_kernal_size * len(config.g_attn_conv_kernal_size), config.d_attn_fc_1),
                nn.ReLU(),
            )
        self.out_u_2 = nn.Sequential(
                nn.Linear(config.d_attn_fc_1, config.d_attn_fc_2),
                nn.ReLU(),
            )

        # l-attn i
        self.l_attn_weight_i = nn.Parameter(torch.randn(config.word_len, config.d_attn_window_size))
        self.l_attn_bias_i = nn.Parameter(torch.randn(config.word_len))
        self.L_attn_i_conv = nn.Sequential(
            nn.Conv2d(
                    in_channels = 1,
                    out_channels = config.l_attn_kernal_size,
                    kernel_size= config.l_attn_conv_kernal_size,
                    padding=[(config.l_attn_conv_kernal_size[0] - 1) // 2,0]
                    ),  # out shape (batch_size,narre_kernel_num, review_words_num, word2vec_dim-1)
                nn.Tanh(),
                # nn.MaxPool2d(kernel_size=(config.word_len,self.word_embd_dim )),
            )
        # g-attn i
        self.g_attn_weight_i = nn.Parameter(torch.randn(config.word_len, config.word_embd_dim))
        self.g_attn_bias_i = nn.Parameter(torch.randn(config.word_len))
        self.G_attn_i_conv = nn.ModuleList([nn.Conv2d(1, config.g_attn_kernal_size, (k, config.word_embd_dim)) for k in config.g_attn_conv_kernal_size])
        # out layer i
        self.out_i_1 = nn.Sequential(
                nn.Linear(config.word_len + config.g_attn_kernal_size*len(config.g_attn_conv_kernal_size), config.d_attn_fc_1),
                nn.ReLU(),
            )
        self.out_i_2 = nn.Sequential(
                nn.Linear(config.d_attn_fc_1, config.d_attn_fc_2),
                nn.ReLU(),
            )

    def forward(self, user_review, item_review,uid,iid):  # input shape(batch_size, review_count, review_length)
        # new_batch_size = user_review.shape[0] * user_review.shape[1]
        # user_review = user_review.reshape(new_batch_size, -1)
        # item_review = item_review.reshape(new_batch_size, -1)
        

        u_vec = self.embedding(user_review) #[bs,300,300]
        i_vec = self.embedding(item_review)

        # u l-attn
        pad_u_vec = F.pad(u_vec,(0,0,self.window_size//2,self.window_size//2)) #[bs,304,300]
        l_att_u = pad_u_vec.unfold(1,self.window_size,1) #[bs,300,300,5]
        s_i_u = torch.sum(self.l_attn_weight_u * l_att_u, dim = [2,3]) + self.l_attn_bias_u # [bs,300]
        s_i_u = F.sigmoid(s_i_u).unsqueeze(dim = 1) #[bs,300,1]
        u_vec_L = s_i_u * u_vec #[bs,300,300]
        u_vec_L_score = self.L_attn_u_conv(u_vec_L.unsqueeze(dim = 1)).squeeze(1) #[bs,1,1]
        u_vec_L = u_vec_L.mul(u_vec_L_score) #[bs,300,300]
        u_vec_L = F.max_pool1d(u_vec_L, u_vec_L.size(2)).squeeze(2)  #[bs,300]
        # u g-attn
        g_att_u = torch.sum(self.g_attn_weight_u * u_vec, dim = [2]) + self.g_attn_bias_u # [bs,300]
        s_i_u_G = F.sigmoid(g_att_u).unsqueeze(dim = 1) # [bs,300,1]
        u_vec_G = s_i_u_G * u_vec #[bs,300,300]
        u_vec_G = [torch.tanh(cnn(u_vec_G.unsqueeze(1))).squeeze(3) for cnn in self.G_attn_u_conv]#[[bs,10,299]*n]
        u_vec_G = [F.max_pool1d(out,out.shape[2]).squeeze(2) for out in u_vec_G]#[[bs,10]*n]
        # u out layer
        u_out = torch.cat([u_vec_L]+u_vec_G, dim = 1) #[bs,330]
        # print('u_out',u_out.shape)
        u_out = self.out_u_1(u_out)#[bs,500]
        u_out = self.out_u_2(u_out)#[bs,50]



        # i l-attn
        pad_i_vec = F.pad(i_vec,(0,0,self.window_size//2,self.window_size//2)) #[bs,304,300]
        l_att_i = pad_i_vec.unfold(1,self.window_size,1) #[bs,300,300,5]
        s_i_i = torch.sum(self.l_attn_weight_i * l_att_i, dim = [2,3]) + self.l_attn_bias_i # [bs,300]
        s_i_i = F.sigmoid(s_i_i).unsqueeze(dim = 1) #[bs,300,1]
        i_vec_L = s_i_i * i_vec #[bs,300,300]
        i_vec_L_score = self.L_attn_i_conv(i_vec_L.unsqueeze(dim = 1)).squeeze(1) #[bs,200,1,1]
        i_vec_L = i_vec_L.mul(i_vec_L_score) #[bs,300,300]
        i_vec_L = F.max_pool1d(i_vec_L, i_vec_L.size(2)).squeeze(2)  #[bs,300]
        # i g-attn
        g_att_i = torch.sum(self.g_attn_weight_i * i_vec, dim = [2]) + self.g_attn_bias_u # [bs,300]
        s_i_i_G = F.sigmoid(g_att_i).unsqueeze(dim = 1) # [bs,300,1]
        i_vec_G = s_i_i_G * i_vec #[bs,300,300]
        i_vec_G = [torch.tanh(cnn(i_vec_G.unsqueeze(1))).squeeze(3) for cnn in self.G_attn_i_conv]#[[bs,100,300,300]*n]
        i_vec_G = [F.max_pool1d(out,out.shape[2]).squeeze(2) for out in i_vec_G]#[[bs,10]*n]
        # i out layer
        i_out = torch.cat([i_vec_L]+i_vec_G, dim = 1) #[bs,330]
        i_out = self.out_i_1(i_out)#[bs,500]
        i_out = self.out_i_2(i_out)#[bs,50]

        prediction = torch.sum(u_out * i_out,dim = 1).float()
        return prediction
