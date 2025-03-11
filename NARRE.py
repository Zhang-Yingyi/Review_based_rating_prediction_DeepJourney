import torch
from torch import nn
import torch.nn.functional as F

class CNN2D(nn.Module):

    def __init__(self, config, word_dim):
        super(CNN2D, self).__init__()
        self.review_num = config.review_num
        self.narre_kernel_num = config.narre_kernel_num
        self.narre_hidden_dim = config.narre_hidden_dim
        self.conv = nn.Sequential(
                nn.Conv2d(
                    in_channels = config.review_num,
                    out_channels = config.narre_kernel_num,
                    kernel_size=config.narre_kernel_size,
                    groups= config.review_num,
                    padding=[(config.narre_kernel_size[0] - 1) // 2,(config.narre_kernel_size[1] - 1) // 2]
                    ),  # out shape (batch_size,narre_kernel_num, review_words_num, word2vec_dim-1)
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(config.review_word_num,1)),
                nn.Dropout(p=config.dropout_prob)
            )
        self.linear = nn.Sequential(
            nn.Linear(config.narre_cnn_out_dim, config.narre_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=config.dropout_prob))
    def forward(self, vec):  # input shape(batch_size, review_num, review_words_num, word2vec_dim)
        latent = self.conv(vec) # out shape (batch_size,narre_kernel_num, 1, word2vec_dim-1)
        latent = torch.squeeze(latent, dim = 2) # out shape (batch_size,narre_kernel_num, word2vec_dim-1)
        latent = self.linear(latent)  # out shape (batch_size,narre_kernel_num, hidden_dim)
        latent = latent.reshape(-1,self.review_num,self.narre_kernel_num//self.review_num * self.narre_hidden_dim)
        return latent # (batch_size,review_num ,narre_kernel_num//review_num * hidden_dim)

class NARRE(nn.Module):

    def __init__(self, config, word_emb,user_num, item_num):
        super(NARRE, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(word_emb))
        self.cnn_u = CNN2D(config, word_dim=self.embedding.embedding_dim)
        self.cnn_i = CNN2D(config, word_dim=self.embedding.embedding_dim)
        self.user_embedding = nn.Embedding(user_num, config.id_embd_num)
        self.item_embedding = nn.Embedding(item_num, config.id_embd_num)
        #narre attntion of user side review
        self.W_O_u = nn.Parameter(
            torch.randn(
                config.narre_hidden_dim *config.each_kernal, config.narre_attn_size)
            ) #[256,10]
        self.W_u_linear = nn.Linear(config.id_embd_num, config.narre_attn_size) 
        self.h_u = nn.Parameter(torch.randn(config.narre_attn_size,1)) #[10,1]
        self.b_u_2 = nn.Parameter(torch.randn(1)) #[1]
        self.Y_u_linear = nn.Linear(config.narre_hidden_dim *config.each_kernal, config.narre_final_size) 

        #narre attntion of item side review
        self.W_O_i = nn.Parameter(
            torch.randn(
                config.narre_hidden_dim *config.each_kernal, config.narre_attn_size)
            ) #[256,10]
        self.W_i_linear = nn.Linear(config.id_embd_num, config.narre_attn_size) 
        self.h_i = nn.Parameter(torch.randn(config.narre_attn_size,1)) #[10,1]
        self.b_i_2 = nn.Parameter(torch.randn(1)) #[1]
        self.Y_i_linear = nn.Linear(config.narre_hidden_dim *config.each_kernal, config.narre_final_size) 

        # predict layer
        self.pred_layer = nn.Linear(config.narre_final_size,1,bias = False) 


        self.b_users = nn.Parameter(torch.randn(user_num, 1))
        self.b_items = nn.Parameter(torch.randn(item_num, 1))
        self.b_0 = nn.Parameter(torch.randn(1))
        self.init_weight()

    def init_weight(self):
        nn.init.uniform_(self.b_users, a=0, b=0.1)
        nn.init.uniform_(self.b_items, a=0, b=0.1)

    def forward(self, user_review, item_review, uid,iid, u_iid_seq, i_uid_seq, has_img, user_mean_rating,food_label,service_label,price_label,ambience_label,price_differ,cate_differ,location_differ,hist_post_img):   
        # new_batch_size = user_review.shape[0] * user_review.shape[1]
        # user_review = user_review.reshape(new_batch_size, -1)
        # item_review = item_review.reshape(new_batch_size, -1)
        u_vec = self.embedding(user_review) # [bs,10,50,300]
        i_vec = self.embedding(item_review)

        user_latent = self.cnn_u(u_vec)  # [bs,10,256]
        item_latent = self.cnn_i(i_vec)
        u_iid_seq_embd = self.item_embedding(u_iid_seq) # [bs,10,16] 
        i_uid_seq_embd = self.user_embedding(i_uid_seq)
        #narre att user
        attn_u = torch.matmul(user_latent,self.W_O_u) + self.W_u_linear(u_iid_seq_embd) #[bs,10,10]
        attn_score_u = F.relu(torch.matmul(attn_u,self.h_u).squeeze(2)) + self.b_u_2 #[bs,10]
        attn_score_u = F.softmax(attn_score_u,dim = 1).unsqueeze(1) #[bs,1,10]
        user_latent = torch.matmul(attn_score_u,user_latent).squeeze(1) #[bs,256]
        user_latent = self.Y_u_linear(user_latent) #[bs,16]
        #narre att item
        attn_i = torch.matmul(item_latent,self.W_O_i) + self.W_i_linear(i_uid_seq_embd) #[bs,10,10]
        attn_score_i = F.relu(torch.matmul(attn_i,self.h_i).squeeze(2)) + self.b_i_2 #[bs,10]
        attn_score_i = F.softmax(attn_score_i,dim = 1).unsqueeze(1) #[bs,1,10]
        item_latent = torch.matmul(attn_score_i,item_latent).squeeze(1) #[bs,256]
        item_latent = self.Y_i_linear(item_latent)

        u_id_embd = self.user_embedding(uid) # [bs,16]
        i_id_embd = self.item_embedding(iid)

        h_0 = (u_id_embd + user_latent) * (i_id_embd + item_latent)

        prediction = self.pred_layer(h_0) + self.b_users[uid] + self.b_items[iid] + self.b_0 #[bs,1]
        return prediction.view(-1)
