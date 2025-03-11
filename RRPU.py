import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel

class CNN(nn.Module):
    def __init__(self, config, word_dim):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels = config.review_word_num,
                out_channels=config.RRPU_user_cnn_channel,
                kernel_size=config.RRPU_kernel_size,
                padding=(config.RRPU_kernel_size - 1) // 2),  # out shape(bs*10, 40, 768)
            nn.MaxPool2d(kernel_size=(config.RRPU_user_cnn_channel,1)),
            nn.Dropout(p=config.dropout_prob)
            )

    def forward(self, vec):  # input shape(new_batch_size, review_length, word2vec_dim)
        latent = self.conv(vec)  # out(bs*10, 1, 768) 
        latent = torch.mean(latent, dim = 1) # [bs*10,768]
        return latent

class AspectLevelModel(nn.Module):
    def __init__(self, config):
        super(AspectLevelModel, self).__init__()
        self.window_size = config.window_size
        self.iten_aspect_matrix = nn.Parameter(torch.randn(config.bert_out_dim, config.RRPU_aspect_dim))
        self.aspect_vec = nn.Parameter(torch.randn(config.window_size * config.RRPU_aspect_dim, 1))
    def forward(self,item_vec):
        item_review_aspect = torch.matmul(item_vec,self.iten_aspect_matrix) # [bs,10*50,32]
        item_review_aspect = item_review_aspect.unfold(1,self.window_size,1).permute(0,1,3,2) #[bs,498,3,32]
        bs,word_num,window,hidden_dim = item_review_aspect.shape
        item_review_aspect = item_review_aspect.reshape(bs,word_num,-1)#[bs,498,3*32] 
        item_review_aspect_attn = F.softmax(torch.matmul(item_review_aspect,self.aspect_vec),dim = 1)  # [bs,498,1]
        item_review_aspect_vec = torch.matmul(item_review_aspect_attn.permute(0,2,1),item_review_aspect) #[bs,3*32]
        return item_review_aspect_vec.squeeze(1)

class NFM(nn.Module):

    def __init__(self, config, p, k):
        super(NFM, self).__init__()
        self.final_dim = config.id_embd_num*2 + config.bert_out_dim + config.window_size * config.RRPU_aspect_dim
        self.linear = nn.Linear(self.final_dim, 1)
        self.v = nn.Parameter(torch.rand(self.final_dim, k) / 10)
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

class RRPU(nn.Module):
    def __init__(self, config, word_emb,user_num, item_num):
        super(RRPU, self).__init__()
        self.bert = AutoModel.from_pretrained(config.bert_model_name, local_files_only=True)
        if config.freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.user_embedding = nn.Embedding(user_num, config.id_embd_num)
        self.item_embedding = nn.Embedding(item_num, config.id_embd_num)
        self.cnn_u = CNN(config, word_dim=config.bert_out_dim)
        self.item_aspect_1 = AspectLevelModel(config)
        self.item_aspect_2 = AspectLevelModel(config)
        self.item_aspect_3 = AspectLevelModel(config)
        self.user_hidden_mean_layer = nn.Sequential(
                nn.Linear(config.bert_out_dim, config.RRPU_user_hidden_dims[0]),
                nn.Linear(config.RRPU_user_hidden_dims[0], config.RRPU_user_hidden_dims[1])
            )
        self.user_hidden_sigma_layer = nn.Sequential(
                nn.Linear(config.bert_out_dim, config.RRPU_user_hidden_dims[0]),
                nn.Linear(config.RRPU_user_hidden_dims[0], config.RRPU_user_hidden_dims[1])
            )
        self.w_h = nn.Parameter(torch.randn(config.RRPU_user_hidden_dims[1], config.RRPU_item_attn_dim))
        self.w_y_1 = nn.Parameter(torch.randn(config.window_size * config.RRPU_aspect_dim, config.RRPU_item_attn_dim))
        self.w_y_2 = nn.Parameter(torch.randn(config.window_size * config.RRPU_aspect_dim, config.RRPU_item_attn_dim))
        self.w_y_3 = nn.Parameter(torch.randn(config.window_size * config.RRPU_aspect_dim, config.RRPU_item_attn_dim))

        self.w_a_1 = nn.Parameter(torch.randn(config.RRPU_item_attn_dim, 1))
        self.w_a_2 = nn.Parameter(torch.randn(config.RRPU_item_attn_dim, 1))
        self.w_a_3 = nn.Parameter(torch.randn(config.RRPU_item_attn_dim, 1))

        self.nfm = NFM(config, config.id_embd_num * 2, 10)

    def forward(self, user_review, item_review,uid,iid, u_iid_seq, i_uid_seq):  
        bs,seq_len,word_len = user_review.shape
        user_review = user_review.int().view(bs*seq_len,word_len) #[bs*10,50]
        item_review = item_review.int().view(bs*seq_len,word_len)
        u_iid_seq_embd = self.item_embedding(u_iid_seq) # [bs,10,16] 
        i_uid_seq_embd = self.user_embedding(i_uid_seq)
        u_id_embd = self.user_embedding(uid) # [bs,16]
        i_id_embd = self.item_embedding(iid)
        # user word embedding layer
        user_review = self.bert(user_review)[0] #[bs*10,50,768]
        # user review representation layer
        user_latent = self.cnn_u(user_review).view(bs,seq_len,-1) #[bs,10,768]
        # user review aggregation layer
        attn = torch.matmul(u_iid_seq_embd,i_id_embd.unsqueeze(dim = 2)) #[bs,10,1] 
        attn_score = F.softmax(attn, dim = 1).permute(0,2,1) #[bs,1,10] 
        user_latent = torch.matmul(attn_score, user_latent) #[bs,1,768] 
        user_latent = user_latent.squeeze(dim = 1)#[bs,768] 

        # item word embedding layer
        item_review = self.bert(item_review)[0].view(bs,seq_len*word_len, -1)  #[bs,10*50,768]
        # item aspect level feature learning
        item_aspect_1 = self.item_aspect_1(item_review) #[bs,3*32]
        item_aspect_2 = self.item_aspect_2(item_review) #[bs,3*32]
        item_aspect_3 = self.item_aspect_3(item_review) #[bs,3*32]
        # item aggregation layer
        mean = self.user_hidden_mean_layer(user_latent) #[bs,3*32]
        sigma = self.user_hidden_sigma_layer(user_latent) #[bs,3*32]
        if torch.cuda.is_available():
            noise = torch.randn(sigma.shape).cuda()
        else:
            noise = torch.randn(sigma.shape)
        h = mean + sigma * noise #[bs,3*32]
        attn_1 = torch.matmul(F.tanh(torch.matmul(h,self.w_h) + torch.matmul(item_aspect_1,self.w_y_1)),self.w_a_1) #[bs,1]
        attn_2 = torch.matmul(F.tanh(torch.matmul(h,self.w_h) + torch.matmul(item_aspect_2,self.w_y_2)),self.w_a_2)
        attn_3 = torch.matmul(F.tanh(torch.matmul(h,self.w_h) + torch.matmul(item_aspect_3,self.w_y_3)),self.w_a_3)
        attn = torch.cat([attn_1,attn_2,attn_3],dim = 1)#[bs, 3]
        attn = torch.softmax(attn, dim =1) #[bs, 3]
        

        item_aspect = torch.cat([item_aspect_1.unsqueeze(1),item_aspect_2.unsqueeze(1),item_aspect_3.unsqueeze(1)],dim = 1)#[bs, 3, 3*32]
        # print(attn.shape,item_aspect.shape)
        item_agg = torch.matmul(attn.unsqueeze(1),item_aspect).squeeze(1) # [bs,1,3*32]-> [bs,3*32]
        # print(u_id_embd.shape,user_latent.shape,item_agg.shape,i_id_embd.shape)
        final_tensor = torch.cat([u_id_embd,user_latent,item_agg,i_id_embd],dim = 1)#[bs,768+16+3*32+768]
        prediction = self.nfm(final_tensor)
        return prediction.view(-1)
