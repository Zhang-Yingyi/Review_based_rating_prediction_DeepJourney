import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

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


class SelfAttn(nn.Module):
    def __init__(self,config):  # p=cnn_out_dim
        super(SelfAttn, self).__init__()
        self.att_linear_i_Q = nn.Linear(config.DeepShare_bilstm_hidden_size*2, config.DeepShare_attn_dim,bias=False)
        self.att_linear_i_K = nn.Linear(config.DeepShare_bilstm_hidden_size*2, config.DeepShare_attn_dim,bias=False)
        self.att_i_V = nn.Parameter(torch.randn(config.DeepShare_attn_dim,1)) # [50,1]
    def forward(self,item_vec):
        item_attn_Q = F.tanh(self.att_linear_i_Q(item_vec)) #[bs,10,50]
        item_attn_K = F.sigmoid(self.att_linear_i_K(item_vec)) #[bs,10,50]
        item_attn_V = torch.matmul(item_attn_Q * item_attn_K,self.att_i_V) #[bs,10,1]
        item_attn_score = F.softmax(item_attn_V,dim = 1) #[bs,10,1]
        item_agg_vec =  item_attn_score * item_vec #[bs,10,2*50]
        return item_agg_vec, item_attn_score

class CNN(nn.Module):

    def __init__(self, config, word_dim):
        super(CNN, self).__init__()
        self.kernel_count = config.kernel_count
        self.review_count = config.review_count

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1, 
                out_channels=config.kernel_count, 
                kernel_size=[config.kernel_size,word_dim], 
                stride=(1, word_dim), 
                padding=0,
                bias=False),
            nn.ReLU(),
            nn.Dropout(p=config.dropout_prob))

        # self.linear = nn.Sequential(
        #     nn.Linear(config.kernel_count, config.cnn_out_dim),
        #     nn.ReLU(),
        #     nn.Dropout(p=config.dropout_prob))

    def forward(self, vec):  # input shape(new_batch_size, review_length, word2vec_dim)
        latent = self.conv(vec.permute(0,3,1,2)).squeeze(3) 
        latent = F.pad(latent, (1, 1, 0, 0), mode='constant', value=0)
        return latent # out shape(batch_size, kernal_count, word2vec_dim)


class Cross_attn(nn.Module):
    def __init__(self, config):
        super(Cross_attn, self).__init__()
        self.rand_marrix = nn.Parameter(torch.randn(config.kernel_count,config.kernel_count))
    def forward(self, vec_u,vec_i):  # input shape(bs, kernel_size, word2vec_dim)
        bs,ks,word_dim = vec_u.shape
        vec_u_tmp = vec_u.permute(0,2,1).reshape(-1,ks) # [bs*word_dim,ks]
        vec_i_tmp = vec_i# [bs,ks,word_dim]
        u_rand = torch.matmul(vec_u_tmp,self.rand_marrix).reshape(-1,word_dim,ks) # [bs,word_dim,ks]
        f = torch.matmul(u_rand,vec_i_tmp) # [bs,word_dim,word_dim]
        att1 = F.tanh(f)# [bs,ks,ks,1]
        pool_user = torch.mean(att1, dim=2)# [bs,word_dim]
        pool_item = torch.mean(att1, dim=1)# [bs,word_dim]
        weight_user = F.softmax(pool_user,dim = 1)
        weight_item = F.softmax(pool_item,dim = 1)
        weight_user_exp = weight_user.unsqueeze(-1)# [bs,word_dim,1]
        weight_item_exp = weight_item.unsqueeze(-1)# [bs,word_dim,1]
        weighted_U = vec_u.permute(0,2,1)*weight_user_exp # [bs,300,ks]
        weighted_I = vec_i.permute(0,2,1)*weight_item_exp # [bs,300,ks]
        return weighted_U,weighted_I
    
class BI_layer_gate(nn.Module):

    def __init__(self, config):
        super(BI_layer_gate, self).__init__()
        self.user_viewpoint_embeddings = [nn.Parameter(torch.randn([1, config.kernel_count])).to(config.device) for i in range(config.num_aspect)]
        self.item_aspect_embeddings = [nn.Parameter(torch.randn([1, config.kernel_count])).to(config.device) for i in range(config.num_aspect)]
        self.W_word_gate_u = [nn.Parameter(torch.randn([config.kernel_count, config.kernel_count])).to(config.device) for i in range(config.num_aspect)]
        self.W_asps_gate_u = [nn.Parameter(torch.randn([config.kernel_count, config.kernel_count])).to(config.device) for i in range(config.num_aspect)]
        self.W_word_gate_i = [nn.Parameter(torch.randn([config.kernel_count, config.kernel_count])).to(config.device) for i in range(config.num_aspect)]
        self.W_asps_gate_i = [nn.Parameter(torch.randn([config.kernel_count, config.kernel_count])).to(config.device) for i in range(config.num_aspect)]
        self.b_gate_u = [nn.Parameter(torch.randn([config.kernel_count])).to(config.device) for i in range(config.num_aspect)]
        self.b_gate_i = [nn.Parameter(torch.randn([config.kernel_count])).to(config.device) for i in range(config.num_aspect)]
        self.num_aspect = config.num_aspect
        self.dp_rate = config.dropout_prob
        self.device = config.device
        #Aspect Self-Attention
        self.W_prj_u = [nn.Parameter(torch.randn([config.kernel_count, config.kernel_count])).to(config.device) for i in range(config.num_aspect)]
        self.b_prj_u = [nn.Parameter(torch.randn([1, 1, 1, config.kernel_count, 1])).to(config.device) for i in range(config.num_aspect)]
        self.b_ij_u = [torch.zeros([1, config.word_len, 1]).to(config.device) for i in range(config.num_aspect)]

        self.W_prj_i = [nn.Parameter(torch.randn([config.kernel_count, config.kernel_count])).to(config.device) for i in range(config.num_aspect)]
        self.b_prj_i = [nn.Parameter(torch.randn([1, 1, 1, config.kernel_count, 1])).to(config.device) for i in range(config.num_aspect)]
        self.b_ij_i = [torch.zeros([1, config.word_len, 1]).to(config.device) for i in range(config.num_aspect)]

        self.itr_0 = config.itr_0

    def asps_gate(self, contextual_words, asps_embeds, W_word, W_asps, b_gate):
        W_word = F.dropout(W_word, self.dp_rate)
        W_asps = F.dropout(W_asps, self.dp_rate)
        bs,word_dim,ks = contextual_words.shape
        contextual_words_reshape = contextual_words.reshape(-1,ks) #[bs*300,ks]
        asps_gate = torch.matmul(contextual_words_reshape, W_word) + torch.matmul(asps_embeds, W_asps) + b_gate #[bs*300,ks]
        asps_gate = F.sigmoid(asps_gate)#[bs*300,ks]
        asps_gate = asps_gate.reshape(-1,word_dim,ks)#[bs,300,ks]
        gated_contextual_words =  contextual_words*asps_gate#[bs,300,ks]
        
        return gated_contextual_words
    
    def asps_prj(self, gated_contextual_embeds, W_prj):
        bs,word_dim,ks = gated_contextual_embeds.shape #[bs,300,ks]
        gated_contextual_embeds_reshape = gated_contextual_embeds.reshape(-1,ks) #[bs*300,ks]
        out = torch.matmul(gated_contextual_embeds_reshape,W_prj)
        out = out.reshape(-1,word_dim,ks) # [bs,word_dim,ks]
        return out
    def squash(self,vector):
        vec_squared_norm =torch.sum(torch.square(vector), -2, keepdim=True)
        scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / torch.sqrt(vec_squared_norm + 1e-9)
        vec_squashed = scalar_factor * vector
        return (vec_squashed)
    
    def simple_self_attn(self, u_hat, caps_b, b_ij,input_masks):
        bs,word_dim,ks = u_hat.shape
        cap_1_b_inputs = torch.zeros([bs,1,1]).to(self.device)
        u_hat = u_hat.reshape(-1,word_dim,1,ks,1) #bs,word_dim,1,50,1
        with torch.no_grad():
            u_hat_stopped = u_hat
        b_ij = cap_1_b_inputs * b_ij
        b_ij = b_ij.unsqueeze(-1).unsqueeze(-1)
        for r_itr in range(self.itr_0):
            if (r_itr > 0):
                input_masks = input_masks.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                b_ij = b_ij * input_masks #bs,word_dim,1,1,1
            c_ij = F.softmax(b_ij, dim=1) #bs,word_dim,1,1,1
            if r_itr == self.itr_0 - 1:
                s_j = c_ij*u_hat #bs,word_dim,1,50,1
                s_j = torch.sum(s_j,dim=1,keepdim=True)+ caps_b #bs,1,1,50,1
                v_j = self.squash(s_j) #bs,1,1,50,1
            elif r_itr < self.itr_0 - 1:
                s_j = c_ij*u_hat_stopped
                s_j = torch.sum(s_j,dim=1,keepdim=True)+ caps_b
                v_j = self.squash(s_j) #bs,1,1,50,1

                v_j_tiled = torch.tile(v_j, [1, word_dim, 1, 1, 1]) #bs,word_dim,1,50,1
                u_produce_v = torch.sum(u_hat_stopped * v_j_tiled, axis=3,keepdim=True) #bs,word_dim,1,1,1

                b_ij += u_produce_v
        v_j = v_j.squeeze(-1)
        v_j = v_j.squeeze(1)
        return v_j

    def forward(self, vec_u,vec_i,users_masks_inputs,items_masks_inputs):  # input shape(bs, kernel_size, word2vec_dim)
        user_pre_attn_contextual_embeds, item_pre_attn_contextual_embeds = [], []
        for i in range(self.num_aspect):
            user_pre_attn_contextual_embeds.append(self.asps_gate( vec_u, self.user_viewpoint_embeddings[i], self.W_word_gate_u[i],self.W_asps_gate_u[i], self.b_gate_u[i]))#[[bs,300,ks]*num_aspect]]

            item_pre_attn_contextual_embeds.append(self.asps_gate( vec_i, self.item_aspect_embeddings[i], self.W_word_gate_i[i],self.W_asps_gate_i[i], self.b_gate_i[i]))

        user_viewpoint_words = []
        item_viewpoint_words = []
        for i in range(self.num_aspect):
            user_viewpoint_words.append(self.asps_prj(user_pre_attn_contextual_embeds[i], self.W_prj_u[i]))
            item_viewpoint_words.append(self.asps_prj(item_pre_attn_contextual_embeds[i], self.W_prj_i[i]))
        user_viewpoint_embeddings = []
        item_aspect_embeddings = []

        for i in range(len(self.W_prj_u)):
            u_viewpoint_v = self.simple_self_attn(user_viewpoint_words[i], self.b_prj_u[i], self.b_ij_u[i],users_masks_inputs)
            user_viewpoint_embeddings.append(u_viewpoint_v)
        for i in range(len(self.W_prj_i)):
            i_asps_v = self.simple_self_attn(item_viewpoint_words[i],self.b_prj_i[i], self.b_ij_i[i],items_masks_inputs)
            item_aspect_embeddings.append(i_asps_v)
        return user_viewpoint_embeddings,item_aspect_embeddings


class AttnAggregator(nn.Module):
    def __init__(self,config):  # p=cnn_out_dim
        super(AttnAggregator, self).__init__()
        self.w_omega = nn.Linear(2*config.kernel_count, config.prima_attn_size, bias=True)

        self.u_omega = nn.Linear(config.prima_attn_size, 1, bias=False)

    def forward(self,aspect_pairs, return_alphas=False):
        '''
        aspect_pairs # [bs, asp*asp, 2*ks]
        '''
        v = self.w_omega(aspect_pairs) # [bs, asp*asp, attn_dim]
        v = F.tanh(v) # [bs, asp*asp, attn_dim]
        vu = self.u_omega(v).squeeze(-1) # [bs, asp*asp]
        alphas = F.softmax(vu,dim = 1).unsqueeze(-1) # [bs, asp*asp,1]
        # print(aspect_pairs.shape)
        # print(alphas.shape)
        output = torch.sum(aspect_pairs * alphas, 1) # [bs,2*ks] 
        if not return_alphas:
            return output
        else:
            return output, alphas


class RPMIA(nn.Module):
    def __init__(self, config, word_emb,user_num, item_num):
        super(RPMIA, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(word_emb))
        self.user_embedding = nn.Embedding(user_num, config.id_embd_num)
        self.item_embedding = nn.Embedding(item_num, config.id_embd_num)
        self.user_bias = nn.Embedding(user_num,1) # final prediction using only
        self.item_bias = nn.Embedding(item_num,1) # final prediction using only

        self.cnn_u = CNN(config, word_dim=self.embedding.embedding_dim)
        self.cnn_i = CNN(config, word_dim=self.embedding.embedding_dim)
        self.cross_attn = Cross_attn(config)
        self.BI_l_g = BI_layer_gate(config)
        self.num_aspect = config.num_aspect
        self.attn_agg = AttnAggregator(config)
        
        self.mlp1 = nn.Linear(2*config.kernel_count,config.prima_latent_dim)
        self.mlp2 = nn.Linear(config.prima_latent_dim,1)
        self.v = nn.Parameter(torch.randn(config.prima_latent_dim, config.prima_v_dim))
        self.device = config.device

        self.fm = FactorizationMachine(2*config.kernel_count, 10)
    
    def pack_vects(self, vect_1, vect_2, type):
        if type == "mul":
            return vect_1* vect_2
        elif type == "sub":
            return vect_1-vect_2
        elif type == "both":
            res = [vect_1* vect_2,vect_1-vect_2]
            res = torch.concat(res,dim = 2)
            return res
    def pred(self,vec):
        # vec [bs, 2*ks]
        out_ori = F.relu(self.mlp1(vec))# [bs,prima_latent_dim]
        out1 = self.mlp2(out_ori) #[bs,1]
        out2 = torch.matmul(out_ori.unsqueeze(-1),out_ori.unsqueeze(1)) #[bs,prima_latent_dim,prima_latent_dim]
        out2_tmp = out2*torch.matmul(self.v,self.v.permute(1,0)) #[bs,prima_latent_dim,prima_latent_dim]
        out2 = torch.sum(out2_tmp,dim=2) #[bs,prima_latent_dim]
        out2 = torch.sum(out2,dim=1,keepdim=True) #[bs,1]
        out3 = torch.sum(out2_tmp.diagonal(dim1=-2, dim2=-1),dim=-1,keepdim=True)
        out = out1+0.5*(out2-out3)
        # print(out1.shape)
        # print(out2.shape)
        # print(out3.shape)
        # print(out.shape)
        # out3 = torch.trace(out2_tmp)
        # out = out1+0.5*(out2-out3)
        return out

    def forward(self, user_review, item_review,uid,iid):  # input shape(batch_size, review_count, review_length)
        # 0. ID embedding
        
        input_masks_u = torch.where(user_review == 0, 0, 1).to(self.device)
        input_masks_i = torch.where(user_review == 0, 0, 1).to(self.device)
        # print(input_masks_u.shape)
        # print(input_masks_i.shape)
        u_id_embd = self.user_embedding(uid)
        i_id_embd = self.item_embedding(iid)
        u_id_bs = self.user_bias(uid)
        i_id_bs = self.item_bias(iid)

        # 1. Embedding Layer
        u_vec = self.embedding(user_review) # [bs,300,300]
        i_vec = self.embedding(item_review)
        u_vec_expend = u_vec.unsqueeze(dim = -1)
        i_vec_expend = i_vec.unsqueeze(dim = -1)

        user_latent = self.cnn_u(u_vec_expend) # [bs,50,300]
        item_latent = self.cnn_i(i_vec_expend)
        weighted_U,weighted_I = self.cross_attn(user_latent,item_latent) # [bs,300,ks], [bs,300,ks]
        user_viewpoint_embeddings,item_aspect_embeddings = self.BI_l_g(weighted_U,weighted_I,input_masks_u,input_masks_i) # [[bs, 1, ks]*num_aspect], [[bs, 1, ks]*num_aspect]
        aspect_embeds_pack = []
        for i in range(self.num_aspect):
            user_asps_embed = user_viewpoint_embeddings[i] # [bs, 1, ks]
            for j in range(self.num_aspect):
                item_asps_embed = item_aspect_embeddings[j] # [bs, 1, ks]
                aspect_embeds_pack.append(self.pack_vects(user_asps_embed, item_asps_embed, "both")) # [bs, 1, 2*ks] 
        aspect_pairs = torch.concat(aspect_embeds_pack, dim=1) # [bs, asp*asp, 2*ks]
        attention_output, alphas = self.attn_agg(aspect_pairs, return_alphas=True) # [bs, 2*ks], # [bs, asp*asp]
        prediction = self.pred(attention_output)

        prediction = prediction+i_id_bs+u_id_bs #[bs,1]

        return prediction.float().view(-1)

