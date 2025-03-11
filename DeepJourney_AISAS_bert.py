import torch
from torch import nn
import torch.nn.functional as F

from transformers import AutoModel

class SelfAttn(nn.Module):
    def __init__(self,config):  # p=cnn_out_dim
        super(SelfAttn, self).__init__()
        self.att_linear_i_Q = nn.Linear(config.bert_out_dim, config.DeepShare_attn_dim,bias=False)
        self.att_linear_i_K = nn.Linear(config.bert_out_dim, config.DeepShare_attn_dim,bias=False)
        self.att_i_V = nn.Parameter(torch.randn(config.DeepShare_attn_dim,1)) # [50,1]
    def forward(self,item_vec):
        item_attn_Q = F.tanh(self.att_linear_i_Q(item_vec)) #[bs,10,50]
        item_attn_K = F.sigmoid(self.att_linear_i_K(item_vec)) #[bs,10,50]
        item_attn_V = torch.matmul(item_attn_Q * item_attn_K,self.att_i_V) #[bs,10,1]
        item_attn_score = F.softmax(item_attn_V,dim = 1) #[bs,10,1]
        item_agg_vec =  item_attn_score * item_vec #[bs,10,2*50]
        return item_agg_vec, item_attn_score

class CoAttn(nn.Module):
    def __init__(self,config):  # p=cnn_out_dim
        super(CoAttn, self).__init__()
        self.co_att_linear_u = nn.Linear(config.bert_out_dim, config.DeepShare_co_attn_dim)
        self.co_att_linear_i = nn.Linear(config.bert_out_dim, config.DeepShare_co_attn_dim)
        self.M = nn.Parameter(torch.randn(config.DeepShare_co_attn_dim,config.DeepShare_co_attn_dim)) # [50,50]

    def forward(self,u_vec,i_vec, item_attn_score):
        seq_num = u_vec.shape[1]
        co_att_u = self.co_att_linear_u(u_vec) #[bs,10,50]
        co_att_i = self.co_att_linear_i(i_vec).permute(0,2,1) #[bs,50,10]
        co_attn_matrix = F.relu(torch.matmul(torch.matmul(co_att_u,self.M),co_att_i)) #[bs,10,10]
        a_u_i = F.max_pool1d(co_attn_matrix * item_attn_score,seq_num) #[bs,10,1]
        user_agg_vec = F.softmax(a_u_i,1) * u_vec #[bs,10,512]
        return user_agg_vec


class AttnAggregator(nn.Module):
    def __init__(self,config):  # p=cnn_out_dim
        super(AttnAggregator, self).__init__()
        self.W_q = nn.Linear(config.id_embd_num, config.id_embd_num, bias=False)
        self.W_k = nn.Linear(config.id_embd_num, config.id_embd_num, bias=False)

    def forward(self,u_iid_seq,i_id_embd):
        '''
        u_iid_seq:[bs,10,16]
        i_id_embd: [bs,16]
        '''
        attn = torch.matmul(u_iid_seq,i_id_embd.unsqueeze(2)) #[bs,10,1]
        attn = F.softmax(attn, dim = 1)
        u_interest = torch.matmul(attn.permute(0,2,1),u_iid_seq) #[bs,1,16]
        return u_interest.squeeze(1)

class IntentGenerator(nn.Module):
    def __init__(self,config):  # p=cnn_out_dim
        super(IntentGenerator, self).__init__()
        self.user_hidden_layer = nn.Sequential(
                nn.Linear(config.bert_out_dim * 1 + config.id_embd_num, config.DeepShare_user_hidden_dims[0]),
                # nn.Tanh(),
                nn.Linear(config.DeepShare_user_hidden_dims[0], config.DeepShare_user_hidden_dims[1]),
                # nn.Sigmoid(),
            )

    def forward(self,user_latent,u_interest):
        latent = torch.cat([user_latent,u_interest], dim = 1)
        out = self.user_hidden_layer(latent) 
        return out


class AttnGreedySearch(nn.Module):
    def __init__(self,config):  # p=cnn_out_dim
        super(AttnGreedySearch, self).__init__()
        self.search_num = config.DeepShare_search_num
        self.proj = nn.Linear(config.bert_out_dim, config.DeepShare_user_hidden_dims[1])
        self.proj_k = nn.Linear(config.DeepShare_user_hidden_dims[1], config.DeepShare_user_hidden_dims[1])

    def attn(self,source_vec,target_vec):
        source_vec = torch.mean(source_vec,dim = 1,keepdim = True)
        attn_score = torch.matmul(target_vec, source_vec.permute(0,2,1)) #[bs,10,1]
        attn_score = torch.softmax(attn_score.squeeze(dim = 2), dim = 1) #[bs,10]
        return attn_score

    def forward(self,user_intent, item_corpus):
        '''
        user_intent: [bs,16]
        item_corpus: [bs,10,50*2]
        '''
        item_corpus = self.proj(item_corpus) #[bs,10,16]
        user_intent = user_intent.unsqueeze(dim = 1) #[bs,1,16]
        hidden_size = item_corpus.shape[2]
        item_vec = item_corpus
        for i in range(self.search_num):
            item_corpus = self.proj_k(item_corpus)
            attn_score = self.attn(user_intent,item_corpus)
            score,idx = torch.topk(attn_score,k=1, dim = 1)
            idx = idx.unsqueeze(1).repeat((1,1,hidden_size)) # [bs,1,16]
            item_vec = torch.gather(item_corpus,1,idx) # [bs,1,16]
            user_intent = torch.cat([user_intent,item_vec], dim = 1)
        return user_intent #[bs,1 + search_num, 16]

class AttnGreedySearchV2(nn.Module):
    def __init__(self,config):  # p=cnn_out_dim
        super(AttnGreedySearchV2, self).__init__()
        self.search_num = config.DeepShare_search_num
        self.proj = nn.Linear(config.bert_out_dim * 2, config.DeepShare_user_hidden_dims[1])
        self.w_s = nn.Linear(config.DeepShare_user_hidden_dims[1], config.DeepShare_user_hidden_dims[1],bias = False)
        self.w_t = nn.Linear(config.DeepShare_user_hidden_dims[1], config.DeepShare_user_hidden_dims[1],bias = False)

    def attn(self,source_vec,target_vec):
        source_vec = torch.mean(source_vec,dim = 1,keepdim = True)
        target_vec = F.tanh(self.w_t(target_vec))
        source_vec = F.tanh(self.w_s(source_vec))
        attn_score = torch.matmul(target_vec, source_vec.permute(0,2,1)) #[bs,10,1]
        attn_score = torch.softmax(attn_score.squeeze(dim = 2), dim = 1) #[bs,10]
        return attn_score

    def forward(self,user_intent, item_corpus):
        '''
        user_intent: [bs,16]
        item_corpus: [bs,10,50*2]
        '''
        item_corpus = self.proj(item_corpus) #[bs,10,16]
        user_intent = user_intent.unsqueeze(dim = 1) #[bs,1,16]
        hidden_size = item_corpus.shape[2]
        item_vec = item_corpus
        for i in range(self.search_num):
            attn_score = self.attn(user_intent,item_corpus)
            score,idx = torch.topk(attn_score,k=1, dim = 1)
            # self.index.append(idx.detach().numpy().tolist())
            idx = idx.unsqueeze(1).repeat((1,1,hidden_size)) # [bs,1,16]
            item_vec = torch.gather(item_corpus,1,idx) # [bs,1,16]
            user_intent = torch.cat([user_intent,item_vec], dim = 1)
        return user_intent #[bs,1 + search_num, 16]

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



class ImageSharePredictor(nn.Module):
    def __init__(self,config): 
        super(ImageSharePredictor, self).__init__()
        self.intent_proj = nn.Linear(config.DeepShare_user_hidden_dims[1] * 4 , config.DeepShare_proj_dim)
        self.price_proj = nn.Linear(1, config.DeepShare_proj_dim)
        self.location_embedding = nn.Embedding(config.DeepShare_location_new_num, config.DeepShare_proj_dim)
        self.cate_embedding = nn.Embedding(config.DeepShare_cate_new_num, config.DeepShare_proj_dim)
        
        self.predictor = nn.Sequential(
                nn.Linear(config.DeepShare_proj_dim * 4 + config.id_embd_num * 2, config.DeepShare_predictor_dim[0]),
                nn.ReLU(),
                nn.Linear(config.DeepShare_predictor_dim[0], config.DeepShare_predictor_dim[1]),
                nn.Sigmoid(),
            )

    def forward(self, rating_ref, food_intent, service_intent, ambience_intent, price_intent, price_differ,cate_differ,location_differ,hist_post_img,u_id_embd,i_id_embd):
        '''
        food_intent:[bs,16]
        service_intent:[bs,16]
        ambience_intent:[bs,16]
        price_intent:[bs,16]
        price_differ:[bs,1], float
        cate_differ: [bs,1], int
        location_differ: [bs,1], int
        '''
        intent_concat = torch.cat([food_intent,service_intent,ambience_intent,price_intent], dim = 1) # [bs,64]
        intent_proj = self.intent_proj(intent_concat) #[bs,16]
        # print(price_differ.shape)
        price_differ = price_differ.unsqueeze(1)
        price_vec = self.price_proj(price_differ.float())#[bs,16]
        location_vec = self.location_embedding(location_differ.int())#[bs,16]
        cate_vec = self.cate_embedding(cate_differ.int())#[bs,16]
        predict_vec = torch.cat([intent_proj, price_vec, location_vec, cate_vec, u_id_embd, i_id_embd], dim = 1) #[bs,16*4+32]
        predict = self.predictor(predict_vec) # [bs,1]
        return predict


class DeepJourney(nn.Module):
    def __init__(self, config, word_emb,user_num, item_num):
        super(DeepJourney, self).__init__()
        # self.embedding = nn.Embedding.from_pretrained(torch.Tensor(word_emb))
        self.bert = AutoModel.from_pretrained(config.bert_model_name, local_files_only=True)
        if config.freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.co_attn_w = CoAttn(config)
        self.co_attn_r = CoAttn(config)
        self.i_attn_w = SelfAttn(config)
        self.i_attn_r = SelfAttn(config)
        self.user_interest = AttnAggregator(config)
        self.img_proj = nn.Linear(config.DeepShare_predictor_dim[1], 1, bias = False)
        self.gate_layer = nn.Linear(config.id_embd_num * 2, 1)

        self.food_intent = IntentGenerator(config)
        self.service_intent = IntentGenerator(config)
        self.ambience_intent = IntentGenerator(config)
        self.price_intent = IntentGenerator(config)

        self.food_search = AttnGreedySearch(config)
        self.service_search = AttnGreedySearch(config)
        self.ambience_search = AttnGreedySearch(config)
        self.price_search = AttnGreedySearch(config)

        self.user_embedding = nn.Embedding(user_num, config.id_embd_num)
        self.item_embedding = nn.Embedding(item_num, config.id_embd_num)

        self.img_predictor = ImageSharePredictor(config)

        self.intent_loss = nn.KLDivLoss(reduction= 'mean')
        self.img_loss = nn.BCELoss(reduction= 'mean')
        self.Loss_weight = config.Loss_weight
        # print(config.DeepShare_user_hidden_dims[-1] * 4 * (1 + config.DeepShare_search_num) + config.DeepShare_attn_dim * 2 + config.id_embd_num * 2)
        self.fm = FactorizationMachine(config.DeepShare_user_hidden_dims[-1] * 4 * (1 + config.DeepShare_search_num) + config.bert_out_dim + config.id_embd_num * 2, 10)

    def forward(self, user_review, item_review, uid,iid, u_iid_seq, i_uid_seq, has_img, user_mean_rating,food_label,service_label,price_label,ambience_label,price_differ,cate_differ,location_differ,hist_post_img):  
 
        # *********** ATTENTION layer *********** 

        bs,seq_len,word_len = user_review.shape
        user_review = user_review.int().view(bs*seq_len,word_len)#[bs*10,50]
        item_review = item_review.int().view(bs*seq_len,word_len)
        u_s = self.bert(user_review)[1].view(bs,seq_len,-1) #[bs*10,512]->[bs,10,512]
        i_s = self.bert(item_review)[1].view(bs,seq_len,-1) 

        # *********** INTEREST layer *********** 
        # id embedding
        u_id_embd = self.user_embedding(uid)
        i_id_embd = self.item_embedding(iid)
        u_iid_seq_embd = self.item_embedding(u_iid_seq)
        i_uid_seq_embd = self.user_embedding(i_uid_seq)

        # basic intrest
        u_interest = self.user_interest(u_iid_seq_embd,i_id_embd)

        # item word level aggregation and user item word level co attention
        item_agg_vec,item_attn_score = self.i_attn_w(i_s) #[bs,10,512]
        user_agg_vec = self.co_attn_w(u_s,i_s,item_attn_score) #[bs,10,512]
        # item review level aggregation and user item review level co attention
        item_agg_vec_r, item_attn_score_r = self.i_attn_r(item_agg_vec) #[bs,10,512]
        item_agg_vec_r = torch.sum(item_agg_vec_r, dim = 1) #[bs,512]
        user_agg_vec_r = torch.sum(self.co_attn_r(user_agg_vec,item_agg_vec,item_attn_score_r), dim = 1) #[bs,512]
        # user intention generation
        food_intent = self.food_intent(user_agg_vec_r,u_interest) #[bs,16]
        service_intent = self.service_intent(user_agg_vec_r,u_interest)
        ambience_intent = self.ambience_intent(user_agg_vec_r,u_interest)
        price_intent = self.price_intent(user_agg_vec_r,u_interest)
    
        # *********** SEARCH layer *********** 
        # 这里利用4个意图搜索商品评论中相关的信息。
        food_search = self.food_search(food_intent, item_agg_vec).view(bs,-1) #[bs, 1 + search_num, 16]->[bs,(1 + search_num)*16]
        service_search = self.service_search(service_intent, item_agg_vec).view(bs,-1)
        ambience_search = self.ambience_search(ambience_intent, item_agg_vec) .view(bs,-1)
        price_search = self.price_search(price_intent, item_agg_vec).view(bs,-1)


        # *********** ACTION layer *********** 
        user_all_intent = torch.cat([food_search,service_search,ambience_search,price_search], dim = 1) #[bs,(1 + search_num)*16*4]
        u_out = torch.cat([u_id_embd,user_all_intent],dim = 1)#[bs,16*4*(search_num+1)+16]
        i_out = torch.cat([i_id_embd,item_agg_vec_r],dim = 1)#[bs,512+16]
        concat_latent = torch.cat([u_out,i_out],dim = 1)
        base_prediction = self.fm(concat_latent).float() #[bs,1]
        
        # *********** SHARE layer *********** 
        img_predict = self.img_predictor(base_prediction - user_mean_rating.float().unsqueeze(1), food_intent, service_intent,ambience_intent,price_intent, price_differ, cate_differ, location_differ, hist_post_img,u_id_embd,i_id_embd) #[bs,1]
        share_action = img_predict
        share_prediction = user_mean_rating.float().unsqueeze(1) + self.img_proj(share_action) #[bs,1]

        # combine the two predict
        gate = self.gate_layer(torch.cat([u_id_embd,i_id_embd], dim = 1))#[bs,1]
        gate = F.sigmoid(gate)
        prediction = (1 - gate) * base_prediction + gate * share_prediction

        # Calculate loss
        # img share loss
        img_predict = img_predict.view(-1)
        L_img = self.img_loss(img_predict,has_img.float())
        # merge loss
        L = self.Loss_weight[0] * L_img
        return prediction.float().view(-1), L