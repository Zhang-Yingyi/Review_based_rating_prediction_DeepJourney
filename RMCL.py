import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel

class IntentGeneratorModel(nn.Module):
    def __init__(self, config):
        super(IntentGeneratorModel, self).__init__()
        self.linear = nn.Sequential(
                nn.Linear(config.bert_out_dim, config.RMCL_intent_MLP[0]),
                nn.Softmax(dim = 1),
                nn.Linear(config.RMCL_intent_MLP[0], config.RMCL_intent_MLP[1]),
                nn.Softmax(dim = 1),
                nn.Linear(config.RMCL_intent_MLP[1], config.RMCL_intent_MLP[2]),
                nn.Softmax(dim = 1),
            )
    def forward(self,vec):
        out = self.linear(vec)
        return out

class RMCL(nn.Module):
    def __init__(self, config, word_emb,user_num, item_num):
        super(RMCL, self).__init__()
        self.intent_num = config.RMCL_num_intent
        self.bert = AutoModel.from_pretrained(config.bert_model_name, local_files_only=True)
        if config.freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.user_intent_vector = nn.Parameter(torch.randn(config.RMCL_num_intent, config.bert_out_dim)) #[10,768]
        self.user_intent_generator = IntentGeneratorModel(config)
        self.item_intent_vector = nn.Parameter(torch.randn(config.RMCL_num_intent, config.bert_out_dim)) #[10,768]
        self.item_intent_generator = IntentGeneratorModel(config)

        self.user_embedding = nn.Embedding(user_num, config.id_embd_num)
        self.item_embedding = nn.Embedding(item_num, config.id_embd_num)

        self.predict_layer = nn.Sequential(
                nn.Linear(config.bert_out_dim * 2 + config.id_embd_num * 2, config.RMCL_predict_MLP[0]),
                nn.Linear(config.RMCL_predict_MLP[0], config.RMCL_predict_MLP[1]),
                nn.Linear(config.RMCL_predict_MLP[1], config.RMCL_predict_MLP[2]),
            )

    def forward(self, user_review, item_review,uid,iid, u_iid_seq, i_uid_seq):  
        bs,seq_len,word_len = user_review.shape
        user_review = user_review.int().view(bs*seq_len,word_len)#[bs*10,50]
        item_review = item_review.int().view(bs*seq_len,word_len)

        user_review = self.bert(user_review)[1].view(bs,seq_len,-1) #[bs*10,768]->[bs,10,768]
        item_review = self.bert(item_review)[1].view(bs,seq_len,-1) 
        user_review = torch.mean(user_review, dim = 1)
        item_review = torch.mean(item_review, dim = 1)

        user_intent_generate = self.user_intent_generator(user_review) #[bs,10]
        user_intent_agg = torch.matmul(user_intent_generate,self.user_intent_vector) #[bs,768]

        item_intent_generate = self.item_intent_generator(item_review)
        item_intent_agg = torch.matmul(item_intent_generate,self.item_intent_vector) #[bs,768]

        u_id_embd = self.user_embedding(uid) # [bs,16]
        i_id_embd = self.item_embedding(iid)

        concat_latent = torch.cat([user_intent_agg*item_intent_agg,user_intent_agg-item_intent_agg,u_id_embd,i_id_embd],dim = 1) #[bs,768*2+16*2]
        prediction = self.predict_layer(concat_latent)

        L_sim = F.cosine_similarity(user_intent_agg,user_review) + F.cosine_similarity(item_intent_agg,item_review) # [bs]
        L_ind = torch.matmul(self.user_intent_vector, self.user_intent_vector.permute(1,0))**2  #[num_intent,num_intent]

        L_cl_u = torch.log(torch.softmax(torch.matmul(user_review,self.user_intent_vector.t()),dim = 1)) # [bs,10]
        L_cl_u = user_intent_generate * L_cl_u #[bs,10]
        L_cl_i = torch.log(torch.softmax(torch.matmul(item_review,self.item_intent_vector.t()),dim = 1)) # [bs,10]
        L_cl_i = item_intent_generate * L_cl_i #[bs,10]

        L_sim = F.sigmoid(torch.sum(L_sim)).view(-1) #[1]
        L_ind = F.sigmoid(torch.sum(L_ind)).view(-1) #[1]
        L_cl = F.sigmoid(torch.sum(L_cl_u + L_cl_i)).view(-1) #[1]
        L = L_sim + L_ind + L_cl
        return (prediction.view(-1),L)
