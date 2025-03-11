import torch
from torch import nn

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


class FM(nn.Module):

    def __init__(self,user_num, item_num, config):
        super(FM, self).__init__()
        self.user_embedding = nn.Embedding(user_num, config.id_embd_num)
        self.item_embedding = nn.Embedding(item_num, config.id_embd_num)
        self.fm = FactorizationMachine(config.id_embd_num * 2, 10)

    def forward(self, user_review, item_review,uid,iid):  # input shape(batch_size, review_count, review_length)
        # new_batch_size = user_review.shape[0] * user_review.shape[1]
        # user_review = user_review.reshape(new_batch_size, -1)
        # item_review = item_review.reshape(new_batch_size, -1)

        u_id_embd = self.user_embedding(uid)
        i_id_embd = self.item_embedding(iid)

        concat_latent = torch.cat((u_id_embd,i_id_embd), dim=1)
        prediction = self.fm(concat_latent)
        prediction = torch.sum(prediction,dim = 1).float()
        return prediction
