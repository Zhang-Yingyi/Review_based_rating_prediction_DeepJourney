import torch
from torch import nn
import torch.nn.functional as F 

class NCF(nn.Module):
	def __init__(self, user_num, item_num,config, GMF_model=None, MLP_model=None):
		super(NCF, self).__init__()
		"""
		user_num: number of users;
		item_num: number of items;
		factor_num: number of predictive factors;
		num_layers: the number of layers in MLP model;
		dropout: dropout rate between fully connected layers;
		model: 'MLP', 'GMF', 'NeuMF-end', and 'NeuMF-pre';
		GMF_model: pre-trained GMF weights;
		MLP_model: pre-trained MLP weights.
		"""		
		self.dropout = config.dropout_prob
		self.model = config.NCF_model
		self.GMF_model = config.GMF_model
		self.MLP_model = config.MLP_model

		self.embed_user_GMF = nn.Embedding(user_num,  config.id_embd_num)
		self.embed_item_GMF = nn.Embedding(item_num,  config.id_embd_num)
		self.embed_user_MLP = nn.Embedding(user_num, config.id_embd_num)
		self.embed_item_MLP = nn.Embedding(item_num, config.id_embd_num)

		MLP_modules = []
		for i in range(1,len(config.num_layers)):
			input_size = config.num_layers[i-1]
			MLP_modules.append(nn.Dropout(p=self.dropout))
			MLP_modules.append(nn.Linear(input_size, config.num_layers[i]))
			MLP_modules.append(nn.ReLU())
		self.MLP_layers = nn.Sequential(*MLP_modules)

		if self.model in ['MLP', 'GMF']:
			predict_size = config.id_embd_num 
		else:
			predict_size = config.id_embd_num + config.num_layers[-1]
		self.predict_layer = nn.Linear(predict_size, 1)

		self._init_weight_()

	def _init_weight_(self):
		""" We leave the weights initialization here. """
		if not self.model == 'NeuMF-pre':
			nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
			nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
			nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
			nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

			for m in self.MLP_layers:
				if isinstance(m, nn.Linear):
					nn.init.xavier_uniform_(m.weight)
			nn.init.kaiming_uniform_(self.predict_layer.weight, 
									a=1, nonlinearity='sigmoid')

			for m in self.modules():
				if isinstance(m, nn.Linear) and m.bias is not None:
					m.bias.data.zero_()
		else:
			# embedding layers
			self.embed_user_GMF.weight.data.copy_(
							self.GMF_model.embed_user_GMF.weight)
			self.embed_item_GMF.weight.data.copy_(
							self.GMF_model.embed_item_GMF.weight)
			self.embed_user_MLP.weight.data.copy_(
							self.MLP_model.embed_user_MLP.weight)
			self.embed_item_MLP.weight.data.copy_(
							self.MLP_model.embed_item_MLP.weight)

			# mlp layers
			for (m1, m2) in zip(
				self.MLP_layers, self.MLP_model.MLP_layers):
				if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
					m1.weight.data.copy_(m2.weight)
					m1.bias.data.copy_(m2.bias)

			# predict layers
			predict_weight = torch.cat([
				self.GMF_model.predict_layer.weight, 
				self.MLP_model.predict_layer.weight], dim=1)
			precit_bias = self.GMF_model.predict_layer.bias + \
						self.MLP_model.predict_layer.bias

			self.predict_layer.weight.data.copy_(0.5 * predict_weight)
			self.predict_layer.bias.data.copy_(0.5 * precit_bias)

	def forward(self, user_reviews, item_reviews, user, item):
		if not self.model == 'MLP':
			embed_user_GMF = self.embed_user_GMF(user)
			embed_item_GMF = self.embed_item_GMF(item)
			output_GMF = embed_user_GMF * embed_item_GMF
		if not self.model == 'GMF':
			embed_user_MLP = self.embed_user_MLP(user)
			embed_item_MLP = self.embed_item_MLP(item)
			interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
			output_MLP = self.MLP_layers(interaction)

		if self.model == 'GMF':
			concat = output_GMF
		elif self.model == 'MLP':
			concat = output_MLP
		else:
			concat = torch.cat((output_GMF, output_MLP), -1)

		prediction = self.predict_layer(concat)
		return prediction.view(-1)


# class NCF(nn.Module):

#     def __init__(self, config, word_emb,user_num, item_num):
#         super(DeepCoNN, self).__init__()
#         # self.embedding = nn.Embedding.from_pretrained(torch.Tensor(word_emb))
#         # self.cnn_u = CNN(config, word_dim=self.embedding.embedding_dim)
#         # self.cnn_i = CNN(config, word_dim=self.embedding.embedding_dim)
#         self.user_embedding = nn.Embedding(user_num, config.id_embd_num)
#         self.item_embedding = nn.Embedding(item_num, config.id_embd_num)
#         self.fm = FactorizationMachine(config.cnn_out_dim * 2 + config.id_embd_num * 2, 10)

#     def forward(self, user_review, item_review,uid,iid):  # input shape(batch_size, review_count, review_length)
#         # new_batch_size = user_review.shape[0] * user_review.shape[1]
#         # user_review = user_review.reshape(new_batch_size, -1)
#         # item_review = item_review.reshape(new_batch_size, -1)
        

#         u_vec = self.embedding(user_review)
#         i_vec = self.embedding(item_review)

#         user_latent = self.cnn_u(u_vec)
#         item_latent = self.cnn_i(i_vec)

#         u_id_embd = self.user_embedding(uid)
#         i_id_embd = self.item_embedding(iid)

#         concat_latent = torch.cat((user_latent, item_latent,u_id_embd,i_id_embd), dim=1)
#         prediction = self.fm(concat_latent)
#         prediction = torch.sum(prediction,dim = 1).float()
#         return prediction
