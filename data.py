import time
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from config import Config
from torch.utils.data import DataLoader
from torch.nn import functional as F

def date(f='%Y-%m-%d %H:%M:%S'):
    return time.strftime(f, time.localtime())


def load_embedding(word2vec_file):
    with open(word2vec_file, encoding='utf-8') as f:
        word_emb = list()
        word_dict = dict()
        word_emb.append([0])
        word_dict['<UNK>'] = 0
        for line in f.readlines():
            tokens = line.split(' ')
            word_emb.append([float(i) for i in tokens[1:]])
            word_dict[tokens[0]] = len(word_dict)
        word_emb[0] = [0] * len(word_emb[1])
    return word_emb, word_dict

def predict_mse(model, dataloader, device):
    mse, sample_count = 0, 0
    mae = 0
    with torch.no_grad():
        for batch in dataloader:
            uid, iid, user_reviews, item_reviews, ratings = map(lambda x: x.to(device), batch)
            predict = model(user_reviews, item_reviews,uid,iid)
            mse += torch.nn.functional.mse_loss(predict, ratings, reduction='sum').item()
            mae += F.l1_loss(predict, ratings, reduction='sum').item()
            sample_count += len(ratings)
    return mse / sample_count, mae / sample_count   # dataloader上的均方误差

def predict_sep_mse(model, dataloader, device):
    mse, sample_count = 0, 0
    mae = 0
    with torch.no_grad():
        for batch in dataloader:
            uid, iid, user_reviews, item_reviews, ratings, u_iid_seq, i_uid_seq = map(lambda x: x.to(device), batch)
            predict = model(user_reviews, item_reviews,uid,iid,u_iid_seq, i_uid_seq)
            if type(predict) == tuple:
                ratings = ratings.float()
                mse += torch.nn.functional.mse_loss(predict[0], ratings, reduction='sum').item()
                mae += F.l1_loss(predict[0], ratings, reduction='sum').item()
                # loss = loss + predict[1]
            else:
                ratings = ratings.float()
                mse += torch.nn.functional.mse_loss(predict, ratings, reduction='sum').item()  # 平方和误
                mae += F.l1_loss(predict, ratings, reduction='sum').item()
            sample_count += len(ratings)
    return mse / sample_count, mae / sample_count # dataloader上的均方误差

def predict_deep_share_mse(model, dataloader, device):
    mse, sample_count = 0, 0
    mae = 0
    with torch.no_grad():
        for batch in dataloader:
            uid, iid, user_reviews, item_reviews, ratings, u_iid_seq, i_uid_seq, has_img,user_mean_rating,food_label,service_label,price_label,ambience_label,price_differ,cate_differ,location_differ,hist_post_img  = map(lambda x: x.to(device), batch)
            predict = model(user_reviews, item_reviews,uid,iid,u_iid_seq, i_uid_seq, has_img,user_mean_rating,food_label,service_label,price_label,ambience_label,price_differ,cate_differ,location_differ,hist_post_img)
            # print(predict)
            if type(predict) == tuple:
                ratings = ratings.float()
                mse += torch.nn.functional.mse_loss(predict[0], ratings, reduction='sum').item()
                mae += F.l1_loss(predict[0], ratings, reduction='sum').item()
                # loss = loss + predict[1]
            else:
                ratings = ratings.float()
                mse += torch.nn.functional.mse_loss(predict, ratings, reduction='sum').item()  # 平方和误
                mae += F.l1_loss(predict, ratings, reduction='sum').item()
            sample_count += len(ratings)
    return mse / sample_count, mae / sample_count   # dataloader上的均方误差


class reviewDeepShareData(Dataset):
    def __init__(self, path,config, is_training=False):
        super(reviewDeepShareData, self).__init__()
        features = np.load(path, allow_pickle=True)
        self.uid = [features[i][0] for i in range(len(features))]
        self.iid = [features[i][1] for i in range(len(features))]
        self.udoc = [np.array(features[i][3]).astype(int) for i in range(len(features))] #laod input_ids
        self.idoc = [np.array(features[i][4]).astype(int) for i in range(len(features))]
        self.u_iid_seq = [np.array(features[i][5]).astype(int) for i in range(len(features))]
        self.i_uid_seq = [np.array(features[i][6]).astype(int) for i in range(len(features))]
        self.has_img = [np.array(features[i][7]).astype(int) for i in range(len(features))]
        self.user_mean_rating = [np.array(features[i][8]).astype(float) for i in range(len(features))]
        self.food_label = [np.array(features[i][9]).astype(float) for i in range(len(features))]
        self.service_label = [np.array(features[i][10]).astype(float) for i in range(len(features))]
        self.price_label = [np.array(features[i][11]).astype(float) for i in range(len(features))]
        self.ambience_label = [np.array(features[i][12]).astype(float) for i in range(len(features))]
        self.price_differ = [np.array(features[i][13]).astype(float) for i in range(len(features))]
        self.cate_differ = [np.array(features[i][14]).astype(float) for i in range(len(features))]
        self.location_differ = [np.array(features[i][15]).astype(float) for i in range(len(features))]
        self.hist_post_img = [np.array(features[i][16]).astype(float) for i in range(len(features))]
        self.labels = [features[i][2] for i in range(len(features))]
        self.is_training = is_training

    def __len__(self):
        return len(self.uid)

    def __getitem__(self, idx):
        uid = int(self.uid[idx])
        iid = int(self.iid[idx])
        udoc = self.udoc[idx]
        idoc = self.idoc[idx]
        u_iid_seq = self.u_iid_seq[idx]
        i_uid_seq = self.i_uid_seq[idx]
        has_img = self.has_img[idx]
        user_mean_rating = self.user_mean_rating[idx]
        food_label = self.food_label[idx]
        service_label = self.service_label[idx]
        price_label = self.price_label[idx]
        ambience_label = self.ambience_label[idx]
        price_differ = self.price_differ[idx]
        cate_differ = self.cate_differ[idx]
        location_differ = self.location_differ[idx]
        hist_post_img = self.hist_post_img[idx]
        label = self.labels[idx]
        return uid, iid, udoc, idoc, label,u_iid_seq,i_uid_seq,has_img,user_mean_rating,food_label,service_label,price_label,ambience_label,price_differ,cate_differ,location_differ,hist_post_img

class reviewBertData(Dataset):
    def __init__(self, path,config, is_training=False):
        super(reviewBertData, self).__init__()
        features = np.load(path, allow_pickle=True)
        self.uid = [features[i][0] for i in range(len(features))]
        self.iid = [features[i][1] for i in range(len(features))]
        self.udoc = [np.array(features[i][3]).astype(int) for i in range(len(features))] #laod input_ids
        self.idoc = [np.array(features[i][4]).astype(int) for i in range(len(features))]
        self.u_iid_seq = [np.array(features[i][5]).astype(int) for i in range(len(features))]
        self.i_uid_seq = [np.array(features[i][6]).astype(int) for i in range(len(features))]
        self.labels = [features[i][2] for i in range(len(features))]
        self.is_training = is_training

    def __len__(self):
        return len(self.uid)

    def __getitem__(self, idx):
        uid = int(self.uid[idx])
        iid = int(self.iid[idx])
        udoc = self.udoc[idx]
        idoc = self.idoc[idx]
        u_iid_seq = self.u_iid_seq[idx]
        i_uid_seq = self.i_uid_seq[idx]
        label = self.labels[idx]
        return uid, iid, udoc, idoc, label,u_iid_seq,i_uid_seq

class reviewSepData(Dataset):
    def __init__(self, path,config, is_training=False):
        super(reviewSepData, self).__init__()
        features = np.load(path, allow_pickle=True)
        self.uid = [features[i][0] for i in range(len(features))]
        self.iid = [features[i][1] for i in range(len(features))]
        self.udoc = [np.array(features[i][3]).astype(int) for i in range(len(features))]
        self.idoc = [np.array(features[i][4]).astype(int) for i in range(len(features))]
        self.u_iid_seq = [np.array(features[i][5]).astype(int) for i in range(len(features))]
        self.i_uid_seq = [np.array(features[i][6]).astype(int) for i in range(len(features))]
        self.labels = [features[i][2] for i in range(len(features))]
        self.is_training = is_training

    def __len__(self):
        return len(self.uid)

    def __getitem__(self, idx):
        uid = int(self.uid[idx])
        iid = int(self.iid[idx])
        udoc = self.udoc[idx]
        idoc = self.idoc[idx]
        u_iid_seq = self.u_iid_seq[idx]
        i_uid_seq = self.i_uid_seq[idx]
        label = self.labels[idx]
        return uid, iid, udoc, idoc, label,u_iid_seq,i_uid_seq


class reviewData():
    def __init__(self, path,config, is_training=False):
        super(reviewData, self).__init__()
        features = np.load(path, allow_pickle=True)
        self.uid = [features[i][0] for i in range(len(features))]
        self.iid = [features[i][1] for i in range(len(features))]
        self.udoc = [np.array(features[i][3]).astype(int) for i in range(len(features))]
        self.idoc = [np.array(features[i][4]).astype(int) for i in range(len(features))]
        self.labels = [features[i][2] for i in range(len(features))]
        self.is_training = is_training

    def __len__(self):
        return len(self.uid)

    def __getitem__(self, idx):
        uid = self.uid[idx]
        iid = self.iid[idx]
        udoc = self.udoc[idx]
        idoc = self.idoc[idx]
        label = self.labels[idx]
        return uid, iid, udoc, idoc, label    
