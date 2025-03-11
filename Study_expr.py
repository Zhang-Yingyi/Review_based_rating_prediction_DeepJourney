import os
import time

import numpy as np
import random
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from config import Config
from data import reviewData, reviewSepData, reviewBertData, reviewDeepJourneyData, load_embedding, predict_mse, predict_sep_mse, predict_deep_share_mse, date

from NCF import NCF
from FM import FM
from Deepconn import DeepCoNN
from ANR import ANR
from NARRE import NARRE
from D_attn import Dattn
from AHN import AHN
from pmf import PMF
from AHAG import AHAG
from RMCL import RMCL
from RRPU import RRPU


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def t_test(ls1,ls2):
    from scipy import stats
    list1 = ls1 #[1.45121,1.420726,1.40809,1.395226,1.408289,1.407572,1.405371,1.428804,1.430641,1.427872]
    list2 = ls2 #[1.3614,1.380571,1.371358,1.394871,1.370172,1.392991,1.381964,1.383205,1.376423,1.368501]
    print(stats.levene(list1, list2, center='median'))
    print(stats.stats.ttest_ind(list1, list2, equal_var=True))


if __name__ == '__main__':
    config = Config()
    print(config)
    word_emb, word_dict = load_embedding(config.word2vec_file)
    
    user_num, item_num = 100000,9961
    print(f'{date()}## Load model and train...')

    from DeepJourney_AISAS_word2vec import DeepJourney
    model = torch.load('./DeepJourney_ASAS_model_best/best_model.pt', map_location=torch.device('cpu'))
    # from DeepJourney_AISAS_word2vec_expr import DeepJourney
    # model = torch.load('./DeepJourney_AISAS_model/best_model.pt', map_location=torch.device('cpu'))
    deepjourney = DeepJourney(config, word_emb,user_num, item_num).to(config.device)
    deepjourney.load_state_dict(model.state_dict())
    test_dataset = reviewDeepJourneyData('../data/Test_review_deep_journey.npy',config)
    test_dlr = DataLoader(test_dataset, batch_size=50)
    i = 0
    uid_ls = []
    iid_ls = []
    ratings_ls = []
    for batch in test_dlr:
        i+=1
        if i>643:
            uid, iid, user_reviews, item_reviews, ratings, u_iid_seq, i_uid_seq, has_img,user_mean_rating,food_label,service_label,price_label,ambience_label,price_differ,cate_differ,location_differ,hist_post_img = map(lambda x: x.to(config.device), batch)
            predict = deepjourney(user_reviews, item_reviews,uid,iid,u_iid_seq, i_uid_seq, has_img,user_mean_rating,food_label,service_label,price_label,ambience_label,price_differ,cate_differ,location_differ,hist_post_img)
            print('************Test*********',str(i))
            uid_ls.append(uid)
            iid_ls.append(iid)
            ratings_ls.append(ratings)
        if i>643:
            break
    print('uid',uid_ls)
    print('iid',iid_ls)
    print('rating',ratings_ls)
    print('intent',deepjourney.all_intent)
    print('search',deepjourney.search_index)
    print('gate',deepjourney.gate_ls)
    print('prediction',deepjourney.prediction_ls)
    print('Img_share',deepjourney.img_pred)