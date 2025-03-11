import os
import time

import numpy as np
import random
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from config import Config
from data import reviewData, reviewSepData, reviewBertData, reviewDeepShareData, load_embedding, predict_mse, predict_sep_mse, predict_deep_share_mse, date

from NCF import NCF
from FM import FM
from Deepconn import DeepCoNN
from RPMIA import RPMIA
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

def train(train_dataloader, valid_dataloader, model, config, model_path):
    print(f'{date()}## Start the training!')
    train_mse, train_mae = predict_mse(model, train_dataloader, config.device)
    valid_mse, valid_mae  = predict_mse(model, valid_dataloader, config.device)
    print(f'{date()}#### Initial train mse {train_mse:.6f}, train mae {train_mae:.6f}, validation mse {valid_mse:.6f}, validation mae {valid_mae:.6f}')
    start_time = time.perf_counter()

    opt = torch.optim.Adam(model.parameters(), config.learning_rate, weight_decay=config.l2_regularization)
    lr_sch = torch.optim.lr_scheduler.ExponentialLR(opt, config.learning_rate_decay)

    best_loss, best_epoch = 100, 0
    for epoch in range(config.train_epochs):
        model.train() 
        total_loss, total_samples = 0, 0
        total_mae = 0
        for batch in train_dataloader:
            uid, iid, user_reviews, item_reviews, ratings = map(lambda x: x.to(config.device), batch)
            predict = model(user_reviews, item_reviews,uid,iid)
            ratings = ratings.float()
            loss = F.mse_loss(predict, ratings, reduction='sum') 
            mae_loss = F.l1_loss(predict, ratings, reduction='sum')
            opt.zero_grad()  
            loss.backward() 
            opt.step() 

            total_mae += mae_loss.item()
            total_loss += loss.item()
            total_samples += len(predict)

        lr_sch.step()
        model.eval() 

        valid_mse, valid_mae = predict_mse(model, valid_dataloader, config.device)
        train_loss = total_loss / total_samples
        train_mae = total_mae / total_samples
        print(f"{date()}#### Epoch {epoch:3d}; train mse {train_loss:.6f}; train mae {train_mae:.6f}; validation mse {valid_mse:.6f}; validation mae {valid_mae:.6f}")

        if best_loss > valid_mse:
            best_loss = valid_mse
            torch.save(model, model_path)

    end_time = time.perf_counter()
    print(f'{date()}## End of training! Time used {end_time - start_time:.0f} seconds.')

def test(dataloader, model):
    print(f'{date()}## Start the testing!')
    start_time = time.perf_counter()
    test_mse_loss, test_mae_loss = predict_mse(model, dataloader, config.device)
    end_time = time.perf_counter()
    print(f"{date()}## Test end, test mse is {test_mse_loss:.6f}, test mae is {test_mae_loss:.6f}, time used {end_time - start_time:.0f} seconds.")


def train_sep(train_dataloader, valid_dataloader, model, config, model_path):
    print(f'{date()}## Start the training!')
    train_mse,train_mae = predict_sep_mse(model, train_dataloader, config.device)
    valid_mse,valid_mae = predict_sep_mse(model, valid_dataloader, config.device)
    print(f'{date()}#### Initial train mse {train_mse:.6f}, train mae {train_mae:.6f}, validation mse {valid_mse:.6f}, validation mae {valid_mae:.6f}')
    start_time = time.perf_counter()

    opt = torch.optim.Adam(model.parameters(), config.learning_rate, weight_decay=config.l2_regularization)
    lr_sch = torch.optim.lr_scheduler.ExponentialLR(opt, config.learning_rate_decay)

    best_loss, best_epoch = 100, 0
    for epoch in range(config.train_epochs):
        model.train()  
        total_loss, total_samples = 0, 0
        total_mae = 0
        for batch in train_dataloader:
            uid, iid, user_reviews, item_reviews, ratings, u_iid_seq, i_uid_seq = map(lambda x: x.to(config.device), batch)
            predict = model(user_reviews, item_reviews,uid,iid,u_iid_seq, i_uid_seq)
            ratings = ratings.float()
            loss = F.mse_loss(predict, ratings, reduction='sum') 
            mae_loss = F.l1_loss(predict, ratings, reduction='sum')
            opt.zero_grad() 
            loss.backward()  
            opt.step()

            total_mae += mae_loss.item()
            total_loss += loss.item()
            total_samples += len(predict)

        lr_sch.step()
        model.eval()  
        valid_mse, valid_mae = predict_sep_mse(model, valid_dataloader, config.device)
        train_loss = total_loss / total_samples
        train_mae = total_mae / total_samples
        print(f"{date()}#### Epoch {epoch:3d}; train mse {train_loss:.6f}; train mae {train_mae:.6f}; validation mse {valid_mse:.6f}; validation mae {valid_mae:.6f}")

        if best_loss > valid_mse:
            best_loss = valid_mse
            torch.save(model, model_path)

    end_time = time.perf_counter()
    print(f'{date()}## End of training! Time used {end_time - start_time:.0f} seconds.')

def test_sep(dataloader, model):
    print(f'{date()}## Start the testing!')
    start_time = time.perf_counter()
    test_mse_loss, test_mae_loss = predict_sep_mse(model, dataloader, config.device)
    end_time = time.perf_counter()
    print(f"{date()}## Test end, test mse is {test_mse_loss:.6f}, test mae is {test_mae_loss:.6f}, time used {end_time - start_time:.0f} seconds.")


def train_sep_bert(train_dataloader, valid_dataloader, model, config, model_path):
    print(f'{date()}## Start the training!')
    train_mse, train_mae = predict_sep_mse(model, train_dataloader, config.device)
    valid_mse, valid_mae = predict_sep_mse(model, valid_dataloader, config.device)
    print(f'{date()}#### Initial train mse {train_mse:.6f}, train mae {train_mae:.6f}, validation mse {valid_mse:.6f}, validation mae {valid_mae:.6f}')
    start_time = time.perf_counter()

    opt = torch.optim.Adam(model.parameters(), config.learning_rate, weight_decay=config.l2_regularization)
    lr_sch = torch.optim.lr_scheduler.ExponentialLR(opt, config.learning_rate_decay)

    best_loss, best_epoch = 100, 0
    for epoch in range(config.train_epochs):
        model.train()
        total_loss, total_samples = 0, 0
        total_mae = 0
        for batch in train_dataloader:
            uid, iid, user_reviews, item_reviews, ratings, u_iid_seq, i_uid_seq = map(lambda x: x.to(config.device), batch)
            predict = model(user_reviews, item_reviews,uid,iid,u_iid_seq, i_uid_seq)
            if type(predict) == tuple:
                ratings = ratings.float()
                loss = F.mse_loss(predict[0], ratings, reduction='sum')
                loss = loss + predict[1]
                mae_loss = F.l1_loss(predict[0], ratings, reduction='sum')
            else:
                ratings = ratings.float()
                loss = F.mse_loss(predict, ratings, reduction='sum')
                mae_loss = F.l1_loss(predict, ratings, reduction='sum')
            opt.zero_grad() 
            loss.backward() 
            opt.step()  

            total_mae += mae_loss.item()
            total_loss += loss.item()
            total_samples += len(predict)

        lr_sch.step()
        model.eval()  
        valid_mse, valid_mae = predict_sep_mse(model, valid_dataloader, config.device)
        train_loss = total_loss / total_samples
        train_mae = total_mae / total_samples
        print(f"{date()}#### Epoch {epoch:3d}; train mse {train_loss:.6f}; train mae {train_mae:.6f}; validation mse {valid_mse:.6f}; validation mae {valid_mae:.6f}")

        if best_loss > valid_mse:
            best_loss = valid_mse
            torch.save(model, model_path)

    end_time = time.perf_counter()
    print(f'{date()}## End of training! Time used {end_time - start_time:.0f} seconds.')

def test_sep_bert(dataloader, model):
    print(f'{date()}## Start the testing!')
    start_time = time.perf_counter()
    test_mse_loss, test_mae_loss = predict_sep_mse(model, dataloader, config.device)
    end_time = time.perf_counter()
    print(f"{date()}## Test end, test mse is {test_mse_loss:.6f}, test mae is {test_mae_loss:.6f}, time used {end_time - start_time:.0f} seconds.")

def train_deep_share(train_dataloader, valid_dataloader, model, config, model_path):
    print(f'{date()}## Start the training!')
    train_mse, train_mae = predict_deep_share_mse(model, train_dataloader, config.device)
    valid_mse, valid_mae = predict_deep_share_mse(model, valid_dataloader, config.device)
    print(f'{date()}#### Initial train mse {train_mse:.6f}, train mae {train_mae:.6f}, validation mse {valid_mse:.6f}, validation mae {valid_mae:.6f}')
    start_time = time.perf_counter()

    opt = torch.optim.Adam(model.parameters(), config.learning_rate, weight_decay=config.l2_regularization)
    lr_sch = torch.optim.lr_scheduler.ExponentialLR(opt, config.learning_rate_decay)

    best_loss, best_epoch = 100, 0
    for epoch in range(config.train_epochs):
        model.train()  
        total_loss, total_samples = 0, 0
        total_mae = 0
        for batch in train_dataloader:
            uid, iid, user_reviews, item_reviews, ratings, u_iid_seq, i_uid_seq, has_img,user_mean_rating,food_label,service_label,price_label,ambience_label,price_differ,cate_differ,location_differ,hist_post_img = map(lambda x: x.to(config.device), batch)
            predict = model(user_reviews, item_reviews,uid,iid,u_iid_seq, i_uid_seq, has_img,user_mean_rating,food_label,service_label,price_label,ambience_label,price_differ,cate_differ,location_differ,hist_post_img)
            if type(predict) == tuple:
                ratings = ratings.float()
                loss = F.mse_loss(predict[0], ratings, reduction='sum')
                loss = loss + predict[1]
                mae_loss = F.l1_loss(predict[0], ratings, reduction='sum')
            else:
                ratings = ratings.float()
                loss = F.mse_loss(predict, ratings, reduction='sum')  
                mae_loss = F.l1_loss(predict, ratings, reduction='sum')
            opt.zero_grad()  
            loss.backward() 
            opt.step() 
            
            total_mae += mae_loss.item()
            total_loss += loss.item()
            total_samples += len(ratings)

        lr_sch.step()
        model.eval() 
        valid_mse, valid_mae = predict_deep_share_mse(model, valid_dataloader, config.device)
        train_loss = total_loss / total_samples
        train_mae = total_mae / total_samples
        print(f"{date()}#### Epoch {epoch:3d}; train mse {train_loss:.6f}; train mae {train_mae:.6f}; validation mse {valid_mse:.6f}; validation mae {valid_mae:.6f}")

        if best_loss > valid_mse:
            best_loss = valid_mse
            torch.save(model, model_path)

    end_time = time.perf_counter()
    print(f'{date()}## End of training! Time used {end_time - start_time:.0f} seconds.')

def test_deep_share(dataloader, model):
    print(f'{date()}## Start the testing!')
    start_time = time.perf_counter()
    test_mse_loss, test_mae_loss = predict_deep_share_mse(model, dataloader, config.device)
    end_time = time.perf_counter()
    print(f"{date()}## Test end, test mse is {test_mse_loss:.6f}, test mae is {test_mae_loss:.6f}, time used {end_time - start_time:.0f} seconds.")



def review_doc_experiment(model):
    train_dataset = reviewData(config.train_file,config)
    valid_dataset = reviewData(config.valid_file,config)
    test_dataset = reviewData(config.test_file,config)
    train_dlr = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_dlr = DataLoader(valid_dataset, batch_size=config.batch_size)
    test_dlr = DataLoader(test_dataset, batch_size=config.batch_size)
    del train_dataset, valid_dataset, test_dataset
    print(f'{date()}## Begain training...')
    os.makedirs(os.path.dirname(config.model_file), exist_ok=True)
    train(train_dlr, valid_dlr, model, config, config.model_file)
    test(test_dlr, torch.load(config.model_file))

def review_sep_experiment(model):
    train_dataset = reviewSepData(config.train_file,config)
    valid_dataset = reviewSepData(config.valid_file,config)
    test_dataset = reviewSepData(config.test_file,config)
    train_dlr = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_dlr = DataLoader(valid_dataset, batch_size=config.batch_size)
    test_dlr = DataLoader(test_dataset, batch_size=config.batch_size)
    del train_dataset, valid_dataset, test_dataset
    print(f'{date()}## Begain training...')
    os.makedirs(os.path.dirname(config.model_file), exist_ok=True)  
    train_sep(train_dlr, valid_dlr, model, config, config.model_file)
    test_sep(test_dlr, torch.load(config.model_file))

def review_sep_bert_experiment(model):
    train_dataset = reviewBertData(config.train_file,config)
    valid_dataset = reviewBertData(config.valid_file,config)
    test_dataset = reviewBertData(config.test_file,config)
    train_dlr = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_dlr = DataLoader(valid_dataset, batch_size=config.batch_size)
    test_dlr = DataLoader(test_dataset, batch_size=config.batch_size)
    del train_dataset, valid_dataset, test_dataset
    print(f'{date()}## Begain training...')
    os.makedirs(os.path.dirname(config.model_file), exist_ok=True)  
    train_sep_bert(train_dlr, valid_dlr, model, config, config.model_file)
    test_sep(test_dlr, torch.load(config.model_file))


def review_deep_journey_experiment(model):
    train_dataset = reviewDeepShareData(config.train_file,config)
    valid_dataset = reviewDeepShareData(config.valid_file,config)
    test_dataset = reviewDeepShareData(config.test_file,config)
    train_dlr = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_dlr = DataLoader(valid_dataset, batch_size=config.batch_size)
    test_dlr = DataLoader(test_dataset, batch_size=config.batch_size)
    del train_dataset, valid_dataset, test_dataset
    print(f'{date()}## Begain training...')
    os.makedirs(os.path.dirname(config.model_file), exist_ok=True) 
    
    train_deep_share(train_dlr, valid_dlr, model, config, config.model_file)
    test_deep_share(test_dlr, torch.load(config.model_file))


if __name__ == '__main__':
    config = Config()
    print(config)
    if config.which_model not in config.bert_model:
        print(f'{date()}## Load embedding and data...')
        word_emb, word_dict = load_embedding(config.word2vec_file)
        print(f'{date()}## Load embedding and data finish')
    else:
        word_emb = []
    
    user_num, item_num = 100000,9961
    print(f'{date()}## Load model and train...')
    if config.which_model =='NCF':
        for i in range(config.repeat_expr):
            seed = random.randint(0,99999)
            setup_seed(seed)
            print('Random seed is',str(seed))
            model = NCF(user_num, item_num, config).to(config.device)
            review_doc_experiment(model)
    elif config.which_model =='FM':
        for i in range(config.repeat_expr):
            seed = random.randint(0,99999)
            setup_seed(seed)
            print('Random seed is',str(seed))
            model = FM(user_num, item_num, config).to(config.device)
            review_doc_experiment(model)
    elif config.which_model =='PMF':
        for i in range(config.repeat_expr):
            seed = random.randint(0,99999)
            setup_seed(seed)
            print('Random seed is',str(seed))
            model = PMF(config, user_num, item_num).to(config.device)
            review_doc_experiment(model)
    elif config.which_model =='DeepConn':
        for i in range(config.repeat_expr):
            seed = random.randint(0,99999)
            setup_seed(seed)
            print('Random seed is',str(seed))
            model = DeepCoNN(config, word_emb,user_num, item_num).to(config.device)
            review_doc_experiment(model)
    elif config.which_model =='D_attn':
        for i in range(config.repeat_expr):
            seed = random.randint(0,99999)
            setup_seed(seed)
            print('Random seed is',str(seed))
            model = Dattn(config, word_emb,user_num, item_num).to(config.device)
            review_doc_experiment(model)
    elif config.which_model =='ANR':
        for i in range(config.repeat_expr):
            seed = random.randint(0,99999)
            setup_seed(seed)
            print('Random seed is',str(seed))
            model = ANR(config, word_emb,user_num, item_num).to(config.device)
            review_doc_experiment(model)
    elif config.which_model =='NARRE':
        for i in range(config.repeat_expr):
            seed = random.randint(0,99999)
            setup_seed(seed)
            print('Random seed is',str(seed))
            model = NARRE(config, word_emb,user_num, item_num).to(config.device)
            review_deep_journey_experiment(model)
    elif config.which_model =='AHN':
        for i in range(config.repeat_expr):
            seed = random.randint(0,99999)
            setup_seed(seed)
            print('Random seed is',str(seed))
            model = AHN(config, word_emb,user_num, item_num).to(config.device)
            review_deep_journey_experiment(model)
    elif config.which_model =='AHAG':
        for i in range(config.repeat_expr):
            seed = random.randint(0,99999)
            setup_seed(seed)
            print('Random seed is',str(seed))
            model = AHAG(config, word_emb,user_num, item_num).to(config.device)
            review_doc_experiment(model)
    elif config.which_model =='RMCL':
        for i in range(config.repeat_expr):
            seed = random.randint(0,99999)
            setup_seed(seed)
            print('Random seed is',str(seed))
            model = RMCL(config, word_emb,user_num, item_num).to(config.device)
            review_sep_bert_experiment(model)
    elif config.which_model =='RRPU':
        for i in range(config.repeat_expr):
            seed = random.randint(0,99999)
            setup_seed(seed)
            print('Random seed is',str(seed))
            model = RRPU(config, word_emb,user_num, item_num).to(config.device)
            review_sep_bert_experiment(model)
    elif config.which_model =='RPMIA':
        for i in range(config.repeat_expr):
            seed = random.randint(0,99999)
            setup_seed(seed)
            print('Random seed is',str(seed))
            model = RPMIA(config, word_emb,user_num, item_num).to(config.device)
            review_doc_experiment(model)
    elif config.which_model =='DeepJourney_AISAS':
        from DeepJourney_AISAS_word2vec import DeepJourney
        for i in range(config.repeat_expr):
            seed = random.randint(0,99999)
            setup_seed(seed)
            print('Random seed is',str(seed))
            model = DeepJourney(config, word_emb,user_num, item_num).to(config.device)
            review_deep_journey_experiment(model)
    elif config.which_model =='DeepJourney_AISAS_bert':
        from DeepJourney_AISAS_bert import DeepJourney
        for i in range(config.repeat_expr):
            seed = random.randint(0,99999)
            setup_seed(seed)
            print('Random seed is',str(seed))
            model = DeepJourney(config, word_emb,user_num, item_num).to(config.device)
            review_deep_journey_experiment(model)
    else:
        for i in range(config.repeat_expr):
            seed = random.randint(0,99999)
            setup_seed(seed)
            print('Random seed is',str(seed))
            model = NCF(config, word_emb,user_num, item_num).to(config.device)
            review_doc_experiment(model)
    
    
