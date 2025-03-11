import argparse
import inspect

import torch


class Config:
    # device = torch.device("cuda:0")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_epochs = 20
    batch_size = 128
    learning_rate = 0.002
    l2_regularization = 1e-6  # 权重衰减程度
    learning_rate_decay = 0.99  # 学习率衰减程度
    which_model = 'DeepConn'

    word2vec_file = '../data/glove.6B.300d.txt'
    # train_file = '../data/train.csv'
    # train_file = '../data/train_small.csv'
    # valid_file = '../data/valid.csv'
    # test_file = '../data/test.csv'
    train_file = '../data/Train.npy'
    valid_file = '../data/Valid.npy'
    test_file = '../data/Test.npy'
    model_file = './DeepConn_model/best_model.pt'

    review_count = 10  # max review count
    review_length = 300  # max review length
    lowest_review_count = 2  # reviews wrote by a user/item will be delete if its amount less than such value.
    PAD_WORD = '<UNK>'
    repeat_expr = 10

    # general
    id_embd_num = 16
    dropout_prob = 0.5
    word_embd_dim = 300
    word_len = 300
    review_num = 10
    review_word_num = 50
    bert_model_name = './bert_small'
    bert_out_dim = 512
    bert_model = ['RMCL','RRPU']
    freeze_bert=False

    # NCF Setting
    NCF_model = 'Mix'
    GMF_model = None
    MLP_model = None
    num_layers = [32,16]
    
    # PMF
    lam_u=0.3
    lam_v=0.3

    # DeepConn Setting
    kernel_count = 100
    kernel_size = 3
    cnn_out_dim = 300  # CNN输出维度

    # D_attn
    d_attn_window_size = 1
    l_attn_hidden_size = 10
    l_attn_kernal_size = 1
    l_attn_conv_kernal_size = [1,300]
    
    g_attn_conv_kernal_size = [2,3,4]
    g_attn_kernal_size = 10
    d_attn_fc_1 = 330
    d_attn_fc_2 = 50

    # ANR Setting
    aspect_num = 5
    aspect_hidden_size = 10
    affinity_hidden_size = 50
    window_size = 3

    # NARRE Setting
    each_kernal = 1
    narre_kernel_num = review_num * each_kernal
    narre_kernel_size = [5,100]
    narre_cnn_out_dim = 299
    narre_hidden_dim = 128

    narre_attn_size = 10
    narre_final_size = 16

    # AHN Setting
    AHN_bilstm_hidden_size = 50
    AHN_item_attn_dim = 50
    AHN_co_attn_dim = 50
    AHN_dim_M = AHN_co_attn_dim
    
    # AHAG setting
    AHAG_attn_dim = 16
    AHAG_attn_head_num = 5
    AHAG_FC_1 = 100
    AHAG_FC_2 = 50
    AHAG_conv_num = 10
    AHAG_conv_kernel = 1
    AHAG_co_attn_dim = 1
    AHAG_kernel_count = 10
    AHAG_kernel_size = 3
    AHAG_cnn_out_dimn = 100
    AHAG_NFM_MLP_dim = [32,1]

    # RMCL setting
    RMCL_num_intent = 10
    RMCL_intent_MLP = [64,32,10]
    RMCL_predict_MLP = [128,32,1]

    # RRPU setting
    RRPU_user_cnn_channel = 40
    RRPU_kernel_size = 3
    RRPU_aspect_dim = 32
    RRPU_user_hidden_dims = [64,32]
    RRPU_item_attn_dim = 32

    # DeepShare w2v version setting
    DeepShare_bilstm_hidden_size = 50
    DeepShare_co_attn_dim = 50
    DeepShare_attn_dim = 50
    DeepShare_user_hidden_dims = [64,16]
    DeepShare_search_num = 3
    DeepShare_intent_num = 4

    DeepShare_proj_dim = 16
    DeepShare_location_new_num = 2
    DeepShare_cate_new_num = 2
    DeepShare_predictor_dim = [64,1]
    # Loss_weight = [5e-3,5e-5] # base
    Loss_weight = 5e-3 # base

    def __init__(self):
        # By the way, we can customize parameters in the command line parameters.
        # For example:
        # python main.py --device cuda:0 --train_epochs 50
        attributes = inspect.getmembers(self, lambda a: not inspect.isfunction(a))
        attributes = list(filter(lambda x: not x[0].startswith('__'), attributes))

        parser = argparse.ArgumentParser()
        for key, val in attributes:
            parser.add_argument('--' + key, dest=key, type=type(val), default=val)
        for key, val in parser.parse_args().__dict__.items():
            self.__setattr__(key, val)

    def __str__(self):
        attributes = inspect.getmembers(self, lambda a: not inspect.isfunction(a))
        attributes = list(filter(lambda x: not x[0].startswith('__'), attributes))
        to_str = ''
        for key, val in attributes:
            to_str += '{} = {}\n'.format(key, val)
        return to_str