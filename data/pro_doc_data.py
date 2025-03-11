import pandas as pd
import time
import re
# from tqdm import tqdm
import copy
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

df=pd.read_csv('../data/5_asp_senti_avg_hist_percent.csv')
print(df.columns)

### Convert user and store id to int
def get_count(df, id):
    playcount_groupbyid = df[[id, 'rating']].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count

user_count = get_count(df, 'userId')
store_count = get_count(df, 'store_id')
unique_uid = user_count.index
unique_sid = store_count.index
user_dict = {}
for i,item in user_count.iterrows():
    user_dict[item['userId']] = i

store_dict = {}
for i,item in store_count.iterrows():
    store_dict[item['store_id']] = i

def numerize(df):
    df['user_id_new'] =  df['userId'].map(lambda x: user_dict[x])
    df['store_id_new'] = df['store_id'].map(lambda x: store_dict[x])
    return df
df=numerize(df)

### Sort df by time
df_rating=df[['user_id_new','store_id_new','rating']]
df['review'] = df['comment'].map(lambda x: eval(x)['text'])
number_sample = df_rating.shape[0]
df['localizedDate'] = df['localizedDate'].astype('datetime64[ns]')
df = df.sort_values(by=['localizedDate'], ascending=True)

def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))
def summary_data():
    print('user_number',len(user_dict))
    print('store_number',len(store_dict))
    print('sample_number',number_sample)
    # print('sample_number',number_sample)

summary_data()

### Clean review and gengerate review doc
from collections import defaultdict
user_reviews_dict = defaultdict(list)  # ｛uid:['r1','r2'...]｝
store_reviews_dict = defaultdict(list)
res = []

MAX_VOCAB = 50000
DOC_LEN = 300
MAX_DF = 0.85 # 超过多少评率的词被剔除，前10%的词被剔除


all_review = list(df.review)
vectorizer = TfidfVectorizer(max_df=MAX_DF, max_features=MAX_VOCAB)
vectorizer.fit(all_review)
vocab = vectorizer.vocabulary_
vocab[MAX_VOCAB] = '<SEP>'

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"sssss ", " ", string)
    string = string.strip().lower()
    string = ' '.join([w for w in string.split() if w in vocab])
    return string

for i,row in df.iterrows():
    str_review = clean_str(str(row['review']).encode('ascii', 'ignore').decode('ascii'))
    # process consumer review
    user_id = row['user_id_new']
    if len(user_reviews_dict[user_id])<=10:
        row['consuemr_previous_review'] = copy.deepcopy(user_reviews_dict[user_id])
    else:
        row['consuemr_previous_review'] = copy.deepcopy(user_reviews_dict[user_id][-10:])
    user_reviews_dict[user_id].append(str_review)
    # process store review
    store_id = row['store_id_new']
    if len(store_reviews_dict[store_id])<=10:
        row['store_previous_review'] = copy.deepcopy(store_reviews_dict[store_id])
    else:
        row['store_previous_review'] = copy.deepcopy(store_reviews_dict[store_id][-10:])
    store_reviews_dict[store_id].append(str_review)
    res.append(row)

review_df = pd.DataFrame(res)
review_df.to_csv('../data/previows_review_data_v2.csv',index = False)


### Map word to id
fea_columns = ['user_id_new','store_id_new','rating',
               'consuemr_previous_review','store_previous_review',]
review_df = review_df[fea_columns]
review_df = review_df.dropna(subset=['user_id_new','store_id_new'])
review_df['user_id_new'] = review_df['user_id_new'].map(int)
review_df['store_id_new'] = review_df['store_id_new'].map(int)
def load_embedding(word2vec_file):
    print('start')
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
def review2id(review):  # 将一个评论字符串分词并转为数字
    PAD_WORD = '<UNK>'
    PAD_WORD_idx = word_dict[PAD_WORD]
    review_length = 300
    try:
        # print(review)
        review = eval(review)
    except:
        # print(review)
        review=[]
    pad_word = ' <SEP> '
    review = pad_word.join(review)
    wids = []
    for word in review.split():
        if word in word_dict:
            wids.append(word_dict[word])  # 单词映射为数字
        else:
            wids.append(PAD_WORD_idx)
    if len(wids)>review_length:
        res = wids[-review_length:]
    else:
        res = [0] * (review_length - len(wids)) + wids
    return res

word2vec_file = '../data/glove.6B.300d.txt'
word_emb, word_dict = load_embedding(word2vec_file)
review_df['user_id_new'] = review_df['user_id_new'].map(int)
review_df['store_id_new'] = review_df['store_id_new'].map(int)
review_df['consuemr_previous_review'] = review_df['consuemr_previous_review'].apply(review2id)
review_df['store_previous_review'] = review_df['store_previous_review'].apply(review2id)
# int_col = ['has_img','txt_len', 'ruzhu_month','hist_post_img_num', 
#                 'food_senti_avg','service_senti_avg','ambience_senti_avg','prices_senti_avg',
#                 'food_asp_senti','service_asp_senti','ambience_asp_senti','prices_asp_senti',
#                 'price_differ', 'cate_differ','location_differ']
review_df = review_df.dropna(subset=['user_id_new','store_id_new'])
# for col in int_col:
#     review_df[col]=review_df[col].astype(int)
samples = review_df.values

number_sample = review_df.shape[0]
data_train = samples[:int(0.8 * number_sample)]
data_validation = samples[int(0.8 * number_sample):int(0.9 * number_sample)]
data_test = samples[int(0.9 * number_sample):-1]
old_user = []
for item in data_train:
    if item[0] not in old_user:
        old_user.append(item[0])
train_res = data_train.tolist()
valid_res = []
for item in data_validation:
    if item[0] not in old_user:
        train_res.append(item)
        old_user.append(item[0])
    else:
        valid_res.append(item)
test_res = []
for item in data_test:
    if item[0] not in old_user:
        train_res.append(item)
        old_user.append(item[0])
    else:
        test_res.append(item)
        
data_train_new = np.array(train_res,dtype=object)
data_valid_new = np.array(valid_res,dtype=object)
data_test_new = np.array(test_res,dtype=object)
# import numpy as np
np.save("../data/Train.npy", data_train_new)
np.save("../data/Valid.npy", data_valid_new)
np.save("../data/Test.npy", data_test_new)