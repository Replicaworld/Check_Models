#!/usr/bin/env python
# coding: utf-8

# In[1]:


# cd home/karanverma


# In[2]:


print('Starting of feature dump')
import logging
import datetime
import time
import os

import time
start_time = time.time()

import os
import sys

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# In[10]:
log_path = '/home/DS_Live_Trials/Median_Classifer_Model_features_log/'
print(os.getcwd())
if not os.path.exists(log_path):
    os.makedirs(log_path)


# In[3]:


import time
start_time = time.time()


# In[5]:


import json
PROJECT = "inshorts-1374"


config_file = '/home/karanverma/files/news_mongo_to_bq.json'
# print(config_file)
configs=json.load(open(config_file,'r'))

configs['mongo_details']['hosts_list'] = ['171.16.11.97','171.16.11.96','171.16.11.94']
configs['mongo_details']['db'] = 'trialDB'
configs['mongo_details']['tbl'] = 'median_classifier_collection'
configs['bq_details']['db'] = 'tmp'
configs['bq_details']['tbl'] = 'device_embedding'
configs

hosts=','.join(configs['mongo_details']['hosts_list'])
bq_location = configs['bq_details']['db']+"."+configs['bq_details']['tbl']
if(configs['write_mode'].strip()==''):
    write_mode = "overwrite"
else:
    write_mode = configs['write_mode']

# conf = SparkConf().setAll([('spark.jars.packages',
# 'com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.27.0,org.mongodb.spark:mongo-spark-connector_2.12:10.1.1')
# ,("spark.mongodb.read.connection.uri","mongodb://{0}:{1}@{2}/{3}?readPreference=secondaryPreferred".format(
# configs['mongo_details']['username'],configs['mongo_details']['password'],hosts,configs['mongo_details']['defaultauthdb']
# )),
#  ("spark.mongodb.write.connection.uri","mongodb://{0}:{1}@{2}/{3}?readPreference=secondaryPreferred".format(
# configs['mongo_details']['username'],configs['mongo_details']['password'],hosts,configs['mongo_details']['defaultauthdb']
# )) ])


# In[6]:


import datetime
import numpy as np
from pyspark.sql import Row
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.sql.functions import lit, udf, when
from pyspark.sql.types import *
from pyspark.sql import DataFrameStatFunctions as stat
import numpy as np
import pandas as pd
from pymongo import MongoClient
from pyspark import SparkContext, SparkConf
from pyspark.sql import Row
from pyspark.sql import SQLContext
from pyspark.sql import functions as F
from pyspark.sql.functions import lit, udf, when
from pyspark.sql.types import *
from pyspark.sql.types import *
from pyspark.sql import Window

import requests
from collections import defaultdict
from pymongo import MongoClient
from tqdm import tqdm
from pyspark.sql import DataFrameStatFunctions as stat
from pyspark.sql import Window
import pyspark.sql.functions as F
import os
import time
import json
import pickle

NIS_DATA_BASE_PATH = "gs://nis-segment-datasource-v3/processed/"
NIS_OLD_DATA_BASE_PATH = "gs://nis-localytics-datasource/processed/"

conf = SparkConf().setAll([('spark.driver.memory', '100g'), ('spark.broadcast.blockSize', '50m'),
                           ("spark.executor.instances", '50'),
                          ('spark.jars.packages',
                            'com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.27.0,org.mongodb.spark:mongo-spark-connector_2.12:10.1.1')
                            ,("spark.mongodb.read.connection.uri","mongodb://{0}:{1}@{2}/{3}?readPreference=secondaryPreferred".format(
                            configs['mongo_details']['username'],configs['mongo_details']['password'],hosts,configs['mongo_details']['defaultauthdb']
                            )),
                             ("spark.mongodb.write.connection.uri","mongodb://{0}:{1}@{2}/{3}?readPreference=secondaryPreferred".format(
                            configs['mongo_details']['username'],configs['mongo_details']['password'],hosts,configs['mongo_details']['defaultauthdb']
                            ))
                                  ])

sc = SparkContext(conf=conf)
sc.addPyFile("/home/karanverma/files/sparktorch-0.2.0.zip")
sqlContext = SQLContext(sc)


# In[7]:


from Utils import *
def getNewsData(d1, d2):
    newsMap = getNewsInDates(d1, d2)
    hashIdList = list(newsMap.keys())

    hashIdsWithFilter = []
    for h in hashIdList:
        if 'newsLanguage' in newsMap[h] and newsMap[h]['newsLanguage'] == 'english' and newsMap[h]['publishGroupList'][0]['countryCode'] == 'IN':
            hashIdsWithFilter.append(h.split('-')[0])
    
    return hashIdsWithFilter, newsMap


# In[8]:


import logging

logger = logging.getLogger(str(datetime.datetime.today().date()))
hdlr = logging.FileHandler(
    log_path + str(datetime.datetime.today().date()) + '.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.DEBUG)


# In[9]:


def Log(s, flag=True):
    if flag:
        logger.info(s)
        print(s)

Log('----', flag=True)


# In[10]:


train_days = 1
test_days = 1
embed_days = 5
n_days = train_days + test_days + embed_days
k = 0
st_date = datetime.datetime.now() - datetime.timedelta(days=n_days)


date_fmt = "%Y/%m/%d"

dates = [(st_date + datetime.timedelta(days=i)) for i in range(n_days+1)]
dates.sort()

dates_str = [date.strftime(date_fmt) for date in dates]
dates_str_embed = dates_str[0:embed_days]
dates_str_train = dates_str[embed_days:embed_days+train_days+1]
dates_str_test = dates_str[-test_days:]
print("embed data dates : ", dates_str_embed, len(dates_str_embed))
print("train data dates : ",dates_str_train, len(dates_str_train) )
print("test data dates : ",dates_str_test, len(dates_str_test))
# millisMin = dates[0].timestamp() * 1000
# millisMax = (dates[-1] + datetime.timedelta(days=1)).timestamp() * 1000
print(dates[0]- datetime.timedelta(days=30), dates[-1]+ datetime.timedelta(days=30))
news_dates = [dates[0]- datetime.timedelta(days=30), dates[-1]+ datetime.timedelta(days=30)]
hashIdsWithFilter, newsMap = getNewsData(news_dates[0], news_dates[1])


# In[11]:


len(newsMap)


# In[12]:


from pyspark.sql.functions import array, col
import torch.nn as nn
from torch.nn.parallel import DataParallel
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification, RobertaTokenizer , RobertaForSequenceClassification
from transformers import EarlyStoppingCallback

# path_embedding = 'gs://pvtrough_asia_south1/clustering/openai_embeddings_raw'
path_embedding = 'gs://pvtrough_asia_south1/clustering/roberta_live_embedding_reduced'
try:
    
    df_embedding = sqlContext.read.csv(path_embedding, sep=',',
                         inferSchema=True, header=True)
    already_embedded = df_embedding.select('hid').distinct().collect()
    already_embedded = [x['hid'] for x in already_embedded]
    print("existing embeddings found")
except:
    print("Existing News Embeddings not found, generating new ones")
    already_embedded = []
# df_embedding.take(1)


# In[108]:


newsMapProcessed = {}
document_embeddings = {}
documents_content = []
documents_id = []

# already_embedded = []
for hId in tqdm(newsMap):
    h = hId.split('-')[0]
    if h in already_embedded or (h not in hashIdsWithFilter) :
        continue
    else:    
        newsMapProcessed[h] = {}
        newsMapProcessed[h]['title'] = newsMap[hId]['title']
        newsMapProcessed[h]['content'] = newsMap[hId]['content']
        newsMapProcessed[h]['features'] = newsMapProcessed[h]['title'] + "." + newsMapProcessed[h]['content']
#         document_embeddings[h] = get_embedding(model, tokenizer, newsMapProcessed[h]['features'])
        documents_content.append(newsMapProcessed[h]['features'])
        documents_id.append(h)
        
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[13]:


import torch.nn as nn
from torch.nn.parallel import DataParallel
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification, RobertaTokenizer , RobertaForSequenceClassification
from transformers import EarlyStoppingCallback


model_name = "roberta-base"
k = 17
# model_path = "output/checkpoint-12000"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=k,output_hidden_states=True)

# Move the model to the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Check the number of available GPUs
if torch.cuda.device_count() > 1:
    # Specify which GPUs to use
    device_ids = [0, 1]  # Adjust the GPU IDs based on your system configuration
    model = DataParallel(model, device_ids=device_ids)
    print(isinstance(model, DataParallel))


def get_embedding(model, tokenizer, text):
    model.eval()

    with torch.no_grad():
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        input_ids = tokenizer.encode(text, add_special_tokens=True, return_tensors='pt').to(device)
        outputs = model(input_ids=input_ids)
        hidden_states = outputs.hidden_states
        last_layer_hidden_states = hidden_states[-1]
        embeddings = torch.mean(last_layer_hidden_states, dim=1).squeeze().cpu().numpy()

    return embeddings


# In[14]:


import numpy as np
import os
import sys
import torch
import numpy as np

from transformers import RobertaModel, RobertaTokenizer

from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity


def Dim_reduction(sentences, tokenizer, model):
    '''
        This method will accept array of sentences, roberta tokenizer & model
        next it will call methods for dimention reduction
    '''

    vecs = []
    with torch.no_grad():

        for sentence in tqdm(sentences):
            inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True,  max_length=64)
            inputs['input_ids'] = inputs['input_ids'].to(DEVICE)
            inputs['attention_mask'] = inputs['attention_mask'].to(DEVICE)

            hidden_states = model(**inputs, return_dict=True, output_hidden_states=True).hidden_states

            #Averaging the first & last hidden states
            output_hidden_state = (hidden_states[-1] + hidden_states[1]).mean(dim=1)

            vec = output_hidden_state.cpu().numpy()[0]

            vecs.append(vec)
    
#     print("35:",vecs)
    #Finding Kernal
    kernel, bias = compute_kernel_bias([vecs])
    kernel = kernel[:, :256]
    #If you want to reduce it to 128 dim
    #kernel = kernel[:, :128]
    embeddings = []
    embeddings = np.vstack(vecs)
#     print("43:", kernel, bias)
#     print("44:",embeddings)
    #Sentence embeddings can be converted into an identity matrix
    #by utilizing the transformation matrix
    embeddings = transform_and_normalize(embeddings, 
                kernel=kernel,
                bias=bias
            )
    return embeddings
def transform_and_normalize(vecs, kernel, bias):
    """
        Applying transformation then standardize
    """
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    return normalize(vecs)
    
def normalize(vecs):
    """
        Standardization
    """
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5
    
def compute_kernel_bias(vecs):
    """
    Calculate Kernal & Bias for the final transformation - y = (x + bias).dot(kernel)
    """
    vecs = np.concatenate(vecs, axis=0)
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(s**0.5))
    W = np.linalg.inv(W.T)
    return W, -mu


# In[15]:

if len(documents_content) > 0:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embeddings =  Dim_reduction(documents_content, tokenizer, model)
    document_embeddings = dict(zip(documents_id, embeddings))


# In[16]:


if len(document_embeddings) > 0:
    df_em_pandas = pd.DataFrame(document_embeddings).T
    df_em_pandas['hid'] = df_em_pandas.index
    df_em_pandas['embed_date'] = (datetime.datetime.now()) .strftime("%Y_%m_%d")
    df_em_spark = (sqlContext.createDataFrame(df_em_pandas))
    df_em_spark.write.partitionBy('embed_date').csv(path_embedding , sep=',', header=True,mode = 'append')


# In[17]:


    df_embedding = sqlContext.read.csv(path_embedding, sep=',',
                             inferSchema=True, header=True)
    df_embedding = df_embedding.distinct()
    # df_embedding.take(1)


    # In[80]:


    Log(path_embedding)


    # In[18]:


    Log("Embedding Created at %s"%path_embedding)
    # df_embedding.write.parquet('gs://pvtrough_asia_south1/clustering/roberta_live_embedding_reduced_bq',mode='overwrite')
    sqlContext.setConf("temporaryGcsBucket","pvtrough_asia_south1")
    df_embedding.write.mode('append').format('bigquery').option('project',
                                                 PROJECT).option('table','tmp_persisted.roberta_live_embedding_reduced').save()
    Log("Embedding Table Loaded at tmp_persisted.roberta_live_embedding_reduced ")


# In[ ]:


from google.cloud import bigquery
PROJECT = 'inshorts-1374'
bq = bigquery.Client(project=PROJECT)
from dateutil.relativedelta import relativedelta
import datetime
import time 
import sys
# pytz_1=pytz.timezone('Asia/Calcutta')
main_query_st_time = time.time()
func_mapper={"avg":"avg(field)",
 "median":"PERCENTILE_CONT(field, 0.5)"
}

allowed_time_and_their_prefix={"quarter_of_day":"QD","hour_of_day":"HD","day_of_week":"DW","daily":"D","weekend":"WND","overall":"O"}
part_of_sec_pri_concept=set(["quarter_of_day","hour_of_day","day_of_week"])
inp_tbl='inshorts_raw_data.inshorts_union_raw_v1'
#configs
# run_dt=datetime.date.today()-relativedelta(hours=5,minutes=30)
run_dt=datetime.datetime.now()-relativedelta(hours=5,minutes=30)

pre_select_filter_expr = " and eventname='TimeSpent-Front'"


post_select_filter_expr = " and ts<100"
time_and_their_agg={
    "quarter_of_day":{"median":{"avg":{"sec":28,"pri":28}}}
    ,"hour_of_day":{"median":{"avg":{"sec":196,"pri":196}}}
    ,"day_of_week":{"median":{"avg":{"sec":7,"pri":7}}}
    ,"daily":{"median":{"avg":7}}
    ,"weekend":{"median":{"avg":2}}
    ,"overall":{"median":{"avg":1}}
}

case_sensitive_catg_list=['national','entertainment','sports','technology','world','business','politics','startup']
process_last_n_days=7

news_embed_length=256
device_feature_out_tbl='tmp_persisted.device_l_features'
news_feature_out_tbl='tmp_persisted.hash_l_features'
category_feature_out_tbl='tmp_persisted.category_l_features'
news_embedding_tbl='tmp_persisted.roberta_live_embedding_reduced'
news_mongo_tbl="inshorts_mongo_data_tech.news_data"
device_embedding_tbl='tmp_persisted.device_embedding'
dev_master_tbl='inshorts_scheduled.device_master'
#configs

news_df="(select hid as hashid,["+','.join(["`{0}`".format(i) for i in range(news_embed_length)])+"] as newsemb from {0})".format(news_embedding_tbl)
mongo_news_df='''(select hashid,min (hr_since_publish) as hr_since_publish from
  (select split(_id,'-')[OFFSET(0)] as hashid,timestamp_diff(current_timestamp(),createdAt,hour) as hr_since_publish from {0})
group by hashid)'''.format(news_mongo_tbl)

select_expr=''' deviceid,split(hashid,'-')[OFFSET(0)] as hashid,overalltimespent as ts,eventtimestamp '''



gcs_inp_fmt="%Y/%m/%d"
minimal_bucket_prefix="gs://inshorts-minimal-event/data/inshorts-minimal-event-v1/"

if( (run_dt.hour>=5 and run_dt.minute>=30 ) and (run_dt.hour<=9 and run_dt.minute<=30 )) :
    new_uri=[minimal_bucket_prefix+run_dt.strftime(gcs_inp_fmt)+"/*.gz",
            minimal_bucket_prefix+(run_dt-relativedelta(days=1)).strftime(gcs_inp_fmt)+"/*.gz" 
            ]
else:
    new_uri=[minimal_bucket_prefix+(run_dt).strftime(gcs_inp_fmt)+"/*.gz" ]
    
ext_tsf_tbl_configs = {
"bq_dataset": "tmp_persisted",
"bq_name": "raw_ext1",
"bq_autodetect": True,
"src_file_typ": "NEWLINE_DELIMITED_JSON",
"src_uris" : new_uri
}


def create_ext_tbl(ext_tbl):
# ext_tbl["src_uris"] = [ext_tbl["buck_dir"]+date+"/*.gz" for date in list(set(all_dates))]
# print (ext_tbl)
    drop_prev_table = '''drop table if exists {0}.{1}.{2}'''.format(PROJECT,ext_tbl['bq_dataset'],ext_tbl['bq_name'])

    external_config = bigquery.ExternalConfig(ext_tbl["src_file_typ"])
    external_config.source_uris = ext_tbl["src_uris"]
    external_config.autodetect = ext_tbl["bq_autodetect"]

    tsfront_table = bigquery.Table(bigquery.TableReference(bq.dataset(ext_tbl['bq_dataset']), ext_tbl['bq_name']))
    tsfront_table.external_data_configuration = external_config
    bq.query(drop_prev_table).result()
    bq.create_table(tsfront_table)
create_ext_tbl(ext_tsf_tbl_configs)

def common_raw_df(process_last_n_days,durations):
    return '''
                        (select *,{7},row_number() over (partition by deviceid order by eventtimestamp) as unique_row_id from
                          (select deviceid,hashid,sum(ts) as ts,max(eventtimestamp) as eventtimestamp
                          ,datetime(timestamp_millis(max(eventtimestamp)),'Asia/Calcutta') as dt from
                            ((select {0} from {1} where event_date>='{2}' and event_date<='{3}' {5})
                            union all
                            (select device_id as deviceid,split(hashid,'-')[OFFSET(0)] as hashid,short_time as ts,`at` as eventtimestamp  from tmp_persisted.raw_ext1
                            where date(timestamp_millis(`at`))>='{2}' and date(timestamp_millis(`at`))<='{3}' and event_name='TimeSpent-Front'))
                          group by deviceid,hashid
                          )
                        where date(dt)>='{4}' and date(dt)<='{3}' {6}
                        )'''.format(select_expr,inp_tbl
    ,(run_dt-relativedelta(days=process_last_n_days)).strftime('%Y-%m-%d'),run_dt.strftime('%Y-%m-%d'),(run_dt-relativedelta(days=process_last_n_days-1)).strftime('%Y-%m-%d')
    ,pre_select_filter_expr,post_select_filter_expr
        ,"["+",".join([handle_first_duration(x) for x in durations])+"] as arr_time_obj")
                                                                                                                                    

master_dt=run_dt-relativedelta(days=1)

max_inst_time={}
first_agg_expr,first_agg_cols=[],[]
second_agg_expr,second_agg_cols,sec_pri_arr_for_x=[],{},{}
field_and_first_agg_func_map={'ts':set()}
for x in time_and_their_agg:
    max_for_x=0
    sec_pri_arr_for_x[x] = set()
    
    if(x not in allowed_time_and_their_prefix.keys()):
        sys.exit("{0} duration not in scope".format(x))
    for first_agg in time_and_their_agg[x].keys():
        if(first_agg not in func_mapper.keys()):
            sys.exit("{1} first level agg in {0} not in scope".format(x,first_agg))
        if (first_agg not in field_and_first_agg_func_map['ts']):
            field_and_first_agg_func_map['ts'].add(first_agg)
            first_agg_expr.append("{0} over(first_grp) as {1}_{2}".format(func_mapper[first_agg].replace('field','ts'),first_agg,'ts'))
            first_agg_cols.append("{0}_{1}".format(first_agg,'ts'))

        for second_agg in time_and_their_agg[x][first_agg].keys():
            if(second_agg not in func_mapper.keys()):
                sys.exit("{2} second level agg in {0}[{1}] not in scope".format(x,first_agg,second_agg))
            if(x in part_of_sec_pri_concept):
                invalid_det=set(time_and_their_agg[x][first_agg][second_agg].keys())-set(['sec','pri'])
                if(len(invalid_det)>0):
                    sys.exit("{3} extra in {0}[{1}][{2}] only sec:Secondary,pri:Primary allowed".format(x,first_agg,second_agg,','.join(invalid_det)))
                else:
                    for sec_pri_val in time_and_their_agg[x][first_agg][second_agg].keys():
                        sec_pri_arr_for_x[x].add(sec_pri_val)
                        
                        max_for_x=max(max_for_x,time_and_their_agg[x][first_agg][second_agg][sec_pri_val])
                        case_when_cond_for_col="when duration='{0}' and  sec_pri='{4}' and instances<={1} then {2}_{3}".format(
                            allowed_time_and_their_prefix[x],time_and_their_agg[x][first_agg][second_agg][sec_pri_val],first_agg,'ts',sec_pri_val)
                        tmp_col_name="{0}_{1}_{2}".format(second_agg,first_agg,'ts')
                        if(tmp_col_name not in second_agg_cols):
                            second_agg_cols[tmp_col_name]=[case_when_cond_for_col]
                        else:
                            second_agg_cols[tmp_col_name].append(case_when_cond_for_col)

            else:
                sec_pri_arr_for_x[x].add('pri')
                max_for_x=max(max_for_x,time_and_their_agg[x][first_agg][second_agg])
                case_when_cond_for_col="when duration='{0}' and  sec_pri='{4}' and instances<={1} then {2}_{3}".format(
                    allowed_time_and_their_prefix[x],time_and_their_agg[x][first_agg][second_agg],first_agg,'ts','pri')
                tmp_col_name="{0}_{1}_{2}".format(second_agg,first_agg,'ts')
                if(tmp_col_name not in second_agg_cols):
                    second_agg_cols[tmp_col_name]=[case_when_cond_for_col]
                else:
                    second_agg_cols[tmp_col_name].append(case_when_cond_for_col)
                
    max_inst_time[x]=max_for_x
for i in second_agg_cols.keys():    
    second_agg_expr.append("{0} over(second_grp) as {1}".format(func_mapper[second_agg].replace('field',"(case "+" ".join(second_agg_cols[i])+" end)"),i)) 
    
def epoch (x):
    return int(1000*time.mktime(x.timetuple()))

                      
def handle_first_duration(x):
    if(x=='quarter_of_day'):
        return "(case "+" ".join(["when eventtimestamp>={0} and eventtimestamp<{1} then '{2}_{3}_{4}'".format(epoch(run_dt-relativedelta(hours=6*(i+1))),epoch(run_dt-relativedelta(hours=6*i))
                                    ,allowed_time_and_their_prefix[x],4-(i%4),i+1) for i in range(max_inst_time[x])])+" else '' end)"# as "+x
    if(x=='hour_of_day'):
        return "(case "+" ".join(["when eventtimestamp>={0} and eventtimestamp<{1} then '{2}_{3}_{4}'".format(epoch(run_dt-relativedelta(hours=1*(i+1))),epoch(run_dt-relativedelta(hours=1*i))
                                    ,allowed_time_and_their_prefix[x],24-(i%24),i+1) for i in range(max_inst_time[x])])+" else '' end)"# as "+x
    if(x=='day_of_week'):
        return "(case "+" ".join(["when eventtimestamp>={0} and eventtimestamp<{1} then '{2}_{3}_{4}'".format(epoch(run_dt-relativedelta(days=1*(i+1))),epoch(run_dt-relativedelta(days=1*i))
                                    ,allowed_time_and_their_prefix[x],(run_dt-relativedelta(days=1*i)).weekday()+1,i+1) for i in range(max_inst_time[x])])+" else '' end)"# as "+x
    if(x=='daily'):
        return "(case "+" ".join(["when eventtimestamp>={0} and eventtimestamp<{1} then '{2}_{3}_{4}'".format(epoch(run_dt-relativedelta(days=1*(i+1))),epoch(run_dt-relativedelta(days=1*i))
                                    ,allowed_time_and_their_prefix[x],0,i+1) for i in range(max_inst_time[x])])+" else '' end)"# as "+x
    if(x=='weekend'):
        return "(case "+" ".join(["when eventtimestamp>={0} and eventtimestamp<{1} then '{2}_{3}_{4}'".format( epoch( run_dt - relativedelta(days = (i*7) + run_dt.weekday() +3) ),
                                                                                                      epoch( run_dt - relativedelta(days = (i*7) + run_dt.weekday() +1) )
                                    ,allowed_time_and_their_prefix[x],0,i+1) for i in range(max_inst_time[x])])+" else '' end)"# as "+x
    
    if(x=='overall'):
        return "'{0}_0_1'".format(allowed_time_and_their_prefix[x])

def level_q(grp_lev):
    if('category' in grp_lev):
        special_grp_handle="CROSS JOIN UNNEST (categories) as category"
        catg_list="where category in ('{0}')".format("','".join(case_sensitive_catg_list)) 
        defining_catg=",category.element as category"
    else:
        special_grp_handle,catg_list,defining_catg='','',''
        
    dev_l='''
    select {8},duration,sec_pri,sec_val,{13} from
      (select *,{12}  from
        (select * except(sec_val),(case when sec_pri='sec' then sec_val else null end) as sec_val from
          (select {8},duration,instances,sec_val,{10},{11} from
            (select *,{9} from
              (select *,split(time_obj,'_')[OFFSET(0)] as duration,cast(split(time_obj,'_')[OFFSET(1)] as int) as sec_val,cast(split(time_obj,'_')[OFFSET(2)] as int) as instances from
                (select * from
                  (select *  from
                    (select * {16} from
                    {17}
                    {14}
                    )
                  {15}
                  )
                CROSS JOIN UNNEST (arr_time_obj) as time_obj
                )
              where time_obj!=''
              )
            WINDOW first_grp AS (partition by {8},duration,instances)
            )
          group by {8},duration,instances,sec_val
          )
        CROSS JOIN UNNEST (sec_pri_arr) as sec_pri
        )
      WINDOW second_grp AS (partition by {8},duration,sec_pri,sec_val) 
      )
    group by {8},duration,sec_pri,sec_val
    '''.format(
        select_expr,inp_tbl
    ,(run_dt-relativedelta(days=process_last_n_days)).strftime('%Y-%m-%d'),run_dt.strftime('%Y-%m-%d'),(run_dt-relativedelta(days=process_last_n_days-1)).strftime('%Y-%m-%d')
    ,pre_select_filter_expr,post_select_filter_expr
        ,"["+",".join([handle_first_duration(x) for x in time_and_their_agg.keys()])+"] as arr_time_obj"
    ,grp_lev, ','.join(first_agg_expr)
        ,"(case "+" ".join(["when duration='{0}' then {1} ".format(allowed_time_and_their_prefix[x],list(sec_pri_arr_for_x[x]))
                      for x in sec_pri_arr_for_x]) + "end) as sec_pri_arr"
        ,','.join(["max({0}) as {0}".format(i) for i in first_agg_cols])
        ,','.join(second_agg_expr)
        ,','.join(["max({0}) as {0}".format(i) for i in second_agg_cols.keys()])
        ,special_grp_handle,catg_list,defining_catg
        ,common_raw_df(process_last_n_days,time_and_their_agg.keys())
    )
    
    return dev_l

bq.query('''CREATE OR REPLACE TABLE {0} AS  '''.format(device_feature_out_tbl)+level_q('deviceid')).result()
bq.query('''CREATE OR REPLACE TABLE {0} AS '''.format(news_feature_out_tbl)+level_q('hashid')).result()
# bq.query('''CREATE OR REPLACE TABLE {0} AS '''.format(category_feature_out_tbl)+level_q('category')).result()
                                                                                                                                
bq.query('''
CREATE OR REPLACE TABLE {10} AS
  (select dev_embed_df.*,devicehashcode from
    (select deviceid,array_agg(index_val/(case when total_multip=0 then 1 else total_multip end)) as device_emb,max(total_multip) as total_multip from
      (select deviceid,index_no,sum(multip*index_val) as index_val,sum(multip) as total_multip from
        (select deviceid ,multip,index_val,index_no from
            (select event_df.*,newsemb,(ts-avg_median_ts) as multip from
              {11}event_df
              left join
              {7}news_embed_df
              on (news_embed_df.hashid=event_df.hashid)
              left join
              (select deviceid,avg_median_ts from {8} where duration='{9}')overall_median
              on (event_df.deviceid=overall_median.deviceid)
            )
        CROSS JOIN UNNEST (newsemb) as index_val with offset as index_no
        )
      group by deviceid,index_no
      )
    group by deviceid
    )dev_embed_df
    left join
    (select deviceid,devicehashcode from {12} where app_name='inshorts')dev_mast
    on (dev_embed_df.deviceid=dev_mast.deviceid)
  )
'''.format(
select_expr,inp_tbl
,(run_dt-relativedelta(days=process_last_n_days)).strftime('%Y-%m-%d'),run_dt.strftime('%Y-%m-%d'),(run_dt-relativedelta(days=process_last_n_days-1)).strftime('%Y-%m-%d')
,pre_select_filter_expr,post_select_filter_expr,news_df,device_feature_out_tbl, allowed_time_and_their_prefix['overall'],device_embedding_tbl
,common_raw_df(process_last_n_days,time_and_their_agg.keys()),dev_master_tbl)
     ).result()

pivot_cols=[]
for x in list(set(sec_pri_arr_for_x.keys())-set(['overall'])):
    for i in list(sec_pri_arr_for_x[x]): 
        pivot_cols.append("{0}_{1}".format(allowed_time_and_their_prefix[x],i))
        
final_out_tbl='tmp_persisted.feature_per_event'
bq.query('''
CREATE OR REPLACE TABLE {15} AS
(select event_df.*,hr_since_publish from
  (select * from
    (select raw_df.deviceid,raw_df.hashid,ts,eventtimestamp,unique_row_id,pivot_cols  ,struct(dev_l,hash_l) as features from
      (select * except(sec_val), (case when sec_pri='pri' then null else sec_val end) as sec_val from 
          (select * except(sec_pri_arr),concat(duration,'_',sec_pri) as pivot_cols
          from
            (select *,{8} from         
              (select *,split(time_obj,'_')[OFFSET(0)] as duration,cast(split(time_obj,'_')[OFFSET(1)] as int) as sec_val from
                (select * except(arr_time_obj) from
                {16}
                CROSS JOIN UNNEST (arr_time_obj) as time_obj
                )
              where time_obj!=''
              )
            )
          CROSS JOIN UNNEST (sec_pri_arr) as sec_pri
          )
      )raw_df
      left join 
      (select deviceid,duration,sec_pri,sec_val,struct({9}) as dev_l from {10}
      )dev_feat
      on (dev_feat.deviceid=raw_df.deviceid and dev_feat.duration=raw_df.duration and dev_feat.sec_pri=raw_df.sec_pri and 
      (dev_feat.sec_val=raw_df.sec_val or (dev_feat.sec_val is null and raw_df.sec_val is null) ))
      left join 
      (select hashid,duration,sec_pri,sec_val,struct({9}) as hash_l from {11}
      )hash_feat
      on (hash_feat.hashid=raw_df.hashid and hash_feat.duration=raw_df.duration and hash_feat.sec_pri=raw_df.sec_pri and 
      (hash_feat.sec_val=raw_df.sec_val or (hash_feat.sec_val is null and raw_df.sec_val is null) ))  

    )
  PIVOT (
  any_value(features)
  for pivot_cols in ({12})
  ))event_df
  left join
  {13}news_df
  on (news_df.hashid=event_df.hashid)
--  left join
--  (select deviceid,device_emb from {14})device_embed_df
--  on (event_df.deviceid=device_embed_df.deviceid)
)
'''.format(select_expr,inp_tbl
    ,(run_dt-relativedelta(days=process_last_n_days)).strftime('%Y-%m-%d'),run_dt.strftime('%Y-%m-%d'),(run_dt-relativedelta(days=process_last_n_days-1)).strftime('%Y-%m-%d')
    ,pre_select_filter_expr,post_select_filter_expr
           ,"["+",".join([handle_first_duration(x) for x in time_and_their_agg.keys()])+"] as arr_time_obj"
           ,"(case "+" ".join(["when duration='{0}' then {1} ".format(allowed_time_and_their_prefix[x],list(sec_pri_arr_for_x[x]))
                      for x in sec_pri_arr_for_x]) + "end) as sec_pri_arr"
           ,','.join(second_agg_cols.keys()),device_feature_out_tbl,news_feature_out_tbl,"'"+"','".join(pivot_cols)+"'"
           ,mongo_news_df,device_embedding_tbl,final_out_tbl,
           common_raw_df(1,time_and_their_agg.keys())
          )
     ).result()
main_query_time = time.time() - main_query_st_time
Log("Main Data Query Executed")


# In[ ]:





# In[ ]:


try:
    embedding_length = len(embeddings[0])
except:
    if 'openai' in path_embedding:
        embedding_length = 1536
    else:
        embedding_length = 256
Log("embedding length : %s"%embedding_length)


# In[21]:


df_embedding = sqlContext.read.csv(path_embedding, sep=',',
                         inferSchema=True, header=True)
df_embedding = df_embedding.distinct()
feat_cols = [str(x) for x in range(0,embedding_length)]
df_embedding = df_embedding.select('hid', array([col(x) for x in feat_cols ]).alias('embedding'))
# df_embedding.show()


# In[118]:


# In[41]:


temp = sqlContext.read.format('bigquery').option('project',PROJECT).option('table','tmp_persisted.device_embedding').load()
temp.printSchema()


# In[42]:


from pyspark.sql.functions import array_contains, size, col
from pyspark.ml.clustering import KMeans, GaussianMixture, BisectingKMeans, LDA
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
from sklearn.cluster import AgglomerativeClustering
from pyspark.ml.feature import MinMaxScaler, PCA

# print(path_vec)
temp = sqlContext.read.format('bigquery').option('project',PROJECT).option('table','tmp_persisted.device_embedding').load()
df = temp.filter(~array_contains(temp.device_emb, float('nan')))

# df = temp.sample(fraction=0.4)
print(embedding_length)
df = df.dropna(how='all')
df = df.where(size(col("device_emb")) == embedding_length)

to_vector_udf = udf(lambda arr: Vectors.dense(arr), VectorUDT())
df_with_vector = df.select("deviceid", to_vector_udf("device_emb").alias("features"))


# num_components = embedding_length  # Number of components should match the original vector length
# pca = PCA(k=50, inputCol="features", outputCol="scaled_features")
# pca_model = pca.fit(df_with_vector)
# scaled_data = pca_model.transform(df_with_vector)


# In[45]:


train_data_events = sqlContext.read.format('bigquery').option('project',
                                                              PROJECT).option('table','tmp_persisted.feature_per_event').load()
# train_data_events.take(2)


# In[48]:


# train_data_events.printSchema()


# In[51]:


import pyspark.sql.functions as F
train_data = train_data_events.select("deviceid", 'hashid', 'ts', 'eventtimestamp',
                                      F.col("D_pri.dev_l.avg_median_ts").alias("dev_median_ts"),
                                     F.col("QD_sec.dev_l.avg_median_ts").alias("qs_dev_median_ts"),
                                     F.col("DW_sec.dev_l.avg_median_ts").alias("dw_dev_median_ts"),
                                     F.col("HD_sec.dev_l.avg_median_ts").alias("hd_dev_median_ts"),
                                     F.col("QD_pri.dev_l.avg_median_ts").alias("qdp_dev_median_ts"),
                                     F.col("HD_pri.dev_l.avg_median_ts").alias("hdp_dev_median_ts"),
                                     F.col("DW_pri.dev_l.avg_median_ts").alias("dwp_dev_median_ts"),
                                     F.col("WND_pri.dev_l.avg_median_ts").alias("wnd_dev_median_ts"),
                                     
                                     
                                     F.col("D_pri.hash_l.avg_median_ts").alias("hash_median_ts"),
                                     F.col("QD_sec.hash_l.avg_median_ts").alias("qs_hash_median_ts"),
                                     F.col("DW_sec.hash_l.avg_median_ts").alias("dw_hash_median_ts"),
                                     F.col("HD_sec.hash_l.avg_median_ts").alias("hd_hash_median_ts"),
                                     F.col("QD_pri.hash_l.avg_median_ts").alias("qdp_hash_median_ts"),
                                     F.col("HD_pri.hash_l.avg_median_ts").alias("hdp_hash_median_ts"),
                                     F.col("DW_pri.hash_l.avg_median_ts").alias("dwp_hash_median_ts"),
                                     F.col("WND_pri.hash_l.avg_median_ts").alias("wnd_hash_median_ts"),
                                     )
train_data = train_data.withColumn("deviation_from_median", F.col("ts") - F.col("dev_median_ts"))
   
train_data = train_data.sort("deviceId", "eventtimestamp")


# In[52]:


num_partitions = 300
train_data = train_data.withColumn("partition_hash", F.abs(F.hash("deviceId")) % num_partitions)
joined_df = train_data.join(df_with_vector, "deviceid")
joined_df = joined_df.join(df_embedding, joined_df.hashid == df_embedding.hid)
joined_df = joined_df.withColumn("rating",F.when(joined_df.deviation_from_median >= 0, 
                                                     1.0).otherwise(0.0))


# In[161]:


# joined_df.columns


# In[53]:


device_features = [
 'dev_median_ts',
 'qs_dev_median_ts',
 'dw_dev_median_ts',
 'hd_dev_median_ts',
 'qdp_dev_median_ts',
 'hdp_dev_median_ts',
 'dwp_dev_median_ts',
 'wnd_dev_median_ts']

news_features = ['hash_median_ts',
 'qs_hash_median_ts',
 'dw_hash_median_ts',
 'hd_hash_median_ts',
 'qdp_hash_median_ts',
 'hdp_hash_median_ts',
 'dwp_hash_median_ts',
 'wnd_hash_median_ts']


# In[54]:


# from pyspark.ml.feature import VectorAssembler

# to_vector_udf = udf(lambda arr: Vectors.dense(arr), VectorUDT())
# train_cols = ['features', 'deviation_from_median' ,'rating', 'partition_hash']
# train_data_null = joined_df.select(*train_cols,*device_features,*news_features, to_vector_udf("embedding").alias("embedding_vec"))
# # assembler = VectorAssembler(inputCols=['features', 'embedding_vec'], outputCol="vector_features")
# #assembler.transform(joined_df)


# # In[111]:


# # train_data_null.select('features','embedding_vec', 'rating', 'partition_hash', *device_features).write.format('bigquery').option('project',
# #                                              PROJECT).option('table','tmp.train_data_null').mode('overwrite').save()


# # In[55]:


# # train_data_null  = sqlContext.read.format('bigquery').option('project',PROJECT).option('table','tmp.train_data_null').load()
# path_train = "gs://pvtrough_asia_south1/clustering/data_train_null"
# train_data_null.select('features','embedding_vec', 'rating', 'partition_hash', *device_features, *news_features).\
#                                                             write.partitionBy('partition_hash').\
#                                                             parquet(path_train,
#                                                                             mode= 'overwrite')
# Log("Train Data dumped into GCS at : %s"%path_train)



from pyspark.sql.functions import current_timestamp
feature_insert_st_time = time.time()
temp = sqlContext.read.format('bigquery').option('project',PROJECT).option('table','tmp_persisted.device_embedding').load()
df = temp.filter(~array_contains(temp.device_emb, float('nan')))
df = df.join(joined_df.select('deviceid',*device_features).distinct(),['deviceid'],"inner")
df = df.distinct()
df = df.withColumnRenamed("deviceId", "_id")
df = df.fillna(0)
df = df.withColumn("lastUpdatedAt", current_timestamp())
# df = df.withColumn("features", expr("transform(features, x -> cast(x as double))"))


# In[62]:


all_columns = df.columns
df = df.select(*all_columns, array([col(x) for x in device_features]).alias('device_features'))


# In[63]:


dev_mongo_columns = ['_id',
 'device_emb',
 'lastUpdatedAt',
 'device_features']


# In[64]:


#dump data to mongo
configs['mongo_details']['tbl'] = 'deviceEmbeddingCollection'
df.select(*dev_mongo_columns).write.mode("overwrite").format('mongodb').option('database', configs['mongo_details']['db']).option('collection', configs['mongo_details']['tbl']).save()
    


# In[65]:


df_embedding_m = df_embedding
df_embedding_m = df_embedding_m.join(joined_df.select('hashid',*news_features).distinct(),
                                     df_embedding_m.hid == joined_df.hashid,"inner")
df_embedding_m = df_embedding_m.distinct()
df_embedding_m = df_embedding_m.withColumnRenamed("hid", "_id")
df_embedding_m = df_embedding_m.fillna(0)
df_embedding_m = df_embedding_m.withColumn("lastUpdatedAt", current_timestamp())


# In[66]:


all_columns = df_embedding_m.columns
df_embedding_m = df_embedding_m.select(*all_columns, array([col(x) for x in news_features]).alias('news_features'))
df_embedding_m.printSchema()


# In[67]:


news_mongo_columns = ['_id',
 'embedding',
 'hashid',
 'lastUpdatedAt',
 'news_features']


# In[68]:


configs['mongo_details']['tbl_1'] = 'newsEmbeddingCollection'
df_embedding_m.select(*news_mongo_columns).write.mode('overwrite').format('mongodb').option('database', configs['mongo_details']['db']).option('collection', configs['mongo_details']['tbl_1']).save()
feature_insert_time = time.time() - feature_insert_st_time
Log("Device and News features Inserted")


# In[72]:


# df.select('_id').distinct().take(5)
Log("Main Query took %s percent of time" % round(100 * main_query_time / (time.time() - start_time), 3))
# Log("Model Training took %s percent of time" % round(100 * training_time / (time.time() - start_time), 3))
Log("Feature Insertion took %s percent of time" % round(100 * feature_insert_time / (time.time() - start_time), 3))
Log("Exectution finished successfully, in : %s minutes, %s seconds " % (int((time.time() - start_time) / 60), int((time.time() - start_time) % 60)), flag=True)

