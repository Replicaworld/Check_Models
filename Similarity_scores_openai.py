#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[158]:
print('Starting of Similarity Score Update')
import logging
import datetime
import time
import os

import time
start_time_all = time.time()

import os
import sys

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

log_path = '/home/DS_Live_Trials/similarity_score_update_logs/'
print(os.getcwd())
if not os.path.exists(log_path):
    os.makedirs(log_path)


# In[2]:


import json
PROJECT = "inshorts-1374"


config_file = '/home/karanverma/files/news_mongo_to_bq.json'
# print(config_file)
configs=json.load(open(config_file,'r'))

configs['mongo_details']['hosts_list'] = ['172.16.15.6']
configs['mongo_details']['db'] = 'svd'
configs['mongo_details']['tbl'] = 'newsSimSpentCollection'
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


# In[4]:


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
import pyspark.sql.functions as f
from pyspark.sql import Row
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


# In[3]:


import time
start_time = time.time()


# In[5]:


# In[160]:


def get_path(dates, prefix, padding=None):
    dates_ = dates + []
    if padding:
        st_date, ed_date = sorted(dates)[0], sorted(dates)[-1]
        for i in range(1, 4):
            d = (datetime.datetime.strptime(ed_date, date_fmt) + datetime.timedelta(days=i)).strftime(date_fmt)
            if d < datetime.datetime.today().strftime(date_fmt):
                dates_.append(d)
    dates_ = list(set(dates_) - set(["2019/02/17", "2019/02/18", "2019/05/28", "2019/06/03", "2019/07/02", "2019/07/03", "2019/07/04", "2019/11/13", "2019/11/14", "2020/02/22", "2020/03/31", "2020/04/16", "2020/04/18", "2020/05/11", "2021/05/13"]))
    paths = []
    for date in dates_:
        base_path = NIS_DATA_BASE_PATH
        if date < "2018/06/26":
            base_path = NIS_OLD_DATA_BASE_PATH
        paths.append(base_path + date + "/" + prefix + "/*.parquet")
    return paths

date_fmt = "%Y/%m/%d"
month_fmt = "%Y/%m"

def millis2date(x):
    try:
        if x < 15000000000:
            return datetime.datetime.fromtimestamp(x).strftime(date_fmt)
        else:
            return datetime.datetime.fromtimestamp(x / 1000.).strftime(date_fmt)
    except:
        return "1970/01/01"

def millis2month(x):
    try:
        if x < 15000000000:
            return datetime.datetime.fromtimestamp(x).strftime(month_fmt)
        else:
            return datetime.datetime.fromtimestamp(x / 1000.).strftime(month_fmt)
    except:
        return "1970/01"

millis2date_udf = F.udf(millis2date, StringType())
millis2month_udf = F.udf(millis2month, StringType())

def divide_maps(d1, d2):
    keys = set(d1.keys()).intersection(set(d2.keys()))
    res = {}
    for k in keys:
        res[k] = d1[k] * 1. / (d2[k] + 1e-10)
    return res

def timediff(y, x, date_fmt="%Y/%m/%d"): 
    end = datetime.datetime.strptime(y, date_fmt)
    start = datetime.datetime.strptime(x, date_fmt)
    delta = (end - start).days
    return delta

def monthdiff(y, x, month_fmt="%Y/%m"): 
    millis = y - x
    delta = millis / (1000 * 3600 * 24 * 30)
    return delta

timediff_udf = udf(timediff, IntegerType())
monthdiff_udf = udf(monthdiff, IntegerType())

def filter_platform(data, platform=None):
    if platform == "ANDROID":
        data = data.filter(data.platform == "ANDROID")
    elif platform == "IOS":
        data = data.filter(data.platform != "ANDROID")
    return data

def filter_category(data, categories=None):
    if categories:
        data = data.filter(data.categoryWhenEventHappened.isin(categories))
    return data

def filter_tenant(data, tenant=None):
    if tenant in ['hi', 'HINDI']:
        data = data.filter(data.tenant.isin(['hi', 'HINDI', 'Hindi', 'hindi']))
    elif tenant in ['en', 'ENGLISH']:
        data = data.filter(~data.tenant.isin(['hi', 'HINDI', 'Hindi', 'hindi']))
    return data


def filter_app(data, app_name=None):
    if 'appName' in data.columns:
        if app_name:
            data = data.filter(data.appName == app_name)
        else:
            data = data.filter((data.appName != "mini") & (data.appName != "crux"))
    return data


# In[6]:


# In[161]:


def get_raw_path(date, hours=None):
    paths = []
    base_path = NIS_RAW_DATA_BASE_PATH + date
    if not hours:
        return base_path + "/*/*.gz"
    for hour in hours:
        paths.append(base_path + "/" + str(hour).zfill(2) + "/*.gz")
    return ",".join(paths)

def process_raw_data(paths):
    def view_data_filters(x):
        x = x['properties']
        deviceid_filter = ('deviceId' in x) and (x['deviceId'] != '')
        time_filter = ('timeSpent' in x) and (int(x['timeSpent']) <= 100) and (int(x['timeSpent']) >= 0)
        return deviceid_filter and time_filter

    try:
        rdd = sc.textFile(paths)             .map(json.loads)             .filter(lambda x: "batch" in x).flatMap(lambda x: x["batch"])             .filter(lambda x: ("event" in x) and (x["event"].lower() == "timespent-front"))             .filter(view_data_filters)             .map(lambda x: x['properties'])
        view_data = rdd.map(lambda x: (x['deviceId'], x['hashId'][:-2], x['timeSpent']))             .toDF(['deviceId', 'hashId', 'timeSpent'])
        
        view_data = view_data.filter(view_data.timeSpent.isNotNull())
        #view_data = view_data.filter((getHashBucketUDF(view_data.deviceId) >= 16) & (getHashBucketUDF(view_data.deviceId) <= 25))
        view_data = view_data.groupby(view_data.deviceId, view_data.hashId)                              .agg(F.max(view_data.timeSpent).alias('overallTimeSpent'))
        
        return view_data
    except Exception as e:
        logger.warning("Error processing data: " + str(e))


# In[162]:


# In[7]:


from pymongo import MongoClient
from pymongo import UpdateOne

def getMongoClient():
    hosts = ['171.16.11.97','171.16.11.96','171.16.11.94']
    host = ','.join([i + ":27017" for i in hosts])
    conn_url = 'mongodb://root:superman@' + host
    client = MongoClient(conn_url)
    return client

def getMongoColl(db_name, coll_name, hosts=[]):
    host = ','.join([i + ":27017" for i in hosts])
    conn_url = 'mongodb://root:superman@' + host + '/' + db_name
    client = MongoClient(conn_url, minPoolSize=10, maxPoolSize=None)
    return client[db_name][coll_name]

def getNewsInHashIds(hashIds):
    news_coll = getMongoColl(db_name='nis-news', 
                           coll_name='News', 
                           hosts=['172.16.11.196', '172.16.11.195', '172.16.11.194'])
    
    newsMap = {}
    array = [t for t in news_coll.find({'_id' : {"$in" : hashIds}})]
    
    for i in range(len(array)):
        newsMap[array[i]['_id']] = array[i]
        
    return newsMap

def getNewsInDates(begin, end):
    news_coll = getMongoColl(db_name='nis-news', 
                           coll_name='News', 
                           hosts=['172.16.11.196', '172.16.11.195', '172.16.11.194'])
    
    newsMap = {}
    array = [t for t in news_coll.find({'createdAt' : {"$gt" : begin, "$lt" : end}})]
    
    for i in range(len(array)):
        newsMap[array[i]['_id']] = array[i]
        
    return newsMap
def getNewsData(d1, d2):
    newsMap = getNewsInDates(d1, d2)
    hashIdList = list(newsMap.keys())

    hashIdsWithFilter = []
    for h in hashIdList:
        if 'newsLanguage' in newsMap[h] and newsMap[h]['newsLanguage'] == 'english' and newsMap[h]['publishGroupList'][0]['countryCode'] == 'IN':
            hashIdsWithFilter.append(h.split('-')[0])
    
    return hashIdsWithFilter, newsMap


# In[163]:


# In[8]:


import logging

logger = logging.getLogger(str(datetime.datetime.today().date()))
hdlr = logging.FileHandler(
    log_path + str(datetime.datetime.today().date()) + '.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.DEBUG)

def Log(s, flag=True):
    if flag:
        logger.info(s)
        print(s)

Log('----', flag=True)


# In[9]:


# In[164]:


n_days = 1

st_date = datetime.datetime.now() - datetime.timedelta(days=n_days+2)

date_fmt = "%Y/%m/%d"

dates = [(st_date + datetime.timedelta(days=i)) for i in range(n_days+1)]
dates.sort()
#print(dates)
dates_str = [date.strftime(date_fmt) for date in dates]
print(dates_str)
# millisMin = dates[0].timestamp() * 1000
# millisMax = (dates[-1] + datetime.timedelta(days=1)).timestamp() * 1000

hashIdsWithFilter, newsMap = getNewsData(dates[0], datetime.datetime.now())


# In[165]:


# In[10]:


print(dates[0], datetime.datetime.now())


# In[11]:


#need to comment this
NIS_RAW_DATA_BASE_PATH = "gs://inshorts-segment-raw/data/segment-raw-v5a/"
#path = get_raw_path(datetime.datetime(2023, 2, 22).strftime("%Y/%m/%d"))

path = get_raw_path(datetime.datetime.now().strftime("%Y/%m/%d"))
today_data = process_raw_data(path)
today_data = today_data.filter(today_data.hashId.isin(hashIdsWithFilter))


# In[12]:


last_date = datetime.datetime.now() - datetime.timedelta(days=1)
# print(last_date.strftime("%Y/%m/%d"))
path = get_raw_path(last_date.strftime("%Y/%m/%d"))
yesterday_data = process_raw_data(path)
yesterday_data = yesterday_data.filter(yesterday_data.hashId.isin(hashIdsWithFilter))


# In[13]:


datestr = [d.strftime(date_fmt) for d in dates]
paths = get_path(datestr, 'timeSpentFrontEvents')

data = sqlContext.read.parquet(*paths)

data = filter_app(data, app_name=None)
data = filter_tenant(data, tenant='en')

# data = data.filter((data.eventTimestamp > millisMin) & (data.eventTimestamp < millisMax)) (F.split(data.hashId, '-')[0]).alias('hashId')) 
data = data.select(data.deviceId, data.overallTimeSpent,  data.hashId)      .groupby('deviceId', 'hashId')         .agg(F.max('overallTimeSpent').alias('overallTimeSpent'))

data = data.filter(data.hashId.isin(hashIdsWithFilter))


# In[14]:


data = data.union(today_data)
data = data.union(yesterday_data)
data=data.filter(data.overallTimeSpent<100)
Log("Data Loaded")


# In[15]:


# suffixes = ['CV_High', 'CV_Med', 'CV_Low']
# for suffix in suffixes:
start_time=time.time()
suffix = 'News to News Similarity model' 
Log("Calculating for %s"%suffix)

# data_filtered = data.filter(F.udf(lambda hashId, overallTimeSpent: overallTimeSpent > (2.5 * newsMeanMap[hashId][0]/newsMeanMap[hashId][1]), BooleanType())('hashId', 'overallTimeSpent'))
# data_filtered=data_filtered.join(df_temp,["deviceId"],"inner")


# In[16]:


hids_1 = [h+"-1" for h in hashIdsWithFilter]


# In[17]:


data_grouped = data.select('hashId', 'overallTimeSpent').groupBy('hashId').mean()
data_grouped = data_grouped.toPandas()
# data_grouped


# In[ ]:


from pymongo import UpdateOne
import datetime
import logging
import os
import shutil
import sys
import json

import numpy as np
from pymongo import MongoClient
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql import functions as F
from pyspark.sql.types import *
#from Utils import *
from tqdm import tqdm

def getMongoClient():
    hosts = ['171.16.11.97','171.16.11.96','171.16.11.94']
    host = ','.join([i + ":27017" for i in hosts])
    conn_url = 'mongodb://root:superman@' + host
    client = MongoClient(conn_url)
    return client
#trial_algo_suffix = 'CV_High'

def getMongoClient_openAI():
    hosts = ['172.16.15.6']
    host = ','.join([i + ":27017" for i in hosts])
    conn_url = 'mongodb://root:superman@' + host
    client = MongoClient(conn_url)
    return client



mongoClient = getMongoClient()
mongoClient_openAI = getMongoClient_openAI()
trialDB = mongoClient['trainDB']
svdDB = mongoClient_openAI['svd']

newsVectorCollection = svdDB['OAInewsEmbeddings']
newsSimSpentCollection = svdDB['newsSimSpentCollection']



def getNewsVectorsFromMongo(hashIds):
    newsVectorMap = {}
    cursor = newsVectorCollection.find({"_id" : { "$in" : hashIds }})

    for c in cursor:
        newsVectorMap[c['_id']] = np.array(c['embedding'])

    return newsVectorMap

def insertNewsVectorsInMongo(newsVectors):
    for hashId in newsVectors:
        key = {"_id" : hashId}
        newsData = {"$set" : {"embedding" : list(newsVectors[hashId])}}
        newsVectorCollection.update_one(key, newsData, upsert=True)

def getNewsTSpentFromMongo(hashIds):
    newsTSpentMap = {}
    cursor = newsTSpentCollection.find({"_id" : {"$in" : hashIds }})

    for c in cursor:
        data = {}
        for cluster in c['tSpent']:
            data[int(cluster)] = c['tSpent'][cluster]
        newsTSpentMap[c['_id']] = data

    return newsTSpentMap

def insertNewsTSpentInMongo(newsTSpentMap):
    for hashId in newsTSpentMap:
        key = {"_id" : hashId}
        data = {}
        for cluster in newsTSpentMap[hashId]:
            data[str(cluster)] = newsTSpentMap[hashId][cluster]
        tSpentData = {"$set" : {"tSpent" : data}, "$setOnInsert": {"createdAt": datetime.datetime.now()}}
        newsTSpentCollection.update_one(key, tSpentData, upsert=True)



# In[19]:


vectors = getNewsVectorsFromMongo(hids_1)


# In[20]:


embedding_data = pd.DataFrame(vectors).T


# In[21]:


import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(embedding_data)
similarity_df = pd.DataFrame(cosine_sim, columns=embedding_data.index, index=embedding_data.index)
# similarity_df['rsu'] = df.sum(axis=1, numeric_only=True)
# similarity_df


# In[22]:


normalized_similarity_df = similarity_df.div(similarity_df.sum(axis=1), axis=0)
# normalized_similarity_df


# In[23]:


data_grouped['hashId'] += "-1"
# data_grouped


# In[24]:


data_grouped.index = data_grouped.hashId
timespent_map = data_grouped.T.to_dict()
# timespent_map


# In[25]:


tspent_notfound_count = 0
mongo_df = normalized_similarity_df.copy()
for c in normalized_similarity_df.columns:
    if c not in timespent_map.keys():
        tspent_notfound_count +=1
        print(c)
        continue
    mongo_df[c] *= timespent_map[c]['avg(overallTimeSpent)']


# In[26]:


mongo_df['_id'] = mongo_df.index


# In[95]:


# mongo_df #['4m3zd8qp-1']


# In[27]:


df_sim_spark = (sqlContext.createDataFrame(mongo_df))


# In[29]:


mongo_columns = df_sim_spark.columns
mongo_columns.remove('_id')


# In[34]:

from pyspark.sql.functions import current_timestamp
df_sim_mongo = df_sim_spark.withColumn("newsSimMap", f.to_json(f.struct(mongo_columns))).select('_id', 'newsSimMap')
df_sim_mongo = df_sim_mongo.withColumn("lastUpdatedAt", current_timestamp())

# In[35]:


# configs['mongo_details']['tbl'] = 'newsEmbeddingCollection'

Log("Inserting News Score Data")
df_sim_mongo.write.mode('overwrite').format('mongodb').option('database', 
                                              configs['mongo_details']['db']).option('collection', 
                                                                                 configs['mongo_details']['tbl']).save()
    



# In[ ]:


Log("Exectution finished successfully, in : %s minutes, %s seconds " % (int((time.time() - start_time) / 60), int((time.time() - start_time) % 60)), flag=True)

