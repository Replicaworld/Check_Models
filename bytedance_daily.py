#depends on latest parquet
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T
import pyspark.sql.window as W
import sys
import time
import os
import datetime
import subprocess
from dateutil.relativedelta import relativedelta

#config
pilot_run='N'
execute_for_last_n_days=1

if (pilot_run=="Y"):
    event_out_loc="gs://pvtrough_asia_south1/tmp/event_incremental/"
    content_out_loc="gs://pvtrough_asia_south1/tmp/content/"
    user_out_loc="gs://pvtrough_asia_south1/tmp/user/"
else:
    event_out_loc="gs://historical_training/event_incremental/"
    content_out_loc="gs://historical_training/content/"
    user_out_loc="gs://historical_training/user/"    
tmp_content_loc="gs://pvtrough_asia_south1/tmp/news_gcs_copy/"
#config
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

conf = SparkConf().setAll([('spark.jars.packages','com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.27.0')]).setAppName("bytedance_daily")
sc = SparkContext.getOrCreate(conf=conf)
spark=SparkSession(sc)

bucket='pvtrough_asia_south1/tmp/tmpgcs_for_bq'
spark.conf.set('temporaryGcsBucket', bucket)
spark.conf.set("viewsEnabled","true")
spark.conf.set("materializationDataset",'tmp')
#upstream
distr_map=F.broadcast(spark.read.format('bigquery').option("query","SELECT distinct state_name,district_code as districtCode,district_name FROM `inshorts-1374.public_adhoc.public_state_dist_subd_map`").load().cache())
author_map=F.broadcast(spark.read.format('bigquery').option("query","SELECT _id as hashid,author,countryCode FROM `inshorts-1374.inshorts_mongo_data_tech.news_data`").load().cache())
# only diff with realtime logic may be customField1 starts getting values as 'My Feed' but we will be sending them as Others
category_expr=F.expr('''(case when eventname='Deck Card View' then 'Deck'
--when eventname='TimeSpent-Front' and customfield1 is not null then (case when customfield1='similarNews' then 'Similar News' else 'Others' end)
when upper(categoryWhenEventHappened) in ('NULL','','UNKNOWN') or categoryWhenEventHappened is null then 'Unknown'
when categoryWhenEventHappened in ('my feed','My Feed') then 'My Feed'
when categoryWhenEventHappened in ('all news','All News') then 'All News'
when categoryWhenEventHappened in ('unread','Unread') then 'Unread'
when categoryWhenEventHappened in ('top stories','Top Stories') then 'Top Stories'
when categoryWhenEventHappened in ('trending','Trending') then 'Trending'
when categoryWhenEventHappened in ('bookmark','Bookmark') then 'Bookmark'
when categoryWhenEventHappened in ('Bookmarks') then 'Bookmarks'
else 'Others' end) as categoryWhenEventHappened''')
traffic_src_expr=F.expr('''coalesce(feedtype,"") as traffic_source''')
def purge(path):
    if(pilot_run=='N'):
        for i in subprocess.check_output("gsutil ls -R {0}".format(path), shell=True).decode('utf-8').split('\n'):
            if(delete_dt in i):
                subprocess.check_output("gsutil rm -r {0}/{1}".format(path,delete_dt), shell=True)
                break
            
def union_process(gcs_path_prefix,dt_logic):
    event_df=((spark.read.parquet(*[path+"/timeSpentFrontEvents" for path in gcs_path_prefix])
             .select(F.col('deviceId'),F.col('eventName').alias('event_type'),F.col('eventTimestamp'),F.col('hashId'),category_expr,
                     F.col('cardViewPosition'),F.col('overallTimeSpent'),F.lit(None).cast('string').alias('searchTerm'),F.col('platform'),
                     F.col('appVersion'),F.col('osVersion'),F.col('model'),F.col('networkType'),F.col('locality'),F.col('districtCode'),
                     F.col('appname'),traffic_src_expr,F.lit(None).cast('double').alias('video_duration'),F.lit(None).cast('double').alias('video_watch_duration'))
    .filter((F.col('event_type').isin(['TimeSpent-Front'])) & (dt_logic) & (F.upper(F.coalesce(F.col('appname'),F.lit(''))).isin(['','UNKNOWN','INSHORTS'])))
                    .drop('appname'))
    .unionAll(
        spark.read.parquet(*[path+"/deckEvents" for path in gcs_path_prefix])
        .filter((F.col('eventName') == 'Deck Card View') & (F.col('type')=='DECK_CONTENT') & (dt_logic))
        .withColumn('customfield1',F.lit(None).cast('string'))
     .select(F.col('deviceId'),F.lit('TimeSpent-Front').alias('event_type'),F.col('eventTimestamp'),F.col('hashId'),category_expr
    ,F.col('position').alias('cardViewPosition'),(F.col('timespent')/1000).alias('overallTimeSpent'),F.lit(None).cast('string').alias('searchTerm'),F.col('platform')
    ,F.col('appVersion'),F.col('osVersion'),F.col('model'),F.col('networkType'),F.col('locality'),F.col('districtCode'),traffic_src_expr,F.lit(None).cast('double').alias('video_duration'),F.lit(None).cast('double').alias('video_watch_duration'))  )
    .unionAll(
        spark.read.parquet(*[path+"/otherEvents" for path in gcs_path_prefix])
        .select(F.col('deviceId'),F.col('eventName').alias('event_type'),F.col('eventTimestamp'),F.col('hashId'),category_expr,F.col('cardViewPosition'),
            F.col('overallTimeSpent'),F.lit(None).cast('string').alias('searchTerm'),F.col('platform'),F.col('appVersion'),F.col('osVersion'),F.col('model')
            ,F.col('networkType'),F.col('locality'),F.col('districtCode'),F.col('appname'),traffic_src_expr,F.col('customfield2').cast('double').alias('video_duration'),F.col('customfield3').cast('double').alias('video_watch_duration'))
    .filter((F.col('event_type').isin(['News Bookmarked','News Unbookmarked','News Shared','TimeSpent-Back','Video Ended'])) & (dt_logic) & (F.upper(F.coalesce(F.col('appname'),F.lit(''))).isin(['','UNKNOWN','INSHORTS'])))
    .drop('appname'))
    .unionAll(
        spark.read.parquet(*[path+"/searchEvents" for path in gcs_path_prefix])
        .withColumn('customfield1',F.lit(None).cast('string'))
        .select(F.col('deviceId'),F.col('eventName').alias('event_type'),F.col('eventTimestamp'),F.col('hashId'),category_expr,
                F.lit(None).alias('cardViewPosition'),F.lit(None).alias('overallTimeSpent'),F.col('searchTerm'),F.col('platform'),
                F.col('appVersion'),F.col('osVersion'),F.col('model'),F.col('networkType'),F.lit(None).alias('locality'),
F.lit(None).alias('districtCode'),traffic_src_expr,F.lit(None).cast('double').alias('video_duration'),F.lit(None).cast('double').alias('video_watch_duration')
               )
    .filter((F.col('event_type').isin(['Search'])) & (dt_logic) )
             .drop('appname'))
    )

    return (event_df.join(spark.read.parquet(tmp_content_loc).select(F.col('_id').alias('hashid'),F.col('videoLength'),F.col('similarGroupId'))
                          ,on=['hashid'],how="left")
            .withColumn('hashid',F.when(F.isnull(F.col('similarGroupId')),F.col('hashid')).otherwise(F.col('similarGroupId')))
            .withColumn('video_duration',F.col('videoLength'))
           )


gcs_path_prefix=[]
for i in range(execute_for_last_n_days):
    run_dt=datetime.date.today() + relativedelta(days=i)
    master_dt=run_dt- relativedelta(days=1)
    # gcs_path_prefix.append("gs://nis-segment-datasource-v3/processed/"+master_dt.strftime("%Y/%m/%d"))
    delete_dt=(master_dt- relativedelta(days=7)).strftime("%Y/%m/%d")
    purge("gs://historical_training/user")
    purge("gs://historical_training/content")
    # if(i==0):
    tz_boundary=int(time.mktime((run_dt+relativedelta(hours=-5,minutes=-30)).timetuple())*1000)
    
    #downstream
    #news tbl
    spark.read.format('bigquery').option('query',
    '''select news_df._id,is_recommendable,title,content,newsType,author,
(case when date(timestamp_millis(createdAt),'Asia/Calcutta')>='2024-03-15' then 
        (case when publisherUserType='CREATOR' then split(array_to_string([case when array_to_string(categories,',')!='' then array_to_string(categories,',') end,'creator'],','),',')
        when publisherUserType='VENDOR' then split(array_to_string([case when array_to_string(categories,',')!='' then array_to_string(categories,',') end,'vendor'],','),',') end
        )
    else (case when vendor!='INSHORTS' then split(array_to_string([case when array_to_string(categories,',')!='' then array_to_string(categories,',') end,'vendor'],','),',')
          else  categories end
          )
    end) as categories,hashtags,imageUrl,videoUrl,available_location,createdAt, updatedAt,newsLanguage,sourceName,with_video,is_ad,similarGroupId,videolength from
        (select _id,split(_id,'-')[offset(0)] as grp_id,(case when LimitToSpecificAudience=True or deleted=True or coalesce(trim(sponsoredBy),'')!=''or (coalesce(publisherUserType,'')!='INSHORTS' and newsType='VIDEO_NEWS') then 0 else 1 end) as  is_recommendable,title,content,newsType,author,categories,hashtags,imageUrl,videoUrl,
        (case when newslocationlevel='SUB_DISTRICT' then newsSubDistrict when newslocationlevel='DISTRICT' or newslocationlevel is null then newsDistrict when newslocationlevel='STATE' then newsState end) as available_location,unix_millis(createdAt) as createdAt,unix_millis(updatedAt) as updatedAt,newsLanguage,sourceName,(case when trim(videoUrl)!='' then 1 else 0 end) as with_video,(case when coalesce(trim(sponsoredBy),'')='' then 0 else 1 end) as is_ad
    ,(case when coalesce(trim(similarGroupId),'')='' then null else similarGroupId end) as similarGroupId,(case when coalesce(trim(videoLength),'')='' then null when videoLength like ('%:%') then (60*safe_cast(split(videolength,':')[OFFSET(0)] as int64))+safe_cast(split(videolength,':')[OFFSET(1)] as int64) else cast(videolength as float64) end) as videolength,publisherUserType
        from inshorts_mongo_data_tech.news_data 
        where coalesce(isAutoGen,False)=False 
        -- and unix_millis(createdAt)<{0} 
        )news_df
        left join
        (select _id,vendor from inshorts_raw_data.news_group_data)grp_df
        on (grp_df._id=news_df.grp_id)'''
    ).load().coalesce(1).write.parquet(tmp_content_loc,mode="overwrite")

    (spark.read.parquet(tmp_content_loc).filter("createdAt<{0}".format(tz_boundary))
     .withColumn('_id',F.when(F.isnull(F.col('similarGroupId')),F.col('_id')).otherwise(F.col('similarGroupId'))).drop('videolength','similarGroupId')
     .withColumn('rn',F.row_number().over(W.Window.partitionBy(F.col('_id')).orderBy(F.col('createdAt').desc()) )).filter("rn=1").drop('rn')
    .coalesce(1).write.json(content_out_loc+master_dt.strftime("%Y/%m/%d")+"/",compression='gzip',mode="overwrite")
    )
    #news tbl
    
    strt_dt="2023-06-26"
    dt_logic=(F.date_format(F.timestamp_seconds(F.col('eventtimestamp')/1000),"yyyy-MM-dd")>strt_dt) & (F.col('eventtimestamp')<tz_boundary)
    gcs_path_prefix=["gs://nis-segment-datasource-v3/processed/"+master_dt.strftime("%Y/%m/%d")]
    # elif(execute_for_last_n_days-1):                
    prev_dt=master_dt- relativedelta(days=1)
    prev_dt_gcs_path_prefix=["gs://nis-segment-datasource-v3/processed/"+prev_dt.strftime("%Y/%m/%d")]
    prev_dt_logic=(F.col('eventtimestamp')>=int(time.mktime((master_dt+relativedelta(hours=-5,minutes=-30)).timetuple())*1000)) & (F.col('eventtimestamp')<int(time.mktime((run_dt+relativedelta(hours=-5,minutes=-30)).timetuple())*1000))
    #comment from next run
    # union_proc1=union_process(gcs_path_prefix,dt_logic)
    # traffic_src_expr=F.expr('''cast(null as string) as traffic_source''')
    # union_proc2=union_process(prev_dt_gcs_path_prefix,prev_dt_logic)
    # union_eve_tbl=union_proc1.unionAll(union_proc2)        
    #comment from next run
    union_eve_tbl=union_process(gcs_path_prefix,dt_logic).unionAll(union_process(prev_dt_gcs_path_prefix,prev_dt_logic))    
    union_eve_tbl.write.parquet("gs://historical_training/temp",mode='overwrite')
    union_eve_tbl=spark.read.parquet("gs://historical_training/temp")
    #downstream
    (union_eve_tbl.join(distr_map,on=['districtCode'],how="left").join(author_map,on=['hashid'],how="left")
     .select(F.col('deviceId'),F.col('event_type'),F.col('eventTimestamp'),F.col('hashId'),F.col('categoryWhenEventHappened'),
    F.col('cardViewPosition'),F.col('overallTimeSpent'),F.col('author'),F.col('searchTerm'),F.col('platform'),F.col('appVersion'),F.col('model'),
    F.col('osVersion'),F.col('networkType'),F.col('countryCode').alias('country'),F.col('state_name').alias('state'),
             F.col('locality'),F.col('district_name').alias('district'),F.col('traffic_source'),F.col('video_watch_duration'),F.col('video_duration'))
     .coalesce(20*execute_for_last_n_days)
     .write.json(event_out_loc+master_dt.strftime("%Y/%m/%d")+"/",compression='gzip',mode="overwrite")
    )


    #upstream
    os.system("python /home/Shrey/schedules_and_shell_scripts/mongo_to_bq_json/generic_mongo_to_gcs.py /home/Shrey/schedules_and_shell_scripts/mongo_to_bq_json/device_relevancy.json")
    os.system("python /home/Shrey/schedules_and_shell_scripts/mongo_to_bq_json/generic_etl_device_location.py /home/Shrey/schedules_and_shell_scripts/mongo_to_bq_json/inshorts_device_location.json")    
    #downstream
    spark.read.format('bigquery').option('query','''
(select 
    lastknownsubadminarea, `model`,coalesce(district_from_dev_loc,district) as district,tenant_i,last_active_at,networkType,platform,os_version,created_datetime
      ,app_updatedat,dev_master.deviceid,group_code,relevancy from
      (SELECT lastknownsubadminarea,device_name as `model`,district,tenant_i,unix_millis(last_active_at) as last_active_at,networkType,platform,os_version,unix_millis(created_datetime) as created_datetime
      ,unix_millis(cast(app_updated_at as timestamp)) as app_updatedat,deviceid,devicehashcode as group_code
      FROM `inshorts-1374.inshorts_scheduled.device_master` where app_name='inshorts' 
       and unix_millis(created_datetime)<{0}
      )dev_master
      left join
      (select _id,struct(RED,GREEN,YELLOW) as relevancy from
        (select _id,color,array_agg(topic order by synctime desc) as topic_arr from
          (select _id,topic,color,synctime,row_number() over (partition by _id,topic order by synctime desc) as rn from
            (SELECT _id,lower(enRelevancy.element.tagid) as topic
            ,upper(enRelevancy.element.relevancyenum) as color
            ,enRelevancy.element.synctime as synctime
            FROM `inshorts-1374.inshorts_raw_data.device_relevancy_tags` 
            CROSS JOIN UNNEST(enRelevancy.list) as enRelevancy
            )
          )
        where rn=1
        group by _id,color
        )
      pivot (
        any_value(topic_arr)
        for color in ('RED','YELLOW','GREEN')
      ))relevancy_info
      on (dev_master.deviceid=relevancy_info._id) 
    left join
    (select (case when coalesce(trim(subDistrictCode),'')!='' then subDistrictCode
      when coalesce(trim(DistrictCode),'')!='' then DistrictCode
        when coalesce(trim(stateCode),'')!='' then stateCode
      end) as district_from_dev_loc,_id,deviceid
      from `inshorts_raw_data.inshorts_device_location`
    where appname='inshorts'
    )device_location
    on (device_location._id=dev_master.deviceid)
  )
    '''.format(tz_boundary)).load().coalesce(4).write.json(user_out_loc+master_dt.strftime("%Y/%m/%d")+"/",compression='gzip',mode="overwrite")


    # purge("gs://historical_training/event_incremental")