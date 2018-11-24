import pandas as pd
import synonyms
import re
pd.set_option('display.max_columns', None)

# load data
trainurl = r'/home/remoteuser/aiya/Titanic/OGeek/oppo_round1_train_20180929.txt'
testurl = r'/home/remoteuser/aiya/Titanic/OGeek/oppo_round1_test_A_20180929.txt'
valiurl = r'/home/remoteuser/aiya/Titanic/OGeek/oppo_round1_vali_20180929.txt'
train = pd.read_csv(trainurl, sep='\t', header=None, names=['prefix', 'query_prediction', 'title', 'tag', 'label'], encoding='utf-8').astype(str)
test = pd.read_csv(testurl, sep='\t', header=None, names=['prefix', 'query_prediction', 'title', 'tag'], encoding='utf-8').astype(str)
vali = pd.read_csv(valiurl, sep='\t', header=None, names=['prefix', 'query_prediction', 'title', 'tag', 'label'], encoding='utf-8').astype(str)

#split query_prediction to list
def split_prediction(text):
    if pd.isnull(text):
        return []
    return [s.strip() for s in text.replace("{", "").replace("}", "").split(", ")]

train['pred_list'] = train['query_prediction'].apply(split_prediction)
test['pred_list'] = test['query_prediction'].apply(split_prediction)
vali['pred_list'] = vali['query_prediction'].apply(split_prediction)


# 标签处理：1.统计样本标签，删除非常规标签 2.将标签类型转为int
## print(train['label'].value_counts()) # 0:1255803 1:744195 音乐：1
## print(vali['label'].value_counts())  # 0:31415   1:18585
train = train[train['label'] != '音乐']
train['label'] = train['label'].apply(lambda x: int(x))
vali['label'] = vali['label'].apply(lambda x: int(x))


# 统计各个prefix、fitle、tag及两两组合对应的点击率
items = ['prefix', 'title', 'tag']
for item in items:
    temp = train.groupby(item, as_index=False)['label'].agg({item+'_click':'sum', item+'_count':'count'})
    temp[item+'_ctr'] = temp[item+'_click']/(temp[item+'_count'])
    train = pd.merge(train, temp, on=item, how='left')
    vali = pd.merge(vali, temp, on=item, how='left')
    test = pd.merge(test, temp, on=item, how='left')

for i in range(len(items)):
    for j in range(i+1, len(items)):
        item_g = [items[i], items[j]]
        temp = train.groupby(item_g, as_index=False)['label'].agg({'_'.join(item_g)+'_click': 'sum','_'.join(item_g)+'count':'count'})
        temp['_'.join(item_g)+'_ctr'] = temp['_'.join(item_g)+'_click']/(temp['_'.join(item_g)+'count']+3)
        train = pd.merge(train, temp, on=item_g, how='left')
        vali = pd.merge(vali, temp, on=item_g, how='left')
        test = pd.merge(test, temp, on=item_g, how='left')


def myslit():
    pass

# 计算pred_list中各个item和title之间的相似度
def similarity(df, url):
    df_pred_lists = df['pred_list']
    df_title = df['title']
    samples = df.shape[0]#得到行数
    max_similarities = [] #相似度
    max_items = []  # 相似度最大的prefix_item
    probability_items = []  #概率最高的prefix_item
    probability_similarities = []
    average_wighted = []
    max_wighted = []
    prob_wighted = []
    pattern = '(.*?)": .*'
    pattern2 = '.*": "(.*?)".*'
    for i in range(samples):#遍历行
        max_similarity = 0.0
        max_probability = 0.0
        aver_wight = 0
        n = 0
        pred_item = ' '
        prefix_item = ' '
        items_num = len(df_pred_lists[i])#遍历每个prediction_query
        for j in range(items_num):
            item_pred_prefix = df_pred_lists[i][j].replace("\"", "").split(':')
            #aver_wight = 0
            if len(item_pred_prefix) >= 2:
                item_pred = re.findall(pattern, df_pred_lists[i][j])[0]
                item_percent = re.findall(pattern2, df_pred_lists[i][j])[0]
                item_title = df_title[i]
                print('index:', i, 'column:', j)
                #print(item_pred, item_percent)
                similarity = synonyms.compare(item_pred, item_title)
                aver_wight += similarity * float(item_percent)
                n = n+1
                #print(similarity)
                if similarity > max_similarity:
                    max_similarity = similarity
                    #max_pred_index = j
                    pred_item = item_pred
                    withted_max_item = float(item_percent) * similarity
                if float(item_percent) > max_probability:
                    max_probability = float(item_percent)
                    max_probability_similarity = similarity
                    #max_probability_index = j
                    prefix_item = item_pred
                    withted_prob_item = float(item_percent) * similarity
        max_items.append(pred_item)
        max_similarities.append(max_similarity)
        max_wighted.append(withted_max_item)


        probability_items.append(prefix_item)
        probability_similarities.append(max_probability_similarity)
        prob_wighted.append(withted_prob_item)
        
        if(n == 0):
              aver_wight = 0
        else:
              aver_wight = round(aver_wight/n , 6)
        average_wighted.append(aver_wight)
        #print(max_items)
        #print(probability_items)
    df['max_item'] = pd.Series(max_items)
    df['max_similarity'] = pd.Series(max_similarities)
    df['max_wighted'] = pd.Series(max_wighted)


    df['probability_item'] = pd.Series(probability_items)
    df['probability_similarity'] = pd.Series(probability_similarities)
    df['probability_wighted'] = pd.Series(prob_wighted)

    df['average_wighted'] = pd.Series(average_wighted)
    df.to_csv(url)
similarity(train,'train_2.csv')
similarity(vali, 'vali_2.csv')
similarity(test, 'test_2.csv')
print('finished')






