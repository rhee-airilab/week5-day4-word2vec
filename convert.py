# coding: utf-8

# # Namu Wiki JSON DB file parsing

# ## Reading DB Dump File

# In[2]:

import json
import pickle
from pprint import pprint

#filename = '../namu/namuwiki_170327.json' 
filename = 'namuwiki_170327.json' 

# Read file to memory, it takes some time.
with open(filename) as data_file:    
    data = json.load(data_file)

#with open('namuwiki.pickle','wb') as pickle_file:
#    pickle.dump(data,pickle_file)
   


# ## Data Exploration

# In[3]:

# data is list of articles
# Let's see how many articles in the database
print("number of articles:", len(data)) 

# Let's see the first article
print("The first article is:")
print(data[0])


# In[4]:

print(data[201704])


# In[5]:

# this black list article does not contain natural language knowledge
black_list_title = ['공지사항/차단 내역/통합본']

# Check some statistics of whole dataset
count_dict = {}
for article in data:
    if article['title'] in black_list_title:
        continue # remove blacklist article
        
#     if(len(article['text']) > 10000 and len(article['text']) < 11000):
#         print(article)
#         break
        
    if count_dict.get(len(article['text'])) == None:
        count_dict[len(article['text'])] = 1
    else:
        count_dict[len(article['text'])] = count_dict[len(article['text'])] + 1        
    
    
print("min text size:", min(count_dict.keys()))
print("max text size:", max(count_dict.keys()))


# In[11]:

MAX_ARTICLE_SIZE = max(count_dict.keys())

bucket_size = 1000
num_bucket = MAX_ARTICLE_SIZE // bucket_size + 1

print('num_bucket:', num_bucket)

bucket_counts = [0] * num_bucket
for key, value in count_dict.items():
    index = key // bucket_size
    bucket_counts[index] = bucket_counts[index] + value

print(bucket_counts)



# # Test parsing

# In[6]:

# Article contains title, text, and other things
# Let's extract title and text from several articles
for i in range(3):
    print(data[i]['title'].encode('utf-8'))
    print(data[i]['text'].encode('utf-8'))
    print()


# ## Preprocessing with RegEx

# In[8]:

# Using regular expression, we can strip some grammar. Let's see how we can do it. 
import re
text = "딴 사람도 아니고 프로팀 [[Counter Logic Gaming|CLG]] 소속 전 서포터 [[스티브 차우|차우스터]]가 남긴 말이다."
t1 = re.sub(r"\[\[([^\]|]*)\]\]", r'\1', text) # remove link
print(t1)
t2 = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]|]+)\]\]", r'\1', text) # remove link
print(t2)


# In[9]:

# We want only plain texts, so strip wiki grammer.
# Refer this link to know more about grammar. https://namu.wiki/w/%EB%82%98%EB%AC%B4%EC%9C%84%ED%82%A4:%EB%AC%B8%EB%B2%95%20%EB%8F%84%EC%9B%80%EB%A7%90

# Use regular expression to capture some pattern

# see http://stackoverflow.com/questions/2718196/find-all-chinese-text-in-a-string-using-python-and-regex
chinese = re.compile(u'[⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎]', re.UNICODE)
japanese = re.compile(u'[\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\uff00-\uff9f\u4e00-\u9faf\u3400-\u4dbf]', re.UNICODE)

# hangul = re.compile('[^ ㄱ-ㅣ가-힣]+') # 한글과 띄어쓰기를 제외한 모든 글자
# hangul = re.compile('[^ \u3131-\u3163\uac00-\ud7a3]+')  # 위와 동일
# result = hangul.sub('', s) # 한글과 띄어쓰기를 제외한 모든 부분을 제거


def strip(text):               
    text = re.sub(r"\{\{\{#\!html[^\}]*\}\}\}", '', text, flags=re.IGNORECASE|re.MULTILINE|re.DOTALL) # remove html
    text = re.sub(r"#redirect .*", '', text, flags=re.IGNORECASE) # remove redirect
    text = re.sub(r"\[\[분류:.*", '', text) # remove 분류
    text = re.sub(r"\[\[파일:.*", '', text) # remove 파일
    text = re.sub(r"\* 상위 문서 ?:.*", '', text) # remove 상위문서        
    text = re.sub(r"\[youtube\(\w+\)\]", '', text, flags=re.IGNORECASE) # remove youtube
    text = re.sub(r"\[include\(([^\]|]*)(\|[^]]*)?\]", r'\1', text, flags=re.IGNORECASE) # remove include
    text = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]|]+)\]\]", r'\1', text) # remove link
    text = re.sub(r"\[\*([^\]]*)\]", '', text) # remove 각주
    text = re.sub(r"\{\{\{([^\ }|]*) ([^\}|]*)\}\}\}", r'\2', text) # remove text color/size
    text = re.sub(r"'''([^']*)'''", r'\1', text) # remove text bold
    text = re.sub(r"(~~|--)([^']*)(~~|--)", '', text) # remove strike-through
    
    text = re.sub(r"\|\|(.*)\|\|", '', text) # remove table
                                   
    text = chinese.sub('', text) # remove chinese
    text = japanese.sub('', text) # remove japanese
    return text

for i in range(2):
    print(data[i]['title'])
    # print(data[i]['text'])
    print(strip(data[i]['text']))
    print()


# In[12]:

# Generate raw text corpus

MIN_TEXT_SIZE = 5000

count = 10
with open('input.txt', 'w') as f:
    for article in data:
        if len(article['text']) < MIN_TEXT_SIZE or len(article['text']) >= MAX_ARTICLE_SIZE:        
            continue # skip too small, too large articles

        text = strip(article['text'])
        f.write("%s\n%s\n\n\n" % (article['title'], text))
        # print(article['title'])
        # print(article['text'])
        # print(text)
        


