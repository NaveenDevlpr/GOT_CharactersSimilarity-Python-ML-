import pandas as pd
import numpy as ny
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
import plotly.express as px


df=pd.read_json('script-bag-of-words.json')


dialogue_dict={}

for index,row in df.iterrows():
    for item in row['text'] :
        if item['name'] in dialogue_dict:
            dialogue_dict[item['name']]=dialogue_dict[item['name']]+ item['text']
            
        else:
            dialogue_dict[item['name']]=item['text']+" "
            

dialogue_df=pd.DataFrame()

dialogue_df['characters']=dialogue_dict.keys()
dialogue_df['words']=dialogue_dict.values()




dialogue_df['num_words']=dialogue_df['words'].apply(lambda x:len(x.split()))
dialogue_df=dialogue_df.sort_values('num_words',ascending=False)

#print(dialogue_df.head(100))


cv=CountVectorizer(stop_words='english')

embedding=cv.fit_transform(dialogue_df['words']).toarray()

embedding=embedding.astype('float64')

tsne=TSNE(n_components=2,verbose=1,random_state=123)

z=tsne.fit_transform(embedding)


dialogue_df['x']=z.T[0]
dialogue_df['y']=z.T[1]

fig=px.scatter(dialogue_df.head(50),x='x',y='y',color='characters')
fig.show()

