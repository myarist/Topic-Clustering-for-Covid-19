# Topic Modelling for Covid-19

Here, I using LDA (Latent Dirichlet Allocation)

## Modules

There are quite a number of modules that I use

```
import os
import pandas as pd
import numpy as np
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as py
import plotly.graph_objs as go
import gensim
from gensim import corpora, models, similarities
import logging
import tempfile
from nltk.corpus import stopwords
from string import punctuation
from collections import OrderedDict
import seaborn as sns
import pyLDAvis.gensim
import matplotlib.pyplot as plt
```

## Trend for May

Then I use **plotly module** for visualizing trend of tweets in May

```
tweets['datetime'] = pd.to_datetime(tweets['datetime'], format='%Y-%m-%d')
tweetsT = tweets['datetime']

trace = go.Histogram(
    x=tweetsT,
    marker=dict(
        color='blue'
    ),
    opacity=0.75
)

layout = go.Layout(
    title='Tweet Activity in May',
    height=450,
    width=1200,
    xaxis=dict(
        title='Date and Month'
    ),
    yaxis=dict(
        title='Tweet Quantity'
    ),
    bargap=0.2,
)

data = [trace]

fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig)
```

![Trend](https://github.com/MyArist/Topic-Modelling-for-Covid-19/blob/master/LDA/tren.png)

## StopWord

Of course I also remove so many stopword inside the tweet

```
list1 = ['corona', 'coronavirus','indonesia', 'indonesian','covid19', 'covid', 'via',
         'city', 'names', 'may', 'today', 'new', 'could', 
         '24', '557', '678', '4', '20', '1520', '25773', '30', '10', '25216', '29', '1', '53', '28',
         'â€¦', 'â€¢', 'â€™', 'â€“', 'Â«', 'â€', 'Â»', 'â‚¬', 'Â£', 'Â©', 'Â°c', ' Â£', 'å', 'â', 'ë']
stoplist = stopwords.words('english') + list(punctuation) + list1
```

## Clustermap

I make clustermap to see the correlation each of word

```
g=sns.clustermap(df_lda.corr(), center=0, standard_scale=1, cmap="RdBu", metric='cosine', linewidths=.75, figsize=(15, 15))
plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
plt.show()
```

![Clustermap](https://github.com/MyArist/Topic-Modelling-for-Covid-19/blob/master/LDA/clustermap.png)

## LDA

Finally, the LDA

![LDA](https://github.com/MyArist/Topic-Modelling-for-Covid-19/blob/master/LDA/lda.png)
