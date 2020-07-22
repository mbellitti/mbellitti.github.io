---
title:  "Community (2/2): Networks of Coworkers"
excerpt_separator: "<!--more-->"
date: 2020-07-21
categories:
  - Blog
tags:
  - jupyter 
  - networkx
---

**Note:** This post is a functioning Jupyter notebook, accessible as a Gist [here](https://gist.github.com/98c9776653fd522c4bbd4833bba88b08).
 
This is the second part ([first part](https://mbellitti.github.io/blog/community-ratings/)) of my musings about Community. In this post I want to explore the work relations between members of the cast. 

I noticed that both Mike and Gus from Breaking Bad are in Community, so I thought: what else did I miss? The truth is that I don't watch that many TV shows and movies, so this could also be a way to discover new ones.

Just like last time we create a `series` object with some basic information about the show:


```python
from imdb import IMDb

ia = IMDb()
series = ia.get_movie('1439629')
```

By default, `series['cast']` is a list containing only the 50 actors who appear most frequently in the series. To access the full 617-actors long list I could use `ia.update(series,'full credits')`, but for my purposes the short list is enough.

My idea is that an actor that appears only in a couple of episodes is not really part of the "main cast", and I would not immediately think "I've seen them in Community" while watching another show. This is somewhat unfair, as there are memorable performances entirely confined to a single episode ([Matt Berry] as the grifting professor, to mention one), but it is a simple filter to implement that mostly captures the correct idea.

In practice, I will keep only actors that have been in at least 4 episodes: from Joel McHale (111 episodes) down to Dan Harmon himself (4 episodes). To figure this out I looked at the [full cast page](https://www.imdb.com/title/tt1439629/fullcredits?ref_=tt_cl_sm#cast), I couldn't find a way to access this information through IMDbPY.


```python
series['cast'][41]
```




    <Person id:1363595[http] name:_Dan Harmon_>



For each actor, I would like to look at the list of movies/shows they have also appeared in (their 'filmography'), and check if there is any significant group of actors that appeared together both in Community and some other show. 

By default, the `actor` object does not contain the filmography, so I have to retrieve it.

The following cells create a dictionary 
```
films = { film : [actor1, actor2, ...] }
```
of all the actors who appeared in `film` (a `Movie` object). The elements of the list of actors are `Person` objects.

To monitor the progress of the loop I am using [tqdm](https://tqdm.github.io/), a small library that makes progress bars. It's one of my favorite tricks: it gives good feedback so I have an idea of how long a loop is going to take, and it's pretty.

Create a dictionary 
```
films = { film : [actor1, actor2, ...] }
```
of all the actors in that movie.



```python
from tqdm.notebook import tqdm
import time
```


```python
films = {}
max_num_actor = 42

for actor in tqdm(series['cast'][:max_num_actor]):
    time.sleep(.1) # it's polite not to spam imdb with too many requests per second
    ia.update(actor,'filmography')
    # imdbpy has different keys for actors and actresses
    filmography = actor['filmography']['actor'] if 'actor' in actor['filmography'] else actor['filmography']['actress']

    for film in filmography:
        films.setdefault(film,[]).append(actor)
#         films.setdefault("tt"+film.getID(),[]).append("nm"+actor.getID()) # string based approach
```


    HBox(children=(FloatProgress(value=0.0, max=42.0), HTML(value='')))


    


It turns out that this produces a very long list of movies:


```python
len(films)
```




    2566



It will be hard to visualize the relations between this many movies. I am interested in movies I'll likely watch, so let's filter out the low-rated ones. 

The movie data by default does not include ratings, so we have to update it. This is a very long operation, so I have run it once and saved the resulting object using `pickle`.


```python
# this takes a long time
# for film,actors in tqdm(films.items()):
#     time.sleep(0.1) # it's polite not to spam too hard
#     ia.update(film)
```


```python
import pickle

# saving the data
# with open('films.pkl','wb') as file:
#     pickle.dump(films,file)

# loading it from disk
with open('films.pkl','rb') as file:
    films = pickle.load(file)
```

I have to choose a cutoff rating to keep only the "high" rated movies, but ratings are quite arbitrary, so I am going to rate them on a curve: let's look at the rating distribution


```python
import numpy as np
ratings = np.array([film['rating'] for film in films if 'rating' in film])
```


```python
import matplotlib.pyplot as plt
plt.style.use("seaborn")
%matplotlib inline
%config InlineBackend.figure_format = 'svg' 

fig,ax = plt.subplots(1,1,figsize=(7,3))
ax.hist(ratings);
ax.set_xticks(range(11))
ax.set_xlim(0,10)
ax.set_xlabel("Rating")
ax.set_ylabel("Count");
```


![](https://raw.githubusercontent.com/mbellitti/mbellitti.github.io/master/_posts/Community-Copy1_files/Community-Copy1_15_0.svg)

If I want to keep only the best 10%, below which rating should I cut? 


```python
np.quantile(ratings,0.9)
```




    8.1



There are a few issues with filtering:
- IMDB has entries for unreleased films, which of course do not have a rating, so I will check explicitly that `'rating'` is a valid key for the movie;
- Some movies have an unusual high rating and very few votes. While this could mean that they are hidden gems, here I will treat them as outliers, and focus on more mainstream movies (defined as "having more than 1000 votes"); 
- Apparently, the Internet MOVIE Database has lots of entries about videogames, which is surprising but irrelevant for my purposes.


```python
min_rating = 8.2
good_films = {film:actors for film,actors in films.items() if ('rating' in film 
                                                               and film['rating'] >= min_rating 
                                                               and film['votes'] > 1000 
                                                               and (film['kind'] != 'video game')
                                                              )}
```

Now I'm ready to build `shared`, a dictionary `{ n: [movie1,movie2,..]}` where all the movies in the list `movie1,movie2,...` contain `n` actors that were also in Community.


```python
shared = {}

for film,actor in good_films.items():
    shared.setdefault(len(actor),[]).append(film)
    
del shared[1] # remove movies that are not shared
del shared[max(shared.keys())] # remove Community from the list
```


```python
for n in range(13,0,-1):
    if n in shared:
        print ("Movies/Series with {} actors of the {} cast:".format(n,series['title']))
        for movie in shared[n]:
            print('\t',movie['title'],'({})'.format(movie['rating']))
        print()
```

    Movies/Series with 7 actors of the Community cast:
    	 Rick and Morty (9.2)
    	 Curb Your Enthusiasm (8.7)
    	 Modern Family (8.4)
    	 How I Met Your Mother (8.3)
    
    Movies/Series with 6 actors of the Community cast:
    	 BoJack Horseman (8.7)
    	 Regular Show (8.4)
    
    Movies/Series with 5 actors of the Community cast:
    	 Brooklyn Nine-Nine (8.4)
    	 Key and Peele (8.3)
    	 Arrested Development (8.7)
    
    Movies/Series with 4 actors of the Community cast:
    	 Adventure Time (8.6)
    	 The Simpsons (8.7)
    	 The Office (8.9)
    
    Movies/Series with 3 actors of the Community cast:
    	 DuckTales (8.2)
    	 Chuck (8.2)
    	 The West Wing (8.8)
    	 Boston Legal (8.4)
    	 Entourage (8.4)
    	 Veep (8.3)
    	 The League (8.2)
    
    Movies/Series with 2 actors of the Community cast:
    	 Sons of Anarchy (8.6)
    	 The Venture Bros. (8.5)
    	 Mad Men (8.6)
    	 Avengers: Endgame (8.4)
    	 Psych (8.3)
    	 House (8.7)
    	 The Soup (8.2)
    	 Friends (8.9)
    	 30 Rock (8.2)
    	 Mr. Show with Bob and David (8.3)
    	 The Daily Show (8.4)
    	 Gravity Falls (8.9)
    	 Parks and Recreation (8.6)
    	 Dexter (8.6)
    	 Veronica Mars (8.3)
    	 Scrubs (8.3)
    	 Kitchen Confidential (8.2)
    	 Flight of the Conchords (8.5)
    


A lot of these are cartoons! I didn't expect it. I think that in the spirit of the exercise I should keep them, since hearing a familiar voice still makes me wondere where else I heard it. 

There are a lot of shows I've heard about, and some I have watched already (Rick and Morty and Modern Family, for example). A show that's ranked high and I haven't watched yet is BoJack Horseman, let's check who are the actors in it. For this part it's convenient to have [Imagus](https://addons.mozilla.org/en-US/firefox/addon/imagus/) installed: it's a Firefox addon that shows a popup with the linked image as you hover.


```python
from IPython.core.display import display, HTML

for movie in shared[6]:
    print(movie['title'])
    for actor in good_films[movie]:
        display(HTML("""<a href="""+actor['full-size headshot']+"""">"""+actor['name']+"""</a>"""))
```

**Note**: this output has been reformatted to appear a little nicer in the post, but the
jupyter notebook prints clickable links too.

**BoJack Horseman**

[Joel McHale](https://m.media-amazon.com/images/M/MV5BMTU2ODA0NzMwNV5BMl5BanBnXkFtZTgwOTA4MDA3NTE@.jpg)

[Alison Brie](https://m.media-amazon.com/images/M/MV5BMjJkNDg5ZDctM2RlZS00NjFmLTkxZjktMWE5NGQzMDg4NDFhXkEyXkFqcGdeQXVyMTMwMDM1OTQ@.jpg)

[Ken Jeong](https://m.media-amazon.com/images/M/MV5BMTQyMTczNzU4Nl5BMl5BanBnXkFtZTYwODUxMjMy.jpg)

[Yvette Nicole Brown](https://m.media-amazon.com/images/M/MV5BMTUxNTYwMTQwNV5BMl5BanBnXkFtZTgwMjQ3NzMxOTE@.jpg)

[Paget Brewster](https://m.media-amazon.com/images/M/MV5BMDA3OGU0YjktNWY5ZC00MDUxLWJhMGUtMDJmZWZkMzczZDcyXkEyXkFqcGdeQXVyMTA4MDI5MTg5.jpg)

[Keith David](https://m.media-amazon.com/images/M/MV5BMTI5OTU4ODI1MF5BMl5BanBnXkFtZTcwMjU1NjkyNA@@.jpg)

**Regular Show**
[Joel McHale] (https://m.media-amazon.com/images/M/MV5BMTU2ODA0NzMwNV5BMl5BanBnXkFtZTgwOTA4MDA3NTE@.jpg")
[Gillian Jacobs] (https://m.media-amazon.com/images/M/MV5BMTk5OTc5MTM1OV5BMl5BanBnXkFtZTcwMTU4NDI1NA@@.jpg")
[Yvette Nicole Brown] (https://m.media-amazon.com/images/M/MV5BMTUxNTYwMTQwNV5BMl5BanBnXkFtZTgwMjQ3NzMxOTE@.jpg")
[Donald Glover] (https://m.media-amazon.com/images/M/MV5BNzUxNTU5ODkxNl5BMl5BanBnXkFtZTgwOTIyNjc5MDI@.jpg")
[Keith David] (https://m.media-amazon.com/images/M/MV5BMTI5OTU4ODI1MF5BMl5BanBnXkFtZTcwMjU1NjkyNA@@.jpg")
[Eddie Pepitone] (https://m.media-amazon.com/images/M/MV5BMTgxNTIyNzIxNV5BMl5BanBnXkFtZTgwODA1NTM2NzE@.jpg")


It seems BoJack is basically a reunion of Seson 6. I think I'll watch that next, and I'll give Regular Show a chance too.

##  Visualizing the network of coworkers
Let's try something fancier. During my master's work I studied the dynamics of stochastic algorithms on random XORSAT instances, and visualized it using [networkx](https://networkx.github.io/), so I'm already a little familiar with it. 

It would be nice to represent the relation between actors who worked on the same movies as a graph, but this kind of visualization becomes very quickly hard to manage. To reduce a little the information density presented, I will keep only movies that have at least 6 people of the Community cast.


```python
import networkx as nx
from networkx.algorithms import bipartite
```

Here I build a dictionary (`shared_films_dic`) from films that have at least two of the people who appeared in Community to the list of people who worked on them. I am not very happy with how this code turned out, double comprehensions are harder to read


```python
# shared_films = set(film for film_list in shared.values() for film in film_list)

shared_films = []

for n,film_list in shared.items():
    if n >= 6: 
        shared_films += film_list
        
shared_films_dic = {film:good_films[film] for film in set(shared_films)}
```

Notice that there is a node connected to every single actor in this list: Community. This is not particularly interesting and adds a lot of edges to the graph, making it harder to draw, so I will remove it.


```python
if series in shared_films_dic:
    del shared_films_dic[series]
```

Convert the dictionary to a networkx `Graph` object


```python
G = nx.Graph(shared_films_dic)
# G = nx.k_core(G,2)
len(G)
```




    28



Out of curiosity: is the graph still connected?


```python
nx.is_connected(G)
```




    True



That's cool. It means that every pair among the actors we are looking at (the 42 main actors in Community) is connected by being "a coworker of a coworker".


```python
fig,ax = plt.subplots(1,1,figsize=(10,8))
pos = nx.kamada_kawai_layout(G)

plt.style.use("default") # make the background white
nx.draw_networkx_nodes(G,pos=pos,ax=ax,node_shape='.',node_color="white")
nx.draw_networkx_labels(G,pos=pos,ax=ax,font_size=8)
nx.draw_networkx_edges(G,pos=pos,alpha=0.5,width=0.5);
```

![](https://raw.githubusercontent.com/mbellitti/mbellitti.github.io/master/_posts/Community-Copy1_files/Community-Copy1_37_0.svg)

This image is in SVG format, I recommend right clicking and using "View Image"
for easier reading of the labels.

An interesting side effect of the [Kamada Kawai](https://www.sciencedirect.com/science/article/abs/pii/0020019089901026?via%3Dihub) layout is that nodes with the same connectivity end up being roughly in the same place. For example, this drawing shows clearly that Bill Parks, David Neher, and J.P. Manoux (upper left area) all have worked on HIMYM and Modern Family.
The same goes for Jerry Minor, Craig Cackowski, Jordan Black, and Dan Bakkedahl. 

This is a surprisingly good tool to quickly identify this kind of pattern.
