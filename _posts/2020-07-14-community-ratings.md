---
title:  "Community"
excerpt_separator: "<!--more-->"
date:  2020-07-10
classes: wide
categories:
  - Blog
tags:
  - jupyter 
  - pandas
---

# Thinking about Community (1/2): Ratings 

**Note;** this post also esists as a [Gist](https://gist.github.com/mbellitti/a26071f8bf3f52005e18bdc8e70842bf), so you can download it, modify and run it yourself.

I have just finished watching Community for the first time, and it is a good show. It left me with two persistent unresolved questions, though:

1. I think that the quality drops significantly around season 4, and it stays low untile the end. Does it really?
1. I've seen many of these people before, where?.

In this post I'll look at the ratings of the episodes, trying to answer my first question. I have an idea for the second one that involves using [networkx](https://networkx.github.io/) to visualize the relations between actors who shared a production, but that's for another day.

First, I have to get my hands on the episode-by-episode ratings. Luckily, there is a nice package ([IMDbPY](https://imdbpy.github.io/)) designed to easily retrieve information about movies and TV shows from [IMDb](https://www.imdb.com/). 

The basic interface is simple: import the package and create an `IMDb` object. 
Its `get_movie(movieID)` method creates a `Movie` object with all the information we need (actually, only some of it. We will `update()` it later). The `movieID` is a unique numerical identifier of the movie/show, found at the end of the URL of its IMDb page: for example, for Community the URL is
```
https://www.imdb.com/title/tt1439629/
```
so the `movieID` is 1439629. You can change this ID to any other show's and run the entire notebook again to repeat the analysis.


```python
from imdb import IMDb

ia = IMDb()

series = ia.get_movie('1439629')
```

The `series` object behaves as a dictionary, with a few extra methods. Let's see what information it has


```python
series.keys()
```




    ['original title',
     'cast',
     'genres',
     'runtimes',
     'countries',
     'country codes',
     'language codes',
     'color info',
     'aspect ratio',
     'sound mix',
     'certificates',
     'number of seasons',
     'rating',
     'votes',
     'cover url',
     'imdbID',
     'plot outline',
     'languages',
     'title',
     'year',
     'kind',
     'series years',
     'akas',
     'seasons',
     'writer',
     'production companies',
     'distributors',
     'other companies',
     'plot',
     'canonical title',
     'long imdb title',
     'long imdb canonical title',
     'smart canonical title',
     'smart long imdb canonical title',
     'full-size cover url']




```python
series["seasons"]
```




    6



I really hoped for a moment. #sixseasonsandamovie

## Episode ratings
By default, `get_movie()` retrieves only some basic information, much less than what's available on IMDb. The rest of it is accessed using   


```python
ia.update(series,'episodes')
```

that creates and populates `series['episodes']`, a dictionary of the form 

```
{ season_number : Episode }
```
with the season number starting at 1. Plotting information from a nested dictionary is a little awkward, so I will flatten it an turn it into a [pandas](https://pandas.pydata.org/) DataFrame, which makes it much easier to analyze.


```python
from imdb.helpers import sortedEpisodes
import pandas as pd
```

Each episode is itself a dictionary:


```python
series['episodes'][1][1].keys()
```




    ['title',
     'kind',
     'episode of',
     'season',
     'episode',
     'rating',
     'votes',
     'original air date',
     'year',
     'plot',
     'canonical title',
     'long imdb title',
     'long imdb canonical title',
     'smart canonical title',
     'smart long imdb canonical title',
     'long imdb episode title',
     'series title',
     'canonical series title',
     'episode title',
     'canonical episode title',
     'smart canonical series title',
     'smart canonical episode title']



I don't really need all this information, so as I create the DataFrame I will only keep the following


```python
columns = "title","season","episode","rating","votes","original air date"
```


```python
episode_table = []

for episode in sortedEpisodes(series):
    episode_table.append({k:episode[k] for k in columns})

episode_df = pd.DataFrame(episode_table)
```

Let's take a look at the first few entries in the DataFrame


```python
episode_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>season</th>
      <th>episode</th>
      <th>rating</th>
      <th>votes</th>
      <th>original air date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Table Read</td>
      <td>-1</td>
      <td>-1</td>
      <td>9.801235</td>
      <td>402</td>
      <td>18 May 2020</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Pilot</td>
      <td>1</td>
      <td>1</td>
      <td>7.701235</td>
      <td>3570</td>
      <td>17 Sep. 2009</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Spanish 101</td>
      <td>1</td>
      <td>2</td>
      <td>7.801235</td>
      <td>3114</td>
      <td>24 Sep. 2009</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Introduction to Film</td>
      <td>1</td>
      <td>3</td>
      <td>8.301235</td>
      <td>3045</td>
      <td>1 Oct. 2009</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Social Psychology</td>
      <td>1</td>
      <td>4</td>
      <td>8.201235</td>
      <td>2798</td>
      <td>8 Oct. 2009</td>
    </tr>
  </tbody>
</table>
</div>



Sometimes IMDb is a little too complete. Episode 0 is not part of the original series (it's a cast reunion), let's drop it.


```python
episode_df.drop(0,inplace=True)
```

(Note: if you changed the `movieID` to some other series don't run this cell.) 

### Rating plots
Now I'm ready to answer my first question. Let's plot the episode ratings, grouped by season: 


```python
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")

%matplotlib inline
%config InlineBackend.figure_format = 'svg' # inline plots are vectorial

episode_df.groupby("season")['rating'].plot(marker='.',figsize=(10,4));
ax = plt.gca()
ax.set_xlim(0,len(episode_df)+1)
ax.set_ylim(6,10)
ax.set_xlabel("Episode Number")
ax.set_ylabel("Rating")
plt.tight_layout()

ax.legend(labels=["Season {}".format(i+1) for i in range(6)],ncol=3,frameon=True,loc="lower left");
```

<img src="https://raw.githubusercontent.com/mbellitti/mbellitti.github.io/master/_posts/ratings.svg?sanitize=true">



Huh, that's not what I expected. I thought season 4 was alright, but I definitely agree with season 6 having a weak start.

Dan Harmon [was fired](https://www.hollywoodreporter.com/news/communitys-dan-harmon-reveals-wild-586084) right before season 4 and rehired for season 5 and 6, and the audience clearly noticed.

**In conclusion, the data shows a consensus about the later seasons being worse than the earlier ones, but I still liked season 4 more then the last two.**

### Pareto Watching
It's known that 80% of the enjoyment comes from the top 20% of the episodes. What are they?


```python
quant80 = episode_df['rating'].quantile(0.8)
episode_df[episode_df["rating"] > quant80]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>season</th>
      <th>episode</th>
      <th>rating</th>
      <th>votes</th>
      <th>original air date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>21</th>
      <td>Contemporary American Poultry</td>
      <td>1</td>
      <td>21</td>
      <td>9.301235</td>
      <td>3926</td>
      <td>22 Apr. 2010</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Modern Warfare</td>
      <td>1</td>
      <td>23</td>
      <td>9.801235</td>
      <td>8985</td>
      <td>6 May 2010</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Epidemiology</td>
      <td>2</td>
      <td>6</td>
      <td>9.301235</td>
      <td>3971</td>
      <td>28 Oct. 2010</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Cooperative Calligraphy</td>
      <td>2</td>
      <td>8</td>
      <td>9.101235</td>
      <td>3286</td>
      <td>11 Nov. 2010</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Conspiracy Theories and Interior Design</td>
      <td>2</td>
      <td>9</td>
      <td>9.401235</td>
      <td>4032</td>
      <td>18 Nov. 2010</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Advanced Dungeons &amp; Dragons</td>
      <td>2</td>
      <td>14</td>
      <td>9.501235</td>
      <td>4700</td>
      <td>3 Feb. 2011</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Paradigms of Human Memory</td>
      <td>2</td>
      <td>21</td>
      <td>9.101235</td>
      <td>3168</td>
      <td>21 Apr. 2011</td>
    </tr>
    <tr>
      <th>48</th>
      <td>A Fistful of Paintballs</td>
      <td>2</td>
      <td>23</td>
      <td>9.701235</td>
      <td>5205</td>
      <td>5 May 2011</td>
    </tr>
    <tr>
      <th>49</th>
      <td>For a Few Paintballs More</td>
      <td>2</td>
      <td>24</td>
      <td>9.601235</td>
      <td>4270</td>
      <td>12 May 2011</td>
    </tr>
    <tr>
      <th>53</th>
      <td>Remedial Chaos Theory</td>
      <td>3</td>
      <td>4</td>
      <td>9.801235</td>
      <td>7773</td>
      <td>13 Oct. 2011</td>
    </tr>
    <tr>
      <th>57</th>
      <td>Documentary Filmmaking: Redux</td>
      <td>3</td>
      <td>8</td>
      <td>9.101235</td>
      <td>2993</td>
      <td>17 Nov. 2011</td>
    </tr>
    <tr>
      <th>63</th>
      <td>Pillows and Blankets</td>
      <td>3</td>
      <td>14</td>
      <td>9.301235</td>
      <td>3793</td>
      <td>5 Apr. 2012</td>
    </tr>
    <tr>
      <th>66</th>
      <td>Basic Lupine Urology</td>
      <td>3</td>
      <td>17</td>
      <td>9.501235</td>
      <td>3710</td>
      <td>26 Apr. 2012</td>
    </tr>
    <tr>
      <th>68</th>
      <td>Curriculum Unavailable</td>
      <td>3</td>
      <td>19</td>
      <td>9.201235</td>
      <td>2844</td>
      <td>10 May 2012</td>
    </tr>
    <tr>
      <th>69</th>
      <td>Digital Estate Planning</td>
      <td>3</td>
      <td>20</td>
      <td>9.501235</td>
      <td>3959</td>
      <td>17 May 2012</td>
    </tr>
    <tr>
      <th>70</th>
      <td>The First Chang Dynasty</td>
      <td>3</td>
      <td>21</td>
      <td>9.101235</td>
      <td>2642</td>
      <td>17 May 2012</td>
    </tr>
    <tr>
      <th>88</th>
      <td>Cooperative Polygraphy</td>
      <td>5</td>
      <td>4</td>
      <td>9.301235</td>
      <td>2931</td>
      <td>16 Jan. 2014</td>
    </tr>
    <tr>
      <th>89</th>
      <td>Geothermal Escapism</td>
      <td>5</td>
      <td>5</td>
      <td>9.401235</td>
      <td>3368</td>
      <td>23 Jan. 2014</td>
    </tr>
    <tr>
      <th>110</th>
      <td>Emotional Consequences of Broadcast Television</td>
      <td>6</td>
      <td>13</td>
      <td>9.601235</td>
      <td>3980</td>
      <td>2 Jun. 2015</td>
    </tr>
  </tbody>
</table>
</div>



Too bad Netflix and Hulu recently pulled "Advanced Dungeons & Dragons", after receiving a formal complaint from the city of Menzoberranzan.

Also, the people at IMDb _really_ like paintball.

### Rating heatmap
Another useful visualization is a rating heatmap, which [was](https://www.reddit.com/r/dataisbeautiful/comments/fx8m1h/oc_24_imdb_episode_heat_map/) [a](https://www.reddit.com/r/dataisbeautiful/comments/764vrp/continuing_my_imdb_heatmap_series_i_present/) [trend](https://www.reddit.com/r/dataisbeautiful/comments/fwjces/oc_the_absolute_quality_of_breaking_bad/) [on](https://www.reddit.com/r/dataisbeautiful/comments/782nxn/bojack_horsemans_imdb_rating_heatmap_oc/) /r/dataisbeautiful some time ago. Here it is in one line of python:


```python
min_acceptable = episode_df["rating"].quantile(.50) # set the median rating as the "white" color in the gradient

episode_df.pivot_table(index='episode',columns='season',values='rating').style.background_gradient(cmap="Greens",vmin=min_acceptable,vmax=10).set_precision(2).set_na_rep('')
```




<style  type="text/css" >
    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row0_col0 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row0_col1 {
            background-color:  #bee5b8;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row0_col2 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row0_col3 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row0_col4 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row0_col5 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row1_col0 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row1_col1 {
            background-color:  #eff9ec;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row1_col2 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row1_col3 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row1_col4 {
            background-color:  #ceecc8;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row1_col5 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row2_col0 {
            background-color:  #eff9ec;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row2_col1 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row2_col2 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row2_col3 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row2_col4 {
            background-color:  #88ce87;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row2_col5 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row3_col0 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row3_col1 {
            background-color:  #bee5b8;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row3_col2 {
            background-color:  #00682a;
            color:  #f1f1f1;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row3_col3 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row3_col4 {
            background-color:  #46ae60;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row3_col5 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row4_col0 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row4_col1 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row4_col2 {
            background-color:  #aedea7;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row4_col3 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row4_col4 {
            background-color:  #37a055;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row4_col5 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row5_col0 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row5_col1 {
            background-color:  #46ae60;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row5_col2 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row5_col3 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row5_col4 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row5_col5 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row6_col0 {
            background-color:  #aedea7;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row6_col1 {
            background-color:  #dbf1d6;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row6_col2 {
            background-color:  #ceecc8;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row6_col3 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row6_col4 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row6_col5 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row7_col0 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row7_col1 {
            background-color:  #73c476;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row7_col2 {
            background-color:  #73c476;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row7_col3 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row7_col4 {
            background-color:  #88ce87;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row7_col5 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row8_col0 {
            background-color:  #aedea7;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row8_col1 {
            background-color:  #37a055;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row8_col2 {
            background-color:  #e7f6e3;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row8_col3 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row8_col4 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row8_col5 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row9_col0 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row9_col1 {
            background-color:  #e7f6e3;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row9_col2 {
            background-color:  #bee5b8;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row9_col3 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row9_col4 {
            background-color:  #dbf1d6;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row9_col5 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row10_col0 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row10_col1 {
            background-color:  #aedea7;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row10_col2 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row10_col3 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row10_col4 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row10_col5 {
            background-color:  #88ce87;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row11_col0 {
            background-color:  #dbf1d6;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row11_col1 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row11_col2 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row11_col3 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row11_col4 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row11_col5 {
            background-color:  #ceecc8;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row12_col0 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row12_col1 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row12_col2 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row12_col3 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row12_col4 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row12_col5 {
            background-color:  #1a843f;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row13_col0 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row13_col1 {
            background-color:  #29914a;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row13_col2 {
            background-color:  #46ae60;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row13_col3 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row13_col4 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row13_col5 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row14_col0 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row14_col1 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row14_col2 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row14_col3 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row14_col4 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row14_col5 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row15_col0 {
            background-color:  #dbf1d6;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row15_col1 {
            background-color:  #aedea7;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row15_col2 {
            background-color:  #ceecc8;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row15_col3 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row15_col4 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row15_col5 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row16_col0 {
            background-color:  #88ce87;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row16_col1 {
            background-color:  #eff9ec;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row16_col2 {
            background-color:  #29914a;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row16_col3 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row16_col4 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row16_col5 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row17_col0 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row17_col1 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row17_col2 {
            background-color:  #eff9ec;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row17_col3 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row17_col4 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row17_col5 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row18_col0 {
            background-color:  #dbf1d6;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row18_col1 {
            background-color:  #88ce87;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row18_col2 {
            background-color:  #5db96b;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row18_col3 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row18_col4 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row18_col5 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row19_col0 {
            background-color:  #dbf1d6;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row19_col1 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row19_col2 {
            background-color:  #29914a;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row19_col3 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row19_col4 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row19_col5 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row20_col0 {
            background-color:  #46ae60;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row20_col1 {
            background-color:  #73c476;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row20_col2 {
            background-color:  #73c476;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row20_col3 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row20_col4 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row20_col5 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row21_col0 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row21_col1 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row21_col2 {
            background-color:  #aedea7;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row21_col3 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row21_col4 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row21_col5 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row22_col0 {
            background-color:  #00682a;
            color:  #f1f1f1;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row22_col1 {
            background-color:  #0b7734;
            color:  #f1f1f1;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row22_col2 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row22_col3 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row22_col4 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row22_col5 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row23_col0 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row23_col1 {
            background-color:  #1a843f;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row23_col2 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row23_col3 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row23_col4 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row23_col5 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row24_col0 {
            background-color:  #dbf1d6;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row24_col1 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row24_col2 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row24_col3 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row24_col4 {
            background-color:  #f7fcf5;
            color:  #000000;
        }    #T_05c9fd60_c643_11ea_b63c_c5370b46cb06row24_col5 {
            background-color:  #f7fcf5;
            color:  #000000;
        }</style><table id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06" ><thead>    <tr>        <th class="index_name level0" >season</th>        <th class="col_heading level0 col0" >1</th>        <th class="col_heading level0 col1" >2</th>        <th class="col_heading level0 col2" >3</th>        <th class="col_heading level0 col3" >4</th>        <th class="col_heading level0 col4" >5</th>        <th class="col_heading level0 col5" >6</th>    </tr>    <tr>        <th class="index_name level0" >episode</th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>    </tr></thead><tbody>
                <tr>
                        <th id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06level0_row0" class="row_heading level0 row0" >1</th>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row0_col0" class="data row0 col0" >7.70</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row0_col1" class="data row0 col1" >8.70</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row0_col2" class="data row0 col2" >8.10</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row0_col3" class="data row0 col3" >7.30</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row0_col4" class="data row0 col4" >7.90</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row0_col5" class="data row0 col5" >7.80</td>
            </tr>
            <tr>
                        <th id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06level0_row1" class="row_heading level0 row1" >2</th>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row1_col0" class="data row1 col0" >7.80</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row1_col1" class="data row1 col1" >8.30</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row1_col2" class="data row1 col2" >8.10</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row1_col3" class="data row1 col3" >7.60</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row1_col4" class="data row1 col4" >8.60</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row1_col5" class="data row1 col5" >7.60</td>
            </tr>
            <tr>
                        <th id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06level0_row2" class="row_heading level0 row2" >3</th>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row2_col0" class="data row2 col0" >8.30</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row2_col1" class="data row2 col1" >8.10</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row2_col2" class="data row2 col2" >8.10</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row2_col3" class="data row2 col3" >7.30</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row2_col4" class="data row2 col4" >9.00</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row2_col5" class="data row2 col5" >7.80</td>
            </tr>
            <tr>
                        <th id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06level0_row3" class="row_heading level0 row3" >4</th>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row3_col0" class="data row3 col0" >8.20</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row3_col1" class="data row3 col1" >8.70</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row3_col2" class="data row3 col2" >9.80</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row3_col3" class="data row3 col3" >7.10</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row3_col4" class="data row3 col4" >9.30</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row3_col5" class="data row3 col5" >8.10</td>
            </tr>
            <tr>
                        <th id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06level0_row4" class="row_heading level0 row4" >5</th>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row4_col0" class="data row4 col0" >7.80</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row4_col1" class="data row4 col1" >7.80</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row4_col2" class="data row4 col2" >8.80</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row4_col3" class="data row4 col3" >7.50</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row4_col4" class="data row4 col4" >9.40</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row4_col5" class="data row4 col5" >7.70</td>
            </tr>
            <tr>
                        <th id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06level0_row5" class="row_heading level0 row5" >6</th>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row5_col0" class="data row5 col0" >7.70</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row5_col1" class="data row5 col1" >9.30</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row5_col2" class="data row5 col2" >7.70</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row5_col3" class="data row5 col3" >7.50</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row5_col4" class="data row5 col4" >8.10</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row5_col5" class="data row5 col5" >7.70</td>
            </tr>
            <tr>
                        <th id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06level0_row6" class="row_heading level0 row6" >7</th>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row6_col0" class="data row6 col0" >8.80</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row6_col1" class="data row6 col1" >8.50</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row6_col2" class="data row6 col2" >8.60</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row6_col3" class="data row6 col3" >6.70</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row6_col4" class="data row6 col4" >7.80</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row6_col5" class="data row6 col5" >7.40</td>
            </tr>
            <tr>
                        <th id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06level0_row7" class="row_heading level0 row7" >8</th>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row7_col0" class="data row7 col0" >8.00</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row7_col1" class="data row7 col1" >9.10</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row7_col2" class="data row7 col2" >9.10</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row7_col3" class="data row7 col3" >7.90</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row7_col4" class="data row7 col4" >9.00</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row7_col5" class="data row7 col5" >7.90</td>
            </tr>
            <tr>
                        <th id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06level0_row8" class="row_heading level0 row8" >9</th>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row8_col0" class="data row8 col0" >8.80</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row8_col1" class="data row8 col1" >9.40</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row8_col2" class="data row8 col2" >8.40</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row8_col3" class="data row8 col3" >7.00</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row8_col4" class="data row8 col4" >8.10</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row8_col5" class="data row8 col5" >7.70</td>
            </tr>
            <tr>
                        <th id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06level0_row9" class="row_heading level0 row9" >10</th>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row9_col0" class="data row9 col0" >8.20</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row9_col1" class="data row9 col1" >8.40</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row9_col2" class="data row9 col2" >8.70</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row9_col3" class="data row9 col3" >7.60</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row9_col4" class="data row9 col4" >8.50</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row9_col5" class="data row9 col5" >7.60</td>
            </tr>
            <tr>
                        <th id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06level0_row10" class="row_heading level0 row10" >11</th>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row10_col0" class="data row10 col0" >7.90</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row10_col1" class="data row10 col1" >8.80</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row10_col2" class="data row10 col2" >7.70</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row10_col3" class="data row10 col3" >8.00</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row10_col4" class="data row10 col4" >8.00</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row10_col5" class="data row10 col5" >9.00</td>
            </tr>
            <tr>
                        <th id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06level0_row11" class="row_heading level0 row11" >12</th>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row11_col0" class="data row11 col0" >8.50</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row11_col1" class="data row11 col1" >8.00</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row11_col2" class="data row11 col2" >8.00</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row11_col3" class="data row11 col3" >8.20</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row11_col4" class="data row11 col4" >7.90</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row11_col5" class="data row11 col5" >8.60</td>
            </tr>
            <tr>
                        <th id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06level0_row12" class="row_heading level0 row12" >13</th>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row12_col0" class="data row12 col0" >8.00</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row12_col1" class="data row12 col1" >7.90</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row12_col2" class="data row12 col2" >8.20</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row12_col3" class="data row12 col3" >7.70</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row12_col4" class="data row12 col4" >8.20</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row12_col5" class="data row12 col5" >9.60</td>
            </tr>
            <tr>
                        <th id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06level0_row13" class="row_heading level0 row13" >14</th>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row13_col0" class="data row13 col0" >7.90</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row13_col1" class="data row13 col1" >9.50</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row13_col2" class="data row13 col2" >9.30</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row13_col3" class="data row13 col3" ></td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row13_col4" class="data row13 col4" ></td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row13_col5" class="data row13 col5" ></td>
            </tr>
            <tr>
                        <th id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06level0_row14" class="row_heading level0 row14" >15</th>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row14_col0" class="data row14 col0" >8.10</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row14_col1" class="data row14 col1" >8.10</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row14_col2" class="data row14 col2" >8.10</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row14_col3" class="data row14 col3" ></td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row14_col4" class="data row14 col4" ></td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row14_col5" class="data row14 col5" ></td>
            </tr>
            <tr>
                        <th id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06level0_row15" class="row_heading level0 row15" >16</th>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row15_col0" class="data row15 col0" >8.50</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row15_col1" class="data row15 col1" >8.80</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row15_col2" class="data row15 col2" >8.60</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row15_col3" class="data row15 col3" ></td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row15_col4" class="data row15 col4" ></td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row15_col5" class="data row15 col5" ></td>
            </tr>
            <tr>
                        <th id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06level0_row16" class="row_heading level0 row16" >17</th>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row16_col0" class="data row16 col0" >9.00</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row16_col1" class="data row16 col1" >8.30</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row16_col2" class="data row16 col2" >9.50</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row16_col3" class="data row16 col3" ></td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row16_col4" class="data row16 col4" ></td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row16_col5" class="data row16 col5" ></td>
            </tr>
            <tr>
                        <th id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06level0_row17" class="row_heading level0 row17" >18</th>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row17_col0" class="data row17 col0" >8.00</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row17_col1" class="data row17 col1" >7.40</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row17_col2" class="data row17 col2" >8.30</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row17_col3" class="data row17 col3" ></td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row17_col4" class="data row17 col4" ></td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row17_col5" class="data row17 col5" ></td>
            </tr>
            <tr>
                        <th id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06level0_row18" class="row_heading level0 row18" >19</th>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row18_col0" class="data row18 col0" >8.50</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row18_col1" class="data row18 col1" >9.00</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row18_col2" class="data row18 col2" >9.20</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row18_col3" class="data row18 col3" ></td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row18_col4" class="data row18 col4" ></td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row18_col5" class="data row18 col5" ></td>
            </tr>
            <tr>
                        <th id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06level0_row19" class="row_heading level0 row19" >20</th>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row19_col0" class="data row19 col0" >8.50</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row19_col1" class="data row19 col1" >7.60</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row19_col2" class="data row19 col2" >9.50</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row19_col3" class="data row19 col3" ></td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row19_col4" class="data row19 col4" ></td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row19_col5" class="data row19 col5" ></td>
            </tr>
            <tr>
                        <th id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06level0_row20" class="row_heading level0 row20" >21</th>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row20_col0" class="data row20 col0" >9.30</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row20_col1" class="data row20 col1" >9.10</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row20_col2" class="data row20 col2" >9.10</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row20_col3" class="data row20 col3" ></td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row20_col4" class="data row20 col4" ></td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row20_col5" class="data row20 col5" ></td>
            </tr>
            <tr>
                        <th id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06level0_row21" class="row_heading level0 row21" >22</th>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row21_col0" class="data row21 col0" >8.00</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row21_col1" class="data row21 col1" >8.00</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row21_col2" class="data row21 col2" >8.80</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row21_col3" class="data row21 col3" ></td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row21_col4" class="data row21 col4" ></td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row21_col5" class="data row21 col5" ></td>
            </tr>
            <tr>
                        <th id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06level0_row22" class="row_heading level0 row22" >23</th>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row22_col0" class="data row22 col0" >9.80</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row22_col1" class="data row22 col1" >9.70</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row22_col2" class="data row22 col2" ></td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row22_col3" class="data row22 col3" ></td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row22_col4" class="data row22 col4" ></td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row22_col5" class="data row22 col5" ></td>
            </tr>
            <tr>
                        <th id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06level0_row23" class="row_heading level0 row23" >24</th>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row23_col0" class="data row23 col0" >8.20</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row23_col1" class="data row23 col1" >9.60</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row23_col2" class="data row23 col2" ></td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row23_col3" class="data row23 col3" ></td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row23_col4" class="data row23 col4" ></td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row23_col5" class="data row23 col5" ></td>
            </tr>
            <tr>
                        <th id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06level0_row24" class="row_heading level0 row24" >25</th>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row24_col0" class="data row24 col0" >8.50</td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row24_col1" class="data row24 col1" ></td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row24_col2" class="data row24 col2" ></td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row24_col3" class="data row24 col3" ></td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row24_col4" class="data row24 col4" ></td>
                        <td id="T_05c9fd60_c643_11ea_b63c_c5370b46cb06row24_col5" class="data row24 col5" ></td>
            </tr>
    </tbody></table>



That was the first time I used a pivot table, this better not awaken anything in me.
