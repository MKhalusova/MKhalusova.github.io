---
layout: post
title: "Pandas Data Structures: DataFrame basics"
date: 2017-09-08
---

In the [previous post](mkhalusova.github.io/blog/2017/08/31/pandas-series) I've summed up some of the basics of using pandas Series data structure. 
Now, let's take a quick look at DataFrame. 

DataFrame is a 2-dimensional labeled data structure with columns of potentially different types. It's sort of like an SQL table, or a dict of Series objects. 

This is the object you use a lot when digging through some data with pandas. I'm not going to describe how to create a DataFrame from lists and dicts, because for me it's more common that I need to dig through some existing data rather than create new data. As it happens with existing data, it comes in all kinds of formats, and for the most typical ones, it's easy to read it into a DataFrame for further manipulation.

For my examples I'll be using `movie_metadata.csv` that has been scraped from IMDB.com by Chuan Sun ([@sundeepblue on Github](https://github.com/sundeepblue/movie_rating_prediction)) .

## Read to DataFrame
Quite often your data comes as an excel or .csv file. 
To read it into a DataFrame, just use, well, `read_excel` or `read_csv`.  

```python
import pandas as pd
df = pd.read_csv('movie_metadata.csv')
```

In ideal world (and in this example) that would be it, but most of the time the original excel or csv files are full of mess. Luckily you don't need to write any code to get only one sheet, or skip rows, or fix encoding - `read_excel` and `read_csv` have a bunch of options to help you fix things.
There are also standard methods to read from an SQL query or database table, or JSON.  
Unfortunately, if you have XML, you'll need to do some parsing yourself. Perhaps, I'll write an example of how I did it in some other post. But here, we have a nice clean csv.

## What's in a DataFrame?
Now, when you have a DataFrame (with loads of data), the fun part begins. 
Pandas allow you to do a lot with a DataFrame with a couple of lines of code. 
But first, let's see what we've got in it. 

Let's see how big is the DataFrame we've got:

```python
df.shape
--------------------
(5043, 28)
--------------------
```
There you go - 5043 rows and 28 columns.

What are the column names?

```python
list(df)
--------------------
['color', 'director_name', 'num_critic_for_reviews', 'duration', 'director_facebook_likes', 'actor_3_facebook_likes', 'actor_2_name', 'actor_1_facebook_likes', 'gross', 'genres', 'actor_1_name', 'movie_title', 'num_voted_users', 'cast_total_facebook_likes', 'actor_3_name', 'facenumber_in_poster', 'plot_keywords', 'movie_imdb_link', 'num_user_for_reviews', 'language', 'country', 'content_rating', 'budget', 'title_year', 'actor_2_facebook_likes', 'imdb_score', 'aspect_ratio', 'movie_facebook_likes']
--------------------
```

Like with Series you can use .head() or tail() to “preview” what the data looks like:
 
```python
df.head(3)
--------------------
    color		director_name  		num_critic_for_reviews  	duration  \
0  Color   	James Cameron                 723.0     			178.0   
1  Color  	Gore Verbinski                   	302.0     			169.0   
2  Color      	Sam Mendes                   	602.0     			148.0   

   	director_facebook_likes  	actor_3_facebook_likes      actor_2_name  \
0                      0.0                   	855.0  			Joel David Moore   
1                    563.0                  	1000.0     		Orlando Bloom   
2                      0.0                   	161.0      		Rory Kinnear   

   	actor_1_facebook_likes        gross                           genres  \
0                  1000.0  		760505847.0  	Action|Adventure|Fantasy|Sci-Fi   
1                 40000.0  		309404152.0       	Action|Adventure|Fantasy   
2                 11000.0  		200074175.0       	 Action|Adventure|Thriller   

          ...          
```

## Let's get some data
Now that we have an idea of what's in this DataFrame, we can start digging some data out of it. 

We can start by getting a column by its name, and set a range of indices we want, say we want first three movie titles:

```python
df['movie_title'][:3]
--------------------
0                                      Avatar 
1    Pirates of the Caribbean: At Worlds End 
2                                     Spectre 
Name: movie_title, dtype: object
--------------------
```

We can specify multiple columns too:

```python
df[['movie_title','director_name']][:3]
--------------------
                                 movie_title   director_name
0                                    Avatar    James Cameron
1  Pirates of the Caribbean: At Worlds End   Gore Verbinski
2                                   Spectre       Sam Mendes
--------------------
```

We can see top 5 directors with the most movies made by them (at least from these records).

```python
df['director_name'].value_counts()[:5]
--------------------
Steven Spielberg    26
Woody Allen         22
Martin Scorsese     20
Clint Eastwood      20
Ridley Scott        17
Name: director_name, dtype: int64
--------------------
```

We can see what's the average number of movies for a director:

```python
df['director_name'].value_counts().mean()
--------------------
2.05963302752
--------------------
```

Let's find all the movies where Woody Allen is a director and plays the main character:

```python
df[(df['director_name'] =='Woody Allen') & (df['actor_1_name']=='Woody Allen')]['movie_title']
--------------------
1861                      The Curse of the Jade Scorpion 
2287                                Deconstructing Harry 
2430                                   Small Time Crooks 
2457                                       Anything Else 
2577                                    Hollywood Ending 
2695                                    New York Stories 
3889                                          Annie Hall 
4250                                             Sleeper 
4252    Everything You Always Wanted to Know About Sex...
4324                                             Bananas 
Name: movie_title, dtype: object
--------------------
```

How many movies are there where a director and the main actor are the same person?

```python
len(df[(df['director_name'] == df['actor_1_name'])])
--------------------
64
--------------------
```

Let's say we'd like to find out whose movies have brought more gross overall. We'll need to use groupby and aggregate for that

```python
movies = df[['director_name','gross']]

movies.groupby('director_name').aggregate(sum).sort_values(by="gross", ascending=False).head(10)
# This line means "Group the rows by director_name. 
# Add up all the values for each director_name, 
# then sort values in column gross in descending order. Show only first 10".

--------------------
                          gross
director_name                  
Steven Spielberg   4.114233e+09
Peter Jackson      2.592969e+09
Michael Bay        2.231243e+09
Tim Burton         2.071275e+09
Sam Raimi          2.049549e+09
James Cameron      1.948126e+09
Christopher Nolan  1.813228e+09
George Lucas       1.741418e+09
Joss Whedon        1.730887e+09
Robert Zemeckis    1.619309e+09
--------------------
```

Just like that. One line, no loops. Isn't it great?






