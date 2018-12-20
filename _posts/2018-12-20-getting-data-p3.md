---
layout: post
title: "Getting data, part 3: APIs"
date: 2018-12-20
---

In this last part of "getting data" sub-series, I want to mention, without going into too much detail, one more way of 
obtaining data that you may need for your Data Science project. 

Many websites, services and tools provide APIs that you can use to request data in a structured format without the need 
for scraping. The most common format of data that you'll get through APIs is going to be JSON, but if you're unlucky, 
you'll get XML which you can parse with BeautifulSoup. 

If you're not familiar with JSON format, here's an detailed introductory tutorial on what the format is and how you can 
work with it using Python's `json` library: [Working With JSON Data in Python](https://realpython.com/python-json/).

But first, you need to get your hands on one! Some websites and platforms allow you to do a few simple things with their 
APIs without authentication, for example, [GitHub](https://developer.github.com/v3/). For instance, it's very easy to get 
a list of Python dictionaries each representing a public repository in my GitHub account.

```python
import requests
import json

endpoint = "https://api.github.com/users/mkhalusova/repos"
repos = json.loads(requests.get(endpoint).text)
print(repos)
```

However, in most cases to work with APIs you'll need to use authenticate yourself before you can request any data. You 
may need to use credentials, request API keys, access tokens, etc. All of these things are, of course, described in the 
API documentation of the website of a tool you're trying to connect to. These docs are the perfect starting point if 
you intend to write your own script or library to authenticate yourself and request data. On the other hand, if you're 
trying to use APIs of any popular service (Twitter, Google services, GitHub, etc.), there are already libraries available 
written by someone before you. So instead of re-inventing the wheel, you can simply use those libraries. 

Here are some useful libraries and tutorials on how to use them. 

***Twitter API***

There are quite a few libraries that let you use Twitter API, but I find that [Tweepy](https://github.com/tweepy/tweepy) 
works really well for me: it's simple, well-documented and provides a good amount of functionality. 
For an example on how to use it, check out [this tutorial](http://socialmedia-class.org/twittertutorial.html) 
by [Wei Xu](https://twitter.com/@cocoweixu) and [Jeniya Tabassum](https://twitter.com/@JeniyaTabassum).   

***Reddit API*** 

[PRAW](https://praw.readthedocs.io/en/latest/) ("Python Reddit API Wrapper") is a python library that allows for simple 
access to reddit’s API. It is quite easy to use and is designed to follow all of reddit's API rules. 
Here's a step by step tutorial on how to get some data for your data science project using PRAW, by [Felippe Rodrigues](https://twitter.com/fsorodrigues) : 
http://www.storybench.org/how-to-scrape-reddit-with-python/

***YouTube API***

Google provides their own neat API libraries to access their services, for Python you'll need Google APIs Client Library 
for Python (`pip install --upgrade google-api-python-client`). 
They also have great [documentation](https://developers.google.com/youtube/v3/quickstart/python).
For an example on how to use YouTube APIs to pull out some specific data check out this 
[blog post](https://medium.com/greyatom/youtube-data-in-python-6147160c5833) by [Sudharsan Asaithambi](https://medium.com/@sudharsanasai).


For more web services, links to their APIs and Python wrappers check out this [github repo](https://github.com/realpython/list-of-python-api-wrappers).

Now if at any point during the holidays you feel bored, you can get yourself some data to wrangle. Happy holidays! :) 







