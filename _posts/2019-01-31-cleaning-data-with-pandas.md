---
layout: post
title: "Cleaning data with pandas"
date: 2019-01-31
---

Whether you want to do an exploratory data analysis, or train a machine learning mode, the first thing you 
inevitably will have to do is clean the data you've got. The only datasets that don't require pre-processing 
in one way or another, are the example datasets that come with machine learning libraries. Otherwise, all 
the data encountered "in the wild" needs to be wrangled before you can use it. In this post I'll only touch on 
tabular data and things that you may need to do to clean it/prepare it, such as: 
* Fix formats
* Deal with dates
* Deal with missing values

## Data format
Upon loading your data in a pandas DataFrame, it's a good idea to check what data types you have in various columns. 
You can do it by calling `dtypes` on the dataframe. 

![df.dtypes](/images/jan31-dtypes.png)

Sometimes you may find out that for whatever reason a column's values were stored in a wrong format. It can be, 
for example, that a column with strings actually contains counts of items, i.e. integers. Or you may want to 
convert a `float64` column to an `int` to use memory in a more efficient manner. 

You can change the data format using `astype`:

![astype](/images/jan31-astype.png)

## Date
Date is an important piece of information that you can extract a number of features from and you'll see it often 
in tabular data - purchase records, publishing date of some content, logs, and so on. And quite often you'll need 
to make sure to parse it properly. Luckily Pandas provides tools to parse a flexibly formatted string date:

![to_datetime](/images/jan-31-date_time.png) 

You can also use format codes to specify the desired output, e.g.:

![date_format](/images/jan31-date-format.png)

Sometimes the dates you have are super precise and what you really need is to group events by, say, an hour. 
You can either truncate the date:
 
![truncate the date](/images/jan31-truncate.png)

Or round it:

![round the date](/images/jan31-round.png)

## Dealing with missing values
Another common problem with data is missing values. It could be due to a field not filled out by a human, or 
parts of records being lost due to errors, or some data not applicable or not available. Whatever the cause, 
missing values are a problem: not many algorithms can work with them. That is why in most cases you'll need to 
do address this issue. 

To check if there are any missing values in a pandas dataframe you can call `df.info()` and see if the total 
number of entries matches the number of non-null entries in each column. For me, however, it seems a little easier to 
spot problematic columns with `df.isnull().sum()`. This simply shows how many missing values there are per column.

![nans](/images/jan31-isnull-sum.png)

The first thing you can do when you have some missing values in the data, is to see whether you can still 
re-collect that data yourself or contact the team that produced it. Perhaps, it's still possible to retrieve what's 
missing. 
 
Otherwise, you can either drop the rows with missing values (`df.dropna()`) or fill them with some value. Dropping 
the rows may be a good choice when there's very few of them compared to the size of the dataset but of course 
it always depends on each particular case and you'll need to assess whether dropping any data is reasonable.

If dropping rows with missing values is not an option, the alternative is to replace them with some other values, 
and you have a choice here what those values can be. 

You can: 
* Check if a missing value actually means anything. E.g. in a house without a garage, a `NaN` in the field for 
"how many cars can fit in the garage" probably means 0. In some cases a `NaN` is actually one of the categories.
![fillna zeros](/images/jan31-fillna-zero.png)

* Replace NaNs with the mean or median for the numerical variables, or the most common value for categorical variables.
![fillna mean](/images/jan31-fillna-mean.png)

* If the data is ordered, it may make sense to take one of the adjacent values as a substitute, either next one or the previous.
* Replace NaNs based on the other data available in your dataset. 
![fillna by group](/images/jan31-lf.png)

In this example we assume that the lot frontage ("LF") is going to be more or less the same across a neighborhood, so 
we fill in the missing values based on the median in a group based on a neighborhood. 

Whichever approach you use, be very careful - by replacing the missing data you are introducing your own assumptions 
and some noise to the data, and in machine learning the "garbage in - garbage out" rule always works. 
So it's always better to retrieve the missing data whenever possible.



