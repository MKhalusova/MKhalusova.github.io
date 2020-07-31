---
date: "2017-08-31T00:00:00Z"
title: 'Pandas Data Structures: Series'
---
 
[Pandas](http://pandas.pydata.org/) is a massively popular python data manipulation and analysis library. It offers data structures and operations that make it easier to manipulate data. One of the fundamental data structures in pandas is Series. 

Series is a one-dimensional labeled array capable of holding any data type. The axis labels are collectively referred to as the index.

This post accumulates some of the things that I've learned about Series that I'd like to keep in one place for myself for a quick future reference. This is merely a basic intro to Series, and there's a lot more to this data structure. 

Before doing anything with Series, we need to import pandas:
```python
import pandas as pd
```

## Series Basics
You can create a Series from a list with any data type:
```python
fruit = ['banana', 'apple', 'orange', 42, ['blueberries','raspberries']]
my_fruit_series = pd.Series(fruit)
----------------
0                        banana
1                         apple
2                        orange
3                            42
4    [blueberries, raspberries]
dtype: object
----------------
```

By default, each item received an index label from 0 to N-1, where N is the length of the Series. 
You can find out the length of the Series by calling `len()` or `size`:
```python
len(my_fruit_series)
----------------
5
----------------
``` 

```python
my_fruit_series.size
----------------
5
----------------
```
If you want to set the indices yourself, you can do that too:
```python
fruit = ['banana', 'apple', 'orange', 42, ['blueberries','raspberries']]
my_fruit_series = pd.Series(fruit, index=list('abcde'))
----------------
a                        banana
b                         apple
c                        orange
d                            42
e    [blueberries, raspberries]
dtype: object
----------------
```
You can also create a Series from a dictionary, in this case keys will be used to build the index.
```python
GOT_cast = {'Tyrion Lannister': 'Peter Dinklage', 'Cersei Lannister': 'Lena Headey',
           'Daenerys Targaryen':'Emilia Clarke','Jon Snow':'Kit Harington'}

GOT_series = pd.Series(GOT_cast)
----------------
Cersei Lannister         Lena Headey
Daenerys Targaryen     Emilia Clarke
Jon Snow               Kit Harington
Tyrion Lannister      Peter Dinklage
dtype: object
----------------
```
Sometimes it's useful to create a large Series of random numbers, for example, here's a Series with 10000 numbers with random numbers from 0 to 500.
```python
import numpy as np
random_numbers = pd.Series(np.random.randint(0,500,10000))
```
If you have a large Series, you may want to use `head()` or `tail()` method to, sort of, preview it.
By default `head()` shows the first 5 elements of the Series, and the `tail()` shows the last 5 elements, however, you can specify any other number. 
```python
GOT_cast = {'Tyrion Lannister': 'Peter Dinklage', 'Cersei Lannister': 'Lena Headey',
           'Daenerys Targaryen':'Emilia Clarke','Jon Snow':'Kit Harington',
           'Sansa Stark':'Sophie Turner', 'Arya Stark':'Maisie Williams',
           'Jaime Lannister':'Nikolaj Coster-Waldau', 'Jorah Mormont':'Iain Glen',
           'Theon Greyjoy':'Alfie Allen','Samwell Tarly':'John Bradley'}

GOT_series = pd.Series(GOT_cast)
GOT_series.head(3)
----------------
Arya Stark            Maisie Williams
Cersei Lannister          Lena Headey
Daenerys Targaryen      Emilia Clarke
dtype: object
----------------
```
If you want to get all the values and don't really care about the index, you can get the array of values with values:
```python 
GOT_series.values
----------------
['Maisie Williams' 'Lena Headey' 'Emilia Clarke' 'Nikolaj Coster-Waldau'
 'Kit Harington' 'Iain Glen' 'John Bradley' 'Sophie Turner' 'Alfie Allen'
 'Peter Dinklage']
----------------
```
You can get the Index object too:
```python 
GOT_series.index
----------------
Index(['Arya Stark', 'Cersei Lannister', 'Daenerys Targaryen',
       'Jaime Lannister', 'Jon Snow', 'Jorah Mormont', 'Samwell Tarly',
       'Sansa Stark', 'Theon Greyjoy', 'Tyrion Lannister'],
      dtype='object')
----------------
```
Speaking of indices, let's see how we can select something out of a Series.
If we know the label where the value is we can use `.loc`, or `[]`
```python 
GOT_series.loc['Tyrion Lannister']
----------------
Peter Dinklage
----------------
```
```python 
GOT_series['Tyrion Lannister']
----------------
Peter Dinklage
----------------
```
If the label isn't in the index, it'll raise a `KeyError`: 
```python 
GOT_series.loc['Bronn']
----------------
Traceback (most recent call last):
  File "/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/pandas/core/indexing.py", line 1434, in _has_valid_type
    error()
  File "/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/pandas/core/indexing.py", line 1429, in error
    (key, self.obj._get_axis_name(axis)))
KeyError: 'the label [Bronn] is not in the [index]'
----------------
```
But you can use a non-existent index to “append” the series: 
```python 
GOT_series.loc['Bronn'] = 'Jerome Flynn'
GOT_series
----------------
Arya Stark                  Maisie Williams
Cersei Lannister                Lena Headey
Daenerys Targaryen            Emilia Clarke
Jaime Lannister       Nikolaj Coster-Waldau
Jon Snow                      Kit Harington
Jorah Mormont                     Iain Glen
Samwell Tarly                  John Bradley
Sansa Stark                   Sophie Turner
Theon Greyjoy                   Alfie Allen
Tyrion Lannister             Peter Dinklage
Bronn                          Jerome Flynn
dtype: object
----------------
```
It's also possible to refer to integer location based index using `.iloc`.
```python 
GOT_series.iloc[2]
----------------
Kit Harington
----------------
```

`.loc`, `.iloc`, and also `[]` indexing can accept a callable as indexer, which I find pretty cool. So you can do something like this:
```python 
random_numbers = pd.Series(np.random.randint(0,500,10000))
random_numbers.loc[lambda s: s > 400].head()
----------------
17    451
19    442
22    488
24    431
32    479
dtype: int64
----------------
```
At first, it seemed to me that the `where()` method would return the same as selection by callable, given the same condition, but there's a big difference. It returns a series of exactly the same shape with those values that match the condition exactly where they are, and the rest is `NaN`.
```python 
random_numbers.loc[lambda s: s > 5]
random_numbers.where(random_numbers>5)
----------------
3    9
6    9
7    6
9    9
dtype: int64
0    NaN
1    NaN
2    NaN
3    9.0
4    NaN
5    NaN
6    9.0
7    6.0
8    NaN
9    9.0
dtype: float64
----------------
```

By default `where()` returns a copy and doesn't modify the original data. There is an optional parameter inplace (`inplace=True`) so that the original data can be modified without creating a copy.

## Some Math with Series
Let's take this Series as an example:
```python
s = pd.Series([3,12,1,7,15])
----------------
0     3
1    12
2     1
3     7
4    15
dtype: int64
----------------
```
You can get the sum of the values with `sum()` from `numpy`:
```python
total = np.sum(s)
total
----------------
38
----------------
```
You can add N to each item in Series using broadcasting (same goes for division, multiplication, subtraction):
```python
s+=2
----------------
0     5
1    14
2     3
3     9
4    17
dtype: int64
----------------
```
You can do this:
```python
s = pd.Series([3,12,1,7,15])
s1 = pd.Series([1,2,3,4,5,6])
s+s1
----------------
0     4.0
1    14.0
2     4.0
3    11.0
4    20.0
5     NaN
dtype: float64
----------------
```
It doesn't matter if the length of two Series is different, or even if the indices are not an exact match.  The result of an operation between unaligned Series will have the union of the indexes involved. If a label is not found in one Series or the other, the result will be marked as missing `NaN`.
 
With `describe()` you can get a quick statistic summary of your data:
```python
random_numbers = pd.Series(np.random.randint(0,500,10000))
random_numbers.describe()
----------------
count    10000.000000
mean       251.338500
std        145.344209
min          0.000000
25%        124.750000
50%        252.000000
75%        379.000000
max        499.000000
dtype: float64
----------------
```
To see if two Series are exactly the same (both indices and values), you can use `equals()`:
```python
s = pd.Series([1,2,3,4,5])
s1 = pd.Series([1,2,3,4,5], index=list('abcde'))
s.equals(s1)
----------------
False
----------------
```
```python
s = pd.Series([1,2,3,4,5])
s2 = pd.Series([1,2,3,4,5])
s.equals(s2)
----------------
True
----------------
```
You can do element-wise comparisons with a scalar value:
```python
s = pd.Series(['foo', 'bar', 'baz'])
s == 'foo'
----------------
0     True
1    False
2    False
dtype: bool
----------------
```
You can locate labels of the minimum and maximum values with the `idxmin()` and `idxmax()` functions:
```python
s = pd.Series([3,12,1,7,15], index=list('abcde'))
s.idxmax()
s.idxmin()
----------------
e
c
----------------
```
## Modifying Data in a Series
Replacing values in a Series with a new value:
```python
s = pd.Series([3,7,12,1,7,15])
s.replace(7,777)
----------------
0      3
1    777
2     12
3      1
4    777
5     15
dtype: int64
----------------
```
### Modifying data with apply()
Let's say we have a Series with heights of my imaginary friends in cm:
 
```python
s = pd.Series([175,168,154,183], index=['Tim', 'Kate', 'Ann', 'Jon'])
----------------
Tim     175
Kate    168
Ann     154
Jon     183
dtype: int64
----------------
```
If I want to convert their height from cm to inches, I can do it with `apply()`:

```python
s = pd.Series([175,168,154,183], index=['Tim', 'Kate', 'Ann', 'Jon'])
s = s.apply(lambda x: x/2.54)
----------------
Tim     68.897638
Kate    66.141732
Ann     60.629921
Jon     72.047244
dtype: float64
----------------
```
Or, I can define a function, and pass it:

```python
s = pd.Series([175,168,154,183], index=['Tim', 'Kate', 'Ann', 'Jon'])
def convert_cm_to_inch(x):
   return x/2.54
s = s.apply(convert_cm_to_inch)
----------------
Tim     68.897638
Kate    66.141732
Ann     60.629921
Jon     72.047244
dtype: float64
----------------
```

If the function takes more arguments, you can specify them with `args=`
```python
s = pd.Series([175,168,154,183], index=['Tim', 'Kate', 'Ann', 'Jon'])
def add_height_of_their_hat(x, hat_height):
   return x+hat_height
s = s.apply(add_height_of_their_hat, args=(15,))
----------------
Tim     190
Kate    183
Ann     169
Jon     198
dtype: int64
----------------
```
I'd say that should do it for beginning with Series, and in the next post I'll cover DataFrame.




