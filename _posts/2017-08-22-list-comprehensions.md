---
layout: post
title: "List Comprehensions"
date: 2017-08-22
---

I've recently discovered list comprehensions in Python, and, boy, does this make me excited.
Given my background in math, this is like finding a postcard from the past in my desk drawer, because they resemble [set builder notation](https://en.wikipedia.org/wiki/Set-builder_notation) from set theory! 

In Python, a list comprehension is another way of iterating through something and building a list. 
You could go with `for` loop:

```L = []
for x in range(10):
    if x% 2 == 1:
        L.append(x**2)
```
Or you could do the same with a list comprehension:

```L = [x**2 for x in range(10) if x% 2 == 1]```

Isn't it much neater? Some say that they are faster than `for` loops because, I qoute, "technically, 
they "run in a C speed", while "the for loop runs in the python virtual machine speed".
However, as with anything, turns out it depends and one needs to `timit` in each particular case.  
In any case, I find that it's a beautiful construct, and if it speeds things up - that's a good reason to learn to use them. 
It's not too hard to transform a `for` loop into a list comprehension either. 
Here's how you generally do it.
This is your loop:

``` 
new_list = []
for i in what_you_are_iterating_through:
    if condition_based_on_i:
        new_list.append(what_you_add_to_new_list)
```
This is a list comprehension you can transform it to:
```
new_list = [what_you_add_to_new_list for i in what_you_are_iterating_through if condition_based_on_i]
```

If it gets too long, you can add line breaks for the sake of better readability:
```
new_list = [
	what_you_add_to_new_list 
	for i in what_you_are_iterating through
	if condition_based_on_i
]
```
 
A shorter construct, readable, and typically faster. What's not to like?  

As a side note, I was surprised to see that there's no such intention action/refactoring in PyCharm. 
Perhaps, I'm missing something, but it seems like a doable IDE action. 
I've found a feature request for this and voted it up. 
If anyone else would like to see this implemented - feel free to add your vote as well [here](
https://youtrack.jetbrains.com/issue/PY-23018). 