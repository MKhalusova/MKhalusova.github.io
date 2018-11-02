---
layout: post
title: "Broadcasting in NumPy"
date: 2017-12-07
---

In arithmetic operations involving matrices and vectors (arrays) their shapes have to be compatible. For instance:

```python
import numpy as np

a = np.array([[1,2,3],[2,3,4],[3,4,5]])
b = np.array([[2,0,1],[2,0,1],[2,0,1]])
c = a * b
c
--------------------
array([[2, 0, 3],
       [4, 0, 4],
       [6, 0, 5]])
--------------------
```

This works just fine because both matrices (arrays) are 3x3, which means we can multiply them.
What NumPy broadcasting allows you to do is to perform arithmetic operations on arrays of different shapes, like this one:

```python
a = np.array([[1,2,3],[2,3,4],[3,4,5]])
b = np.array([2,0,1])
c = a * b
c
--------------------
array([[2, 0, 3],
       [4, 0, 4],
       [6, 0, 5]])
--------------------
```

Or this:

```python
a = np.array([[1,2,3],[2,3,4],[3,4,5]])
b = 2
c = a * b
c
--------------------
array([[ 2,  4,  6],
       [ 4,  6,  8],
       [ 6,  8, 10]])
--------------------
```

What happens in both cases is that the smaller array is "stretched" into an array with the same shape as the bigger one so that the calculation would be possible.

That sounds nice, but it gets even better when you think of what you would have to do without it.
Say, you need an array 3x4 array of fives. That's easy with broadcasting:

```python
a = np.ones((3,4)) * 5
a
--------------------
array([[ 5.,  5.,  5.,  5.],
       [ 5.,  5.,  5.,  5.],
       [ 5.,  5.,  5.,  5.]])
--------------------
```

What would you do if broadcasting wasn't there to help you out? Maybe something like this?

```python
a = np.ones((3,4))
for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        a[i,j] = a[i,j] * 5
a
--------------------
array([[ 5.,  5.,  5.,  5.],
       [ 5.,  5.,  5.,  5.],
       [ 5.,  5.,  5.,  5.]])
--------------------
```

Not only that's more code, if it's not a 3x4 array, but, say 1000 x 1000 it would take a lot longer. Let's see:

```python
import time

a = np.ones((1000,1000))
tic = time.time()
a = a * 5
toc = time.time()
print("with broadcasting: "+ str(1000*(toc-tic))+"ms")

a = np.ones((1000,1000))
tic = time.time()
for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        a[i,j] = a[i,j] * 5
toc = time.time()
print("with for loop: "+ str(1000*(toc-tic))+"ms")

--------------------
with broadcasting: 1.3642311096191406ms
with for loop: 629.5092105865479ms
--------------------
```

Broadcasting together with vectorization really speeds things up!
However, broadcasting can't always work. You can't multiply arrays with any random shapes, they have to be "broadcastable".

To check if broadcasting can be done between two arrays, NumPy compares their shapes, starting with the trailing values and working its way backwards.
If two dimensions are equal, or if one of them equals 1, the comparison continues. Otherwise, you'll see a `ValueError` raised. When one of the shapes runs out of dimensions, NumPy will use 1 in the comparison process until the other shape's dimensions run out as well. If the shapes are suitable for broadcasting, NumPy will “stretch” the smaller one to the shape of the bigger array.

The most typical uses of broadcasting in my experience so far (apart from operations between array and a scalar value) have been operations between a m by n matrix and a 1 by n vector.

In some cases, however, things get a little tricky.
Let's say we have a row vector (1,n) and a column vector(n,1) and we want to sum them up, what will NumPy do? Let's see:

```python
a = np.array([[1],[2],[3]])
b = np.array([3,4,5])
c = a+b
c
--------------------
array([[4, 5, 6],
       [5, 6, 7],
       [6, 7, 8]])
--------------------
```

Turns out both array were stretched to 3 x 3 shape and then summation was performed between them resulting in a 3 by 3 array.

So if you want to sum two vectors and get a vector, don't forget to reshape them.

```python
a = np.array([[1],[2],[3]]).reshape(3,1)
b = np.array([3,4,5]).reshape(3,1)
c = a+b
c
--------------------
array([[4],
       [6],
       [8]])
--------------------
```

Here's another example that wasn't obvious to me.
Consider the following code. It seems like it should work. Intuitively, `a` is an 2 x 4 array, and `b` looks like it has dimensions 2 by 1. However, it doesn't and we get a `ValueError`.

```python
a = np.random.random((2,4))
b = np.array([1,4])
c = a+b
c
--------------------
ValueError: operands could not be broadcast together with shapes (2,4) (2,)
--------------------
```

This is because `b` is (2,)-shaped array, so to make this work we need to reshape it to be (2,1)

```python
a = np.random.random((2,4))
b = np.array([1,4]).reshape((2,1))
c = a+b
c
--------------------
array([[ 1.32328729,  1.60995085,  1.96851683,  1.34086949],
       [ 4.04480947,  4.50054788,  4.20568576,  4.37104299]])
--------------------
```

Given these cases that can be not so obvious to beginners like me, it may be a good idea to explicitly reshape the arrays in the way you assume their shape to be before using broadcasting operations.
This may help avoid some hard to trace bugs in your calculations.

If you want to know more about NumPy broadcasting - check out [NumPy official documentation](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).







