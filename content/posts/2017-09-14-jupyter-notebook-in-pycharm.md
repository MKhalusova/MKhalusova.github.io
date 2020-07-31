---
date: "2017-09-14T00:00:00Z"
title: Jupyter Notebook in PyCharm
---

I find that [Jupiter Notebook](http://jupyter.org/about.html) is a really nice tool for dealing with data, and for learning and exploring the capabilities of various libraries. It makes it easy to quickly experiment with data, see where cleanup is needed, what works and what doesn't, and to plot the results adjusting on the go. It's also super-easy to setup Jupyter in PyCharm.
Here's how you do it.

Create a project in PyCharm, and create a VirtualEnv instead of using your local python.  

![Create a Project](/images/jup-create-ve.png)

Go to Preferences (*Cmd,*), and install packages you'll be using - there's a little plus button under the list of the installed packages. 
You'll need at least `jupyter` and `matplotlib`, but probably also `pandas`, `numpy`, `seaborn`, and maybe something else. 

![Add packages](/images/jup-packages.png)

Now it seems like you're set, but not really. 

![Almost there](/images/jup-almost-there.png)

Add some code in the cell, and click Run (or do it like a pro and use the shortcut - *Shift+Enter*):

![Run](/images/jup-auth.png)

Oops, what's that? We're not done with the setup. Click Cancel here, and you'll get the link in the Run tool window to authenticate you the first time:

![Run toolwindow](/images/jup-link.png)

Click it, and you've got your Notebook running. Now you're really all set:

![Yay! It's working](/images/jup-running.png)

That's pretty much it. Open your notebook (*.ipynb*), add code in the cells, run them, experiment, plot, have fun! 
For my example I took some statistical data from [muenchen.de](https://www.opengov-muenchen.de/) about people visiting movies, museums and classical music concerts. And, well, no surprise there, a lot more people go to see movies rather than listen to classical music. :D

![Not very insightful plot example](/images/jup-plot.png)

