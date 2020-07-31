---
date: "2017-07-24T00:00:00Z"
title: Ready, Steady, Blog! GitHub.io, Jekyll, Docker
---

I've finally got around to setting up a blog. As you can guess from the title, I've picked github.io + jekyll + docker combo. 
I know, all the cool people have been playing with it 3 years ago, but I've only got my hands on it now. 
 
While bringing all these together, I've had a couple of "WTF?" moments and had to google for answers. 
So once I've put everything together, I thought I'd share a number of useful links and hints 
that helped me make it work:  

* [Perfectly clear official intro on how to start with GitHub Pages](https://pages.github.com/) 
* [Great guide on Jekyll by Jonathan McGlone](http://jmcglone.com/guides/github-pages/)
* To preview the blog locally I've installed [Jekyll locally](https://help.github.com/articles/setting-up-your-github-pages-site-locally-with-jekyll/). You'll also need ruby. I'm using [HomeBrew](https://brew.sh) to install and update ruby.
* You can also use jekyll in docker: [official image](https://hub.docker.com/r/jekyll/jekyll/) 

I've had some trouble making docker work, kept receiving an connection timeout, but managed to make it work in the [end](https://github.com/docker/for-mac/issues/1601).

The moment it all finally came together, someone told me that jekyll is no longer hot, and I should try [HUGO](https://gohugo.io/). Sigh... Some other time. It works, I don't yet feel like messing with a running system. 


   







