---
date: "2020-12-15T00:00:00Z"
title: 'Part of speech tagging with Hidden Markov Model in Kotlin'
draft: true
---

In NLP, part-of-speech tagging is a process in which you mark words in a text (corpus) as corresponding parts of speech 
(e.g. noun, verb, etc.) taking into account context, because some words can be ambiguous. For example, take the word 
"sleep". In a sentence "I plan to sleep until lunch tomorrow.", this word is a verb. However, in a sentence "How much sleep 
does one really need?", it is a noun.  

Part-of-speech tagging is fundamental to natural language processing and has multiple applications, such as 
[Named Entity Recognition](https://en.wikipedia.org/wiki/Named-entity_recognition), 
[Word Sense Disambiguation](https://en.wikipedia.org/wiki/Word-sense_disambiguation), and more.

How do you go about this problem? There are different approaches, but the most common ones these days would be: 
- Probabilistic methods: In these methods, you assign a tag based on the probability of it occurring in a given sentence.
One of such approaches uses Hidden Markov Models, and that's what this post covers. 
- Deep Learning methods: Getting state-of-the-art results requires complex recurrent neural networks, such as 
 Meta-BiLSTM Model described in [this paper](https://arxiv.org/abs/1805.08237). 
 
 In this post, I've chosen to cover Hidden Markov Models for a couple of reasons. First, this model is amazingly versatile, 
 and part-of-speech tagging is only one its applications. Others include signal processing, speech recognition, 
 handwriting recognition, musical score following and much more... Perhaps, you'll find your own application for it too!
Second, it took me some time to understand the algorithm used for retrieving tag predictions, the Viterbi algorithm. So 
by explaining how it works I hope to help some of you understand it :) Finally, if you really would rather read about 
 deep learning, don't worry, I'll be digging into some version of LSTM in one of the later articles.
 
### What you will learn:
Alright, so what will you learn from this post? I'll explain what a Markov Model is, and what Markov property is, then 
we'll move to the Hidden Markov Model. I'll show you how you can implement one in Kotlin. 
Next, we'll move on to the Viterbi algorithm - what's the motivation for it, how it works, and how to implement it. 
Finally, just so that you don't get an impression that you have to do everything from scratch, I'll show you how you 
can use POS Taggers available in libraries.

## Data & Some Preprocessing
   
  
## Markov Model and Markov Property

## Hidden Markov Model

## Viterbi Algorithm: Motivation

## Viterbi: Initialization

## Viterbi: Forward Pass

## Viterbi: Backward Pass

## POS Tagging with OpenNLP
 