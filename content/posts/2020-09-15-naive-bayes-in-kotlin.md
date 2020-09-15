---
date: "2020-09-15T00:00:00Z"
title: 'Baseline Sentiment Analysis with Naive Bayes in Kotlin'
draft: true
---
The other weekend I implemented a simple sentiment classifier for tweets in Kotlin with Naive Bayes. 
I originally meant it as a practice exercise for me to get more comfortable with Kotlin, but then I thought that perhaps 
this can also be a good topic to cover in a blog post. So if you are familiar with Kotlin and are curious about 
NLP (natural language processing) this article can help you to get started with some basic NLP. 

So what will you learn from this post? I'll show you how you can implement a Naive Bayes model to classify positive vs 
negative sentiment of tweets, I'll explain how Naive Bayes works, what Bayes' Theorem is, what kind of text preprocessing
 you'll need to do, and what are the limitations of this algorithm. 

At the end of this blog post, you'll find a link to a GitHub repo with Naive Bayes implementation, and a working example
 for tweet classification. Ready to get started? Here we go!

## What is sentiment analysis useful for?
While classifying tweets into positive and negative may seem like a "toy project" kind of activity at a first glance, 
there are real world applications for tools that can do this task well. For instance, if your company announces 
a product X, and thousands of people start tweeting about it, you may want to quickly get an idea of how the product 
has been received - are most people happy about it, or not. I'm sure there are a myriad of startups that offer 
such a service :)

You too will be able to classify text into categories by the end of this article! 
I'll be using Naive Bayes here, which is a simple and fast algorithm but, as the name suggests, it's a little "naive" 
and won't catch on to language structure. Still, it gives you a great baseline model, and once you understand how it works, 
you can apply this algorithm to many other tasks with minor modifications.

## Naive Bayes
Generally speaking, [Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) is a probabilistic classifier 
based on applying [Bayes' Theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem). This brings up a few 
questions - what is Bayes' Theorem, what kind of probabilities are we talking about and where do we get them from?

In this case, what we essentially are going to do is build a vocabulary of all words in our training data, and 
for each word calculate the probability to encounter this word in a positive tweet, and a probability to encounter 
this word in a negative tweet. Then, to predict whether a new tweet is positive or negative, we'll look at the 
words it contains, and use previously calculated probabilities with a few mathematical tricks to arrive at a prediction.
This description skips a ton of details. Let's dig into them!

## Bayes' Theorem
 
Before I show you the formula, consider this example. Say, your friend has 2 coins, one of them is fair (one side is 
heads, and the other one is tails), and the second coin is unfair (both sides are tails). The friend picks a 
coin at random without you looking. If I ask you now to tell me if your friend picked an unfair coin, you won't be 
able to give me a definitive answer - there's after all a 50-50 chance of picking either coin. Suppose your friend now 
tosses the coin once, and you see tails. You may get a bit more confident it is an unfair coin.
What if your friend tossed this coin 10 times and all the times they got tails? In this case you would be almost 
certain the coin was unfair. 
So you already have the intuition for the theorem! Presented with new evidence, you can update your estimate for 
the probability of an event (e.g. coin being unfair).

Here's the basic formula:

 $$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$$
 
Here A and B are events, and $P(A)$ and $P(B)$ are their respective probabilities.
* $P(A|B)$ = probability of A given B.
* $P(B|A)$ = probability of B given A. 
 
In the example above, we want to know what is the probability of the coin being unfair given we've just seen 10 tosses 
all resulting in tails, so
* $A$ == event "picked coin is unfair"
* $B$ == event "all 10 tosses result in tails".  
* $P(A)$ = 0.5
* $P(B|A)$ = probability that we'll get 10 tails if we picked the unfair coin = 1 
However, like in this case, we often don't know $P(B)$, and for such cases, there's an extended formula: 

$$P(A|B) = \frac{P(B|A)*P(A)}{P(B|A)*P(A)+ P(B|notA)*P(notA)}$$ 

It's outside of the scope of this article to prove why it's the same, so you can just trust me on this, or by all means 
 dig deeper and research it further. Here's a [great video](https://youtu.be/HZGCoVF3YvM) explaining the theorem in detail. 
 
So now we need $P(B|notA)$ and $P(not A)$ and we can plug the numbers into the formula and get the answer. 
* $P(B|notA)$ = probability that we'll get 10 tails if we picked a fair coin = $0.5^{10}$
* $ P(notA)$ = probability that we did not pick unfair coin = 0.5

And the probability that the picked coin is unfair given it was tossed 10 times and all we've seen were tails is … 

$$\frac{1 * 0.5}{1 * 0.5+ 0.5^{10} * 0.5} \approx 0.999$$

Your intuition is now confirmed with math. It's nearly guaranteed that the coin was unfair. 
This formula has numerous uses. In our case, the intuition is that the probability that a tweet is positive 
should probably be higher if we know it contains words that are more often encountered in positive sentences, e.g. 
"happy", "awesome", "good", etc.

## Naive Bayes Classifier

The formula above was rather simple to give you intuition on how the theorem works. Now how does that apply to a tweet 
classification problem? We'll be looking at a tweet as a set of N words. It's important to note that this algorithm 
does not take word order into account, only the presence of words. 

Based on the training data, we can calculate for each word, how often it is encountered in positive tweets, 
and how often - in negative. 

These frequency counts allow us, in turn, to calculate for each word in the corpus the probability to encounter it 
in positive examples, and probability to encounter it in negative examples. 

So now, for a new tweet, to predict if it's positive or not, we will need to plug the numbers in the following formula:

$$\frac{P(\text{Positive Class})}{P(\text{Negative Class})} \displaystyle\prod_{i=1}^{m} \frac{P(\text{word}_i | \text{Positive Class})}{P(\text{word}_i | \text{Negative Class})} $$

where $\text{word}_i$ is each word/token in that tweet.

Once we plug the numbers, if the result is larger than 1 (positive probabilities overpower the negative ones), 
then we can predict that the sentiment is positive, and if the result is less than 1, we can predict negative sentiment. 
Hooray! We've arrived at the solution! Right? 

Well... almost. 
Unfortunately it can happen that some of those probabilities equal to 0, then the whole formula will blow up. In fact, 
we can only get a meaningful result when none of those probabilities equal to 0. How can we ensure that? 
With a trick called Laplacian smoothing. Here's how we're going to calculate probability to encounter a word in a class: 

$$P(\text{word}_i | \text{class}) =\frac{freq(\text{word}_i , \text{class}) + 1}{sum(\text{freq}) + V}$$

Here: 
* $freq(\text{word}_i , \text{class})$ = how often this word is encountered in examples of this class (we have positive and negative classes)
* $sum(\text{freq})$ = sum of frequences for all words in the vocabulary
* $V$ = number of unique words in the vocabulary. 

By making this small adjustment, we can make sure we won't end up with probabilities equal to 0.

However, we're not done yet. There's another problem we will almost certainly face. Some words may be so rare, and the 
corpus may be so large, that ultimately the resulting probabilities for them will be so small, they'll cause 
arithmetic underflow. To address this problem instead of using the formula directly, we'll take logarithm:

$$ ln(\frac{P(\text{Positive Class})}{P(\text{Negative Class})} \displaystyle\prod_{i=1}^{m} \frac{P( \text{word}_i | \text{Positive Class})}{P(\text{word}_i | \text{Negative Class})})  = ln(\frac{P(\text{Positive Class})}{P(\text{Negative Class})}) + \sum_{i=1}^m ln(\frac{P(\text{word}_i | \text{Positive Class})}{P(\text{word}_i | \text{Negative Class})}) $$   

As a nice side effect, the result is also easier to interpret. If the result is less than 0, we predict negative class, 
otherwise we predict positive class. And now we're done with formulas! Time to put it all together. 

## Putting it all together

First of all, we need data. If you want to use your own data, then you're going to need to collect and label it. 
For the purposes of this article I took a dataset with positive and negative Tweets that comes with Python NLTK library. 
You can find it [here](http://www.nltk.org/nltk_data/), it’s number 41 on the list. The data is stored in json format in 
two files: `positive_tweets.json` and `negative_tweets.json`. So in this case the label is in the file name.    
There's ton of metadata in those files that I don't need, so I'll use [klaxon](https://github.com/cbeust/klaxon) to get just the "text". 

```kotlin
fun extractTweetsFromJSON (path: String): List<String> {
    val tweets = mutableListOf<String>()
    val parser = Parser()
    File(path).forEachLine { line -> tweets.add((parser.parse(StringBuilder(line)) as JsonObject).string("text").toString()) }
    return Collections.unmodifiableList(tweets)
}
```

### Preprocessing 

After reading the files, I'll have a list of positive tweets, and a list of negative tweets.
Next, I need to preprocess the tweets to remove whatever I won't need for the analysis.  
I decided to get rid of the following:  
* Stock market tickers like $GE
* Old style RT
* URLs
* Hashtags
* Mentions
* XML character encodings like `&amp;`
* Extra spaces

I found Kotlin's extension functions super useful for this task. 
```kotlin
    private fun String.removeTickers() = replace(Regex("\\\$\\w*"), "")
    private fun String.removeRTs() = replace(Regex("^RT[\\s]+"), "")
    private fun String.removeURLs() = replace(Regex("https?://[-a-zA-Z0-9+&@#/%?=~_|!:,.;]*[-a-zA-Z0-9+&@#/%=~_|]"), "")
    private fun String.removeHashtags() = replace("#", "")
    private fun String.removeMentions() = replace(Regex("[@#][\\w_-]+"), "")
    private fun String.removeXMLEncodings() = replace(Regex("&[a-z]*;")," ")
    private fun String.removeExtraSpaces() = replace(Regex("\\s+")," ")
```
How exactly you clean the data depends on what you're working with, and will differ from task to task.

In the next step I need to split the text of each tweet into words (tokens). Token is a more general 
term, as it can be a word, an emoji, a sequence of numbers, etc.  In this example I’ve removed numbers, punctuation 
signs, and emojis. Under other circumstances, I could've left the emojis as tokens. They can potentially be good 
predictors of the sentiment. In this case, however, the data has been collected based on emojis, so leaving them is 
cheating, and won't give me trustworthy results. 

```kotlin
class Tokenizer {
    val emojisRegex = Regex("(?:[<>]?[:;=8][\\-o*']?[)\\](\\[dDpP/:}{@|\\\\]|[)\\](\\[dDpP/:}{@|\\\\][\\-o*']?[:;=8][<>]?|<3)")

    fun tokenize(string: String, leaveEmojis: Boolean): List<String> {
        val emojiMatches = emojisRegex.findAll(string)
        val emojisList = emojiMatches.map { it.value }.toList()
        val withoutEmojis = string.replace(regex = emojisRegex, replacement = "")

        // dropping leftover punctuation and numbers, removing extra white spaces
        val withoutPunctuation = withoutEmojis.replace(regex = Regex("[^a-zA-Z_-]"), replacement = " ").replace(regex = Regex("\\s+")," ").trim()

        // splitting the string into tokens
        val tokensWithoutEmojis: List<String> = withoutPunctuation.split(" ")
        val lowercaseTokens = tokensWithoutEmojis.map { it.toLowerCase() }

        return if (leaveEmojis) lowercaseTokens + emojisList
        else lowercaseTokens
    }

}
```

Once I have each tweet as a list of tokens, I can also remove so called stop words. These are words that are way too 
common in any text and won't add much value. Words like "the", "and", "I", "do" can be easily discarded. 
  
Normally, the final step of data preparation is stemming - reducing words to their word stem/root form. For example, 
words like "beauty", "beautiful"  and "beautify" will have the same representation - "beauti". This is helpful when 
you have a massive text corpus - you'll end up with a smaller vocabulary after stemming. A smaller vocabulary means 
faster computation, and when stemming is done correctly, you don't lose much meaning. 
In my example I didn't stem the words for two reasons: 
* My data is rather small, it wouldn't make that much of a difference
* I want to explore different stemming approaches and, perhaps implement one in Kotlin and write another article 
about it later ;)

Finally, we can get to the algorithm implementation!

### Naive Bayes Classifier implementation in Kotlin
  
What does the classifier need to have? 
* A way to train from data passed to it.  
* A method to generate predictions
* A method for evaluating performance on test data. I'll be using the simplest metric - accuracy.

First, let's build the frequency table of the words, it will be a map of each word to the Pair of its positive and 
negative counts. When done, we'll know how often every word in the vocabulary is encountered in positive tweets, and 
how often in negative.     
 
```kotlin
    private fun buildFrequences(texts: List<List<String>>, targets:List<Int>): Map<String, Pair<Int, Int>>{
        // texts - list of tokenized tweets, targets = labels (will need to combine positive and negative tweets)
        // frequency table of word to Pair<negative (0) count , positive (1) count>
        val frequencyTable = mutableMapOf<String, Pair<Int,Int>>()
        for ((tweet, y)  in texts.zip(targets)) {
            for (word in tweet) {
                val counts = frequencyTable.getOrDefault(word, Pair(0,0))
                if (y == 0) frequencyTable.put(word, Pair(counts.first + 1, counts.second))
                if (y == 1) frequencyTable.put(word, Pair(counts.first, counts.second + 1))
            }
        }
        return frequencyTable
    }
```

Once we have the frequencies we can calculate this part of the prediction equation for each word - $ln(\frac{P(word_i | Positive Class)}{P(word_i | Negative Class)})$

```kotlin
    private fun computeLogLambdas(freqs: Map<String, Pair<Int, Int>>): Map<String, Double> {
        val allPositiveCounts = freqs.values.sumBy { it.second }
        val allNegativeCounts = freqs.values.sumBy {it.first}
        val vocabLength = freqs.size

        val logLamdas = mutableMapOf<String, Double>()

        for (word in freqs.keys) {
            // counting probabilities with Laplacian smoothing to avoid 0s
            val posProb = ((freqs.getValue(word).second + 1).toDouble() / (allPositiveCounts + vocabLength))
            val negProb = ((freqs.getValue(word).first + 1).toDouble() / (allNegativeCounts + vocabLength))
            val logLambda = ln(posProb/negProb)
            logLamdas[word] = logLambda
        }
        return logLamdas
    }
```

Now we have all pieces to train the model: 
```kotlin
    fun train(X: List<List<String>>, Y:List<Int>) {
        require(X.size == Y.size) {"Size of X doesn't match size of Y"}
        this.vocabulary = computeLogLambdas(buildFrequences(X, Y))
        val probPos = ((Y.count { it == 1 }).toDouble()/Y.size)
        val probNeg = ((Y.count { it == 0}).toDouble()/Y.size)
        this.logPrior = ln(probPos/probNeg)
    }
```

To generate a prediction, we can either get the likelihood: 
```kotlin
    fun predictLikelihood(x: List<String>): Double {
        var result = this.logPrior
        for (token in x) {
            result += this.vocabulary.getOrDefault(token, defaultValue = 0.0)
        }

        return result
    }
```

Or we can return the label: 
```kotlin
    fun predictLabel(x: List<String>): Int {
        return if (this.predictLikelihood(x) >= 0) 1
        else 0
    }
```

Finally, it's helpful to know how the classifier will behave on unseen data, and to evaluate that, you're going to need 
a metric. I've written a whole bunch of posts about evaluation metrics, but here I've just implemented the most basic 
one - accuracy. Accuracy is going to tell you the proportion of correct predictions out of all predictions. 
```kotlin
    fun score(xTest: List<List<String>>, yTest:List<Int>): Double {
        require(xTest.size == yTest.size) {"Size of X doesn't match size of Y"}
        val yHat = mutableListOf<Int>()
        for (x in xTest) {
            yHat.add(predictLabel(x))
        }
        var correctPredictions = 0
        for ((y1, y2) in yHat.zip(yTest)) {
            if (y1 == y2) correctPredictions +=1
        }
        return correctPredictions.toDouble()/yTest.size
    }
```

This classifier should give you about 74% accuracy on the NLTK Twitter data. 
It may not be impressive, but it's extremely fast, robust, simple and gives you a decent baseline in no time! 
Check out the repo with the complete example on [GitHub](http://github.com/MKhalusova/kotlin-bayes). 

## Naive Bayes Limitations 
As you may have guessed, Naive Bayes does not take into account sentence structure. It assumes that all 
words are independent, which of course, often is not the case. 
Here's an example. 
* "I feel great even when I don't sleep well." - this somewhat positive statement after the preprocessing 
will turn into the following list of tokens [feel, great, even, sleep, well]

* And here's a somewhat negative sentence - "I don't feel great even when I sleep well.". It gets 
preprocessed into exactly the same list of tokens: [feel great even sleep well]

This algorithm won't see any difference between them. It will also struggle with sarcasm and euphemisms, but, 
to be fair, most algorithms will too. 
  
Congratulations on getting all the way to the end of this rather long article! I hope you enjoyed it and learned a thing or two :)
If you want to tinker with the code, you can find this example [here](http://github.com/MKhalusova/kotlin-bayes). 



 
