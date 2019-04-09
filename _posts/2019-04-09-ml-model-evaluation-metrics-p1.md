---
layout: post
title: "Machine Learning Model Evaluation Metrics part 1: Classification"
date: 2019-04-09
use_math: true
---

If you're only starting your machine learning journey, you may be taking online courses, reading books on the topic, 
dabbling with competitions and maybe even starting your own pet projects. If you do, you will inevitably start to notice 
that there's more than one way to evaluate a trained model. This is especially apparent in competitions - some companies post a 
classification challenge where results are evaluated with ROC/AUC score, others with F1 score, log loss, or some other 
metric. Regression problems, of course, have their own zoo of evaluation metrics. At some point, you gotta ask yourself - 
why are there so many? What do they mean? How are they different? And, if I start my own project, how do I approach 
choosing a metric for it? 
I know I did ask myself those questions, and ended up spending a significant amount of time figuring these things out. 
So to save others the time and frustration I compiled what I now know about evaluation metrics into a talk called "Machine 
Learning Model Evaluation Metrics" which I have recently delivered at AnacondaCON in Austin, Texas. If you prefer reading 
blog posts over watching talks, I intend to publish a couple of blog posts on the topic, and this is the first one. 

As a bonus, this format allows me to add more useful links :)  
In this first blog post, I'll focus on evaluation metrics for classification problems. And to make things a little simpler,
I'll limit myself to binary classification problems. Later, we'll talk how metrics can be extended to a multi-class problem.  

### Classification accuracy
Everybody who has ever build a classifier or read about building one has encountered classification accuracy. 
Classification accuracy (or simply, accuracy) is an evaluation metric that is a portion of correct predictions out of the total number of 
predictions produced by a ML model:

$$\text{Accuracy} = \frac{\text{Number of correct predictions}} {\text{Total number of predictions}}$$   

Accuracy can range from 0 to 1 (or 0 to 100%), and it is easy to intuitively understand. In scikit-learn, all 
estimators have a `score` method that gives you an evaluation metric for your model, and for classifiers it is accuracy. 

Here's an example where I train a basic `LogisticRegression` on a dataset for a binary problem. Once I did that, I can call 
the score method on my model, pass the test data and get the accuracy of my model on the test data.

![Classification Accuracy](/images/metrics/accuracy.png){:width="650px"}

I got almost 96%! This is amazing! Or is it? That's a trick question of course, because at this point we know too little 
and it's impossible to say whether 96% accuracy is "good" or not. In this case, it turns out to be not that great. 

To illustrate what I mean, I built a [DummyClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html) 
on the same data. A `DummyClassifier` in scikit-learn is a classifier that doesn't learn anything from the data but simply 
follows a given strategy. It can generate predictions by respecting the training set's class distribution, or uniformly 
at random, or, like in my case, it will always return the most frequent label in the training set. 

You can use `DummyClassifier` in the same fashion as any other scikit-learn estimator, and like other classifiers it has
a score method that will return its accuracy. So, drumroll please… 

![DummyClassifier](/images/metrics/dummy.png){:width="650px"}

The DummyClassifier has an accuracy of 94%!  This means my `LogisticRegression` model was only tiny bit better than 
predicting the most frequent label in the training set which doesn't seem so good anymore. Why is this happening? 

The reason is in the data, and I haven't shown you my data yet. For this example I used a synthetic dataset with 10 000 
samples, where only 5% of them represent samples of the positive class, while 95% of samples are of negative class:

![Make Classification](/images/metrics/make-classification.png){:width="650px"} 

So  even simply predicting negative class all the time we'll get 95% accuracy. If you're wondering why I got 94% and 
not 95% with the DummyClassifier, that would have to do with how the data got split into train and test sets. 

This type of dataset is what's called a **class-imbalanced dataset**, and unfortunately they are quite common. If you don't 
know that you are working with a class-imbalanced data, classification accuracy can give you a false impression that 
your model is doing great when in reality it is not. So first things first - be sure to check get to know your data!

Another thing to consider is that even if the data is balanced, and we'd like to improve the model, accuracy alone 
doesn't give enough information to diagnose the errors the model is making. Luckily, there are other evaluation metrics 
and diagnostic tools available for classification models.

### Confusion matrix
One of such tools is called Confusion Matrix. It is a table/matrix that shows how many samples a model classified 
correctly for what they are and how many were misclassified as something they are not. 

Confusion matrix is more of a diagnostic tool rather than a metric - it can help you gain insight into the type of errors are model is 
making. Plus, there's a number of evaluation metrics that can be derived from the confusion matrix. 
 
To illustrate Confusion Matrix, I took [Titanic dataset](https://www.kaggle.com/c/titanic) from kaggle where the goal is 
to predict survival of the passengers. Skipping all the data cleanup and feature engineering, I built a basic 
RandomForestClassifier. It has 83.4% accuracy on the test data (which would put me somewhere in the top 5% on the 
[Leaderboard](https://www.kaggle.com/c/titanic/leaderboard)), not bad! 

Let's see what a confusion matrix would look like in this case. First, I need to import confusion_matrix from 
sklearn.metrics. Now, when calling confusion_matrix it's important to remember that by convention, you need to provide 
the actual values first, then the predictions. The same convention is used for other metrics from sklearn.metrics. 

Once I call confusion_matrix, I get this beautiful array where the numbers indicate what has been correctly classified 
and what wasn't. If you haven't seen a confusion matrix before, it may be unclear what's what here.

To help understand it, confusion matrices are typically drawn as tables. I'll do exactly the same: 

[confusion matrix - > table]

Here the rows represent actual values, and the columns stand for predicted. It's important to note that this is a 
convention that is used in scikit-learn, and, for example, tensorflow, but there are tools that have it the other way 
around, so you may see tutorials and articles that have it the other way around (I'm looking at you, [Wikipedia](https://en.wikipedia.org/wiki/Confusion_matrix)!). 
As if we needed to justify the word "confusion" in "confusion matrix" by constantly confusing what goes on what axis, sigh.

We can learn a lot from the confusion matrix. On the diagonal, you'll find all the correct predictions: 
* True Positives (TP): a sample was positive, and the model predicted a positive class for it.
* True Negatives (TN): a sample was negative, and the model predicted a negative class for it.

We also have two types of errors: 
* False Positives (FP): a sample was negative, but the model predicted positive.
* False Negatives (FN): a sample was positive, but the model predicted negative.

If we sum up the correct predictions on the diagonal, and divide by the sum of all cells (all the predictions), we'll get 
classification accuracy. But, of course, we can get some other interesting metrics from the confusion matrix. 

### Precision, Recall, F1 Score
Suppose we're building a spam filter, where "spam" is a positive class, or a recommendation system where item being relevant 
is a positive class. In both cases, to make user experience better we really want to minimize False Positives. False 
Positive in a case of s spam filter would be an important email classified as a spam, and that would make users annoyed. 
False Positive in a recommendation system would mean recommending something irrelevant, which again, can be annoying, and 
will contribute to negative user experience. How can we evaluate which model will be "better" in this case? One metric 
that can help is called Precision.
Here's the formula: 

[FORMULA]

We take the number of true positives, and divide it by the sum of true positives and false positives.
If we happen to be so lucky that we don't have any false positives, we'll get a Precision of 1. And if we had some true 
positives and some false positives, and managed to reduce the number of false positives, this metric will get a little 
closer to 1 compared to what it was.
So. If you care about minimizing the number of false positives, keep an eye on Precision.

Now let's say we're working on predicting cancer diagnosis. In this case the cost of False Negatives is high. If the 
model predicts a patient doesn't have cancer (when in fact they do) and the patient is sent home, it can have much more 
serious implications compared to sending a healthy person for a few more tests. Or, if, say, we're classifying fraudulent transactions,
a false negative here can be quite costly. In cases like these, when we want to minimize the number of false negatives, 
the evaluation metric that can help is Recall. 
The formula is very similar to Precision, instead now we take into account the number of false negatives instead of false positives: 
[FORMULA]

The choice of Precision or Recall as an evaluation metric depends largely on the business problem your model is intended 
to solve. 

There's also an evaluation metric that takes into account both Precision and Recall, and presents another way of summarising 
a confusion matrix in one number. It's called F1 score, and it is the harmonic mean of Precision and Recall:

[FORMULA]

To get precision, recall and f1 score for a model, you can import these metrics from sklearn.metrics and call them in the 
same manner as confusion_matrix (actual values first, then predictions): 

[EXAMPLE]

Note: Evaluation metrics differ - some you want to maximize, others - to minimize. Scikit-learn has a helpful naming 
convention. If the name of a metric has the word "score" in it, it's a metric you want to maximize. If it has words 
"loss" or "error", that's a metric to minimize.

While it's difficult to compare confusion matrices, you can easily use Precision, Recall and F1 score to choose the best 
model, for instance, if you want to use GridSearchCV to find hyperparameters that will give you a model with the highest 
Recall, you can pass 'recall' as a 'scoring' parameter:

[screenshot]

### Matthews Correlation Coefficient
Accuracy and F1 Score are not the only ways to sum up a confusion matrix in a single number. There's also, for example, 
Matthews Correlation Coefficient:

[FORMULA]

One important thing to notice about this formula is that MCC takes into account all 4 cells of Confusion Matrix, unlike 
F1 Score. Indeed, F1 score is based on Precision and Recall, neither of which adds True Negatives to the equation. 
This has some interesting implications. Let's see how MCC is different from F1 Score with a couple of examples. 

 
 




 
