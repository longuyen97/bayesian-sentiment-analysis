<h1 align="center">Bayesian sentiment analysis</h1>

The algorithm would be used primarily on text document. But the implementation was made generic and can be used for almost every kind
of training data. Since the algorithm is stochastic, the result may vary. Following processing steps were tested:

- [Lucene](https://github.com/apache/lucene-solr)'s standard Tokenizer with Porter Stemming, removing stop words, removing HTML tags, lower case texts.
- 2-Gram model. From `"I like Tesla"` we will get `["I", "like", "Tesla", "I_like", "like_Tesla"]`. This step can sometimes improve the 
performance hugely since bayesian inference ignores completely that order of words, which however can very important for the 
linguistic understanding of a text document.
- Removing HTML tags with [JSoup](https://github.com/jhy/jsoup)
- 75% training data, 25% testing data. All data splits are balanced, the labels are equally distributed to avoid bias. 

Improvement suggestion:
- Incorperate [Term frequency–inverse document frequency](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) to weight each token importance.

---


## Results of the models on multiple datasets

Following are some of my experiments on diverse datasets.

##### [Spamming Dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset)
- Training: 0.97 Accuracy
- Testing: 0.96 Accuracy

---

##### [Movie rating Dataset](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- Training: 0.89 Accuracy
- Testing: 0.85 Accuracy

---

##### [Twitter Dataset](https://www.kaggle.com/kazanova/sentiment140)
- Training: 0.91 Accuracy
- Testing: 0.80 Accuracy

---

### References
- [Naive Bayes classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Probabilistic_model)
- [Naive Bayes Classifier for Text Classification](https://medium.com/analytics-vidhya/naive-bayes-classifier-for-text-classification-556fabaf252b)
