## Bayesian sentiment analysis

<img src="data/header.png" width=500px/>

This is a natural language processing project with the goal to employ Bayes Theorem to classify text documents. The focus of 
this machine learning model is speed, explainability and portability (implementable in every programming language without extra effort).

The algorithm was used primarily on text document. But the implementation was made generic and can be used for almost every kind
of training data.

Since the algorithm is stochastic, the result may vary. Following settings were tested:

- [Lucene](https://github.com/apache/lucene-solr)'s standard Tokenizer with Porter Stemming, removing stop words, removing HTML tags, lower case texts.
- 2-Gram model. From `"I like Tesla"` we will get `["I", "like", "Tesla", "I_like", "like_Tesla"]`. This step can sometimes improve the 
performance hugely since bayesian inference ignores completely that order of words, which however can very important for the 
linguistic understanding of a text document.
- Removing HTML tags with [JSoup](https://github.com/jhy/jsoup)
- 75% training data, 25% testing data. All data splits are balanced, the labels are equally distributed to avoid bias. 

Improvement suggestion:
- Incorperate [Term frequency–inverse document frequency](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) to weight each token importance.

---

## Why Naive Bayes for classification?

Applying the Bayes Theorem on the data will result a pragmatic sentiment analysis model with a pretty high accuracy and very high explainability, which could be a deal breaker when using models like [Bert](https://github.com/google-research/bert), [Transformers](https://github.com/huggingface/transformers) or [GPT-3](https://en.wikipedia.org/wiki/GPT-3).

Naive Bayes on the other side may produce less accurate results, but the working principle of this model is very simple. You pump the data into its memory
and each unseen data will be classified by using prior knowledge (this is also the heart of the Bayes theorem). 

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

## Some theory about Bayes theorem and how it can be applied and implemented on document classification

To apply Bayes' rule to problems, here is the general equation:

`
P(A|B) = (P(A) * P(B | A)) / P(B)
`

where `A` is a hypothesis and `B` is data. `P(A | B)` is a conditional probability, meaning the probability that hypothesis `A` is true 
given seen `B` data. 

##### Naive Bayes 

The Bayes theorem can be used in general form applied on text documents. Assuming a corpus of text documents can be used for training 
the model where each document `D` consists of words `Wj`. Each of the document will be labeled to belong to a class `Ci`. We 
are interested in classifying an unseen text document, i.e assigning it to a class of `C` using maximum a posteriori estimation.

`
classify(D) = argmax(P(Ci | D))
`

By definition, we can write

`
classify(D) = argmax((P(Ci) * P(D | Ci)) / P(D))
`

Because `D` consists of `Wj` we can write

`
classify(D) = argmax((P(Ci) * product(P(Wj | Ci))) / product(P(Wj)))
`

Consider that there are only two classes in the dataset, we can conclude 

```
P(negative | D) = (P(negative) * product(P(Wj | negative))) / product(P(Wj))

and 

P(positive | D) = (P(positive) * product(P(Wj | positive))) / product(P(Wj))
```

##### Log likelihood for better numerical handling

To avoid a numerical underflow from many multiplications, taking the logarithm each probability helps without changing the result of the 
maximum likelihood estimator.

The final formula would look like following. 

```
ln(P(negative | D)) 
= ln(P(negative)) + sum(ln(P(Wj | negative))) - sum(ln(P(Wj)))

and 

ln(P(positive | D)) 
= ln(P(positive)) + sum(ln(P(Wj | positive))) - sum(ln(P(Wj)))
```

##### Dealing with unknown words

One final note for implementation details, calculating `P(Wj | Ci)` would be a simple division of how often `Wj` appears in 
every document of `Ci` and how many words `Ci` has overall. Since the equation involves probabilities of each word of a new sentence with 
respect to a class, if a word from the new sentence does not occur in the class within the training set, the equation becomes zero. 
To solve this problem, we use Lidstone Smoothing or Laplace Smoothing by simple adding an additive parameter alpha. Instead of 

```
P(Wj | Ci) = (Wj count in Ci) / (Count of words in Ci)
```

we can do it better by 

```
P(Wj | Ci) = (Wj count in Ci + alpha) / (Count of words in Ci + alpha * length of D)
```

---

##### Why Kotlin?

This project was implemented with Kotlin, a language which combines the best from Python, JavaScript and Java, utilize the best features from each language and result the almost-perfect language. At the moment I am really having fun creating stuff in KotLin and hope for more dominance coming from this language.

---

### References
- [Naive Bayes classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Probabilistic_model)
- [Naive Bayes Classifier for Text Classification](https://medium.com/analytics-vidhya/naive-bayes-classifier-for-text-classification-556fabaf252b)
