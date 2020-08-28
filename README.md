## Bayesian sentiment analysis

This is a natural language processing project with the goal to analyze Twitter users' sentiment, predicting solely through user's tweet content the sentiment of the target person. The data for this analysis and sentiment prediction comes the Gabriel dataset with 1.6 million labeled tweets. 

The data themselves are raw un-preprocessed Tweets and are therefore not suitable for producing "state of the arts" results with 99% accuracy. The challenge of this project is to implement a correctly working Naive Bayes for any kind of data and arbitrary many distinct labels, i.e. generic library for data mining.  

Since the algorithm is stochastic, the result may vary. Following settings were tested and used on processing data:
- Naive white space tokenizer (The model's performance can be much better with a sophisticated tokenizer like one of Lucene).
- Lower case text. So weird capitalization of users won't play a major role.
- 1-Gram model. From `"I like Tesla"` we will get `["I", "like", "Tesla"]`. A 2-Gram model will for example result the feature vector `["I_like", "like_Tesla"]`.
- Remove english stop words. Words like `"I", "that", "he", "she"` do not provide very much entropy and can be safely removed.
- 75% training data, 25% testing data. All data splits are balanced, that means the amount of positive tweets is the same as the amount of negative tweets. 
- Lemmatization with [NLP Stanford](https://nlp.stanford.edu/) yields a very long processing time but a promising result. Since the lemmatization only takes very much time in training time, the model's performance would still be fast in production.

Improvement suggestion:
- Using a better tokenizer like [Lucene Analyzer](https://www.baeldung.com/lucene-analyzers).
- Incorperate [Term frequency–inverse document frequency](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) to weight each token importance.

---

## Some shallow details 

##### Data overview

A typical tweet could look like following:

`
@switchfoot http://twitpic.com/2y1zl - Awww, that's a bummer.  You shoulda got David Carr of Third Day to do it. ;D
`

In this case, the tweet could be seen as very dirty. The username `@switchfoot`, the hyperlink to the image of the tweet and the grammar of the tweet makes a model very difficult to learn from the data. Splitting the tweet with a naive tokenizer will result the following feature vector

`
["@switchfoot", "http://twitpic.com/2y1zl", "-", "Awww,", "that's", "a", "bummer.", "You", "shoulda", "got", "David", "Carr", "of", "Third", "Day", "to", "do", "it.", ";D"]
`

##### Why Naive Bayes for classification?

Applying the Bayes Theorem on the data will result a pragmatic sentiment analysis model with a pretty high accuracy and very high explainability, which could be a deal breaker when using models like [Bert](https://github.com/google-research/bert), [Transformers](https://github.com/huggingface/transformers) or [GPT-3](https://en.wikipedia.org/wiki/GPT-3).

Naive Bayes on the other side may produce less accurate results, but the working principle of this model is very simple. You pump the data into its memory
and each unseen data will be classified by using prior knowledge (this is also the heart of the Bayes theorem). 

---

##### Why Kotlin?

This project was implemented with Kotlin, a language which combines the best from Python, JavaScript and Java, utilize the best features from each language and result the almost-perfect language. At the moment I am really having fun creating stuff in KotLin and hope for more dominance coming from this language.

---

## Result of the implementation

##### Naive Bayes on unprocessed data.
- Training: 0.84 Accuracy
- Testing: 0.76 Accuracy

##### Naive Bayes on all lower case. Terms like "Love" or "love" would therefore be the same.
- Training: 0.85 Accuracy
- Testing: 0.77 Accuracy

##### Word Lemma. Terms like "Love", "Like", "Hate", "Dislike" would therefore be the same.
- Training: 0.89 Accuracy
- Testing: 0.82 Accuracy

##### Random Forrest (3 trees). Combining wisdom of the crowd with Naive Bayes. The major vote out of three models will be the final prediction. The three models themselves were trained with different data.
- Training: 0.81 Accuracy
- Testing: 0.78 Accuracy

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

###### Log likelihood for better numerical handling

To avoid a numerical underflow from many multiplications, taking the logarithm each probability helps without changing the result of the 
maximum likelihood estimator.

The final formula would look like following. 

```
ln(P(negative | D)) 
= ln((P(negative) * product(P(Wj | negative))) / product(P(Wj)))
= ln((P(negative) * product(P(Wj | negative)))) - ln(product(P(Wj))))
= ln((P(negative) * product(P(Wj | negative)))) - ln(product(P(Wj))))
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

### References
- [Naive Bayes classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Probabilistic_model)
- [Naive Bayes Classifier for Text Classification](https://medium.com/analytics-vidhya/naive-bayes-classifier-for-text-classification-556fabaf252b)
