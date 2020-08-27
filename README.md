# Bayesian sentiment analysis

This is a natural language processing project with the goal to analyze Twitter users' sentiment, predicting solely through user's tweet content the sentiment of the target person. The data for this analysis and sentiment prediction comes the Gabriel dataset with 1.6 million labeled tweets. 

The tweets themselves are un-preprocessed features and are therefore not suitable for producing "state of the arts" results with 99% accuracy. The challenge of this project is to implement a correctly working Naive Bayes for any kind of data and arbitrary many distinct labels, i.e. generic library for data mining.  

### Result

Since the algorithm is stochastic, the result may vary. Following settings are used:
- Naive white space tokenizer (The model's performance can be much better with a sophisticated tokenizer like one of Lucene)
- Lower case text
- 1-Gram model
- 75% training data, 25% testing data. All data splits are balanced, that means the amount of positive tweets is the same as the amount of negative tweets. 

Improvement suggestion:
- Using a better tokenizer like [Lucene Analyzer](https://www.baeldung.com/lucene-analyzers).
- Incorperate [Term frequency–inverse document frequency](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) to weight each token importance.

A typical tweet could look like following:

`
@switchfoot http://twitpic.com/2y1zl - Awww, that's a bummer.  You shoulda got David Carr of Third Day to do it. ;D
`

#### Naive Bayes on unprocessed data.
- Training: 0.84 Accuracy
- Testing: 0.76 Accuracy

#### Naive Bayes on all lower case. Terms like "Love" or "love" would therefore be the same.
- Training: 0.85 Accuracy
- Testing: 0.77 Accuracy

#### Word Lemma. Terms like "Love", "Like", "Hate", "Dislike" would therefore be the same.
- Training: 0.89 Accuracy
- Testing: 0.82 Accuracy

#### Random Forrest (3 trees). Combining wisdom of the crowd with Naive Bayes. The major vote out of three models will be the final prediction. The three models themselves were trained with different data.
- Training: 0.81 Accuracy
- Testing: 0.78 Accuracy


### Current state of the art

The current state of the art including neural network models like [Bert](https://github.com/google-research/bert), [Transformers](https://github.com/huggingface/transformers)
or [GPT-3](https://en.wikipedia.org/wiki/GPT-3) do a great work on NLP. However, those models lack expandability, where engineers (if at all) are rarely able to 
tell how a model comes to it decision. 

Naive Bayes on the other side may produce less accurate results, but the working principle of this model is very simple. You pump the data into its memory
and each unseen data will be classified by using prior knowledge (this is also the heart of the Bayes theorem).

### Bayes theorem

To apply Bayes' rule to problems, here is the general equation:

`
P(A|B) = (P(A) * P(B | A)) / P(B)
`

where `A` is a hypothesis and `B` is data. `P(A | B)` is a conditional probability, meaning the probability that hypothesis `A` is true 
given seen `B` data. 

#### Naive Bayes 

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
 

### Implementation

#### Dataset overview

An overview of the dataset. Only the first and fifth column will be used for training and classification.

```csv
"0","1467810369","Mon Apr 06 22:19:45 PDT 2009","NO_QUERY","_TheSpecialOne_","@switchfoot http://twitpic.com/2y1zl - Awww, that's a bummer.  You shoulda got David Carr of Third Day to do it. ;D"
"0","1467810672","Mon Apr 06 22:19:49 PDT 2009","NO_QUERY","scotthamilton","is upset that he can't update his Facebook by texting it... and might cry as a result  School today also. Blah!"
"0","1467810917","Mon Apr 06 22:19:53 PDT 2009","NO_QUERY","mattycus","@Kenichan I dived many times for the ball. Managed to save 50%  The rest go out of bounds"
"0","1467811184","Mon Apr 06 22:19:57 PDT 2009","NO_QUERY","ElleCTF","my whole body feels itchy and like its on fire "
"0","1467811193","Mon Apr 06 22:19:57 PDT 2009","NO_QUERY","Karoli","@nationwideclass no, it's not behaving at all. i'm mad. why am i here? because I can't see you all over there. "
"0","1467811372","Mon Apr 06 22:20:00 PDT 2009","NO_QUERY","joy_wolf","@Kwesidei not the whole crew "
"0","1467811592","Mon Apr 06 22:20:03 PDT 2009","NO_QUERY","mybirch","Need a hug "
"0","1467811594","Mon Apr 06 22:20:03 PDT 2009","NO_QUERY","coZZ","@LOLTrish hey  long time no see! Yes.. Rains a bit ,only a bit  LOL , I'm fine thanks , how's you ?"
"0","1467811795","Mon Apr 06 22:20:05 PDT 2009","NO_QUERY","2Hood4Hollywood","@Tatiana_K nope they didn't have it "
"0","1467812025","Mon Apr 06 22:20:09 PDT 2009","NO_QUERY","mimismo","@twittera que me muera ? "
```

### References
- [Naive Bayes classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Probabilistic_model)
- [Naive Bayes Classifier for Text Classification](https://medium.com/analytics-vidhya/naive-bayes-classifier-for-text-classification-556fabaf252b)
