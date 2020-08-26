# Bayesian sentiment analysis

A NLP project on analyzing twitter users' sentiment, predicting solely through user's tweet content the sentiment of 
the target person.

The data for this analysis and sentiment prediction comes the Gabriel dataset with 1.6 million tweets .This dataset has target feature - sentiment as 0 & 4 (0 - Negative , 4 - Positive). 

The tweets themselves are un-preprocessed features and therefore not suitable for model feeding. The biggest challenge of this project is therefore implementing 
the data pipeline for processing tweets.  

### Motivation

Sentiment analysis is the automated process of analyzing text data and sorting it into sentiments positive, negative (or neutral). 

Using sentiment analysis tools to analyze opinions on Twitter data can help companies understand how people are talking about their brand.

The objective and challenges of sentiment analysis can be shown through some simple examples.

Simple cases: 
- Coronet has the best lines of all day cruisers.
- Bertram has a deep V hull and runs easily through seas.
- Pastel-colored 1980s day cruisers from Florida are ugly.
- I dislike old cabin cruisers.

More challenging examples
- I do not dislike cabin cruisers. (Negation handling).
- Disliking watercraft is not really my thing. (Negation, inverted word order).
- Sometimes I really hate RIBs. (Adverbial modifies the sentiment).
- I'd really truly love going out in this weather! (Possibly sarcastic).

### Basis

#### Current state of the art

The current state of the art including neural network models like [Bert](https://github.com/google-research/bert), [Transformers](https://github.com/huggingface/transformers)
or [GPT-3](https://en.wikipedia.org/wiki/GPT-3) do a great work on NLP. However, those models lack expandability, where engineers (if at all) are rarely able to 
tell how a model comes to it decision. 

Naive Bayes on the other side may produce less accurate results, but the working principle of this model is very simple. You pump the data into its memory
and each unseen data will be classified by using prior knowledge (this is also the heart of the Bayes theorem).

#### Bayes theorem

To apply Bayes' rule to problems, here is the general equation:

P(A|B) = (P(A) * P(B | A)) / P(B)

where A is a hypothesis and B is data. P(A | B) is a conditional probability, meaning the probability that hypothesis A is true 
given seen B data. 

#### Naive Bayes 

The Bayes theorem can be used in general form applied on text documents. Assuming a corpus of text documents can be used for training 
the model where each document `D` consists of words `Wi`. Each of the document will be labeled to belong to a class `Ci`. We 
are interested in classifying an unseen text document, i.e assigning it to a class of `C`.

`
classify(D) = argmax(P(Ci | D))
`

### Preprocessing

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