# Bayesian sentiment analysis

A NLP project on analyzing twitter users' sentiment, predicting solely through user's tweet content the sentiment of 
the target person.

The de.longuyen.main data for this analysis and sentiment prediction is based on Gabriel dataset

Kazanova dataset with 1.6 million tweets will be used for modeling.This dataset has target feature - sentiment as 0 & 4 (0 - Negative , 4 - Positive) which will be relabeled as (0 & 1)

![](data/birds.gif)

### Motivation

Sentiment analysis is the automated process of analyzing text data and sorting it into sentiments positive, negative, or neutral. 

Using sentiment analysis tools to analyze opinions on Twitter data can help companies understand how people are talking about their brand.

### Dataset

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

### Basis

###### Sentiment analysis overview

Sentiment analysis (also known as opinion mining or emotion AI) refers to the use of natural language processing, text analysis, computational linguistics, 
and biometrics to systematically identify, extract, quantify, and study affective states and subjective information. Sentiment analysis is widely applied 
to voice of the customer materials such as reviews and survey responses, online and social media, and healthcare materials for applications that range 
from marketing to customer service to clinical medicine.

The objective and challenges of sentiment analysis can be shown through some simple examples.

Simple cases: 
- Coronet has the best lines of all day cruisers.
- Bertram has a deep V hull and runs easily through seas.
- Pastel-colored 1980s day cruisers from Florida are ugly.
- I dislike old cabin cruisers

More challenging examples
- I do not dislike cabin cruisers. (Negation handling)
- Disliking watercraft is not really my thing. (Negation, inverted word order)
- Sometimes I really hate RIBs. (Adverbial modifies the sentiment)
- I'd really truly love going out in this weather! (Possibly sarcastic)
- Chris Craft is better looking than Limestone. (Two brand names, identifying the target of attitude is difficult).
- Chris Craft is better looking than Limestone, but Limestone projects seaworthiness and reliability. (Two attitudes, two brand names).
- The movie is surprising with plenty of unsettling plot twists. (Negative term used in a positive sense in certain domains).
- You should see their decadent dessert menu. (Attitudinal term has shifted polarity recently in certain domains)
- I love my mobile but would not recommend it to any of my colleagues. (Qualified positive sentiment, difficult to categorise)
- Next week's gig will be right koide9! ("Quoi de neuf?" Fr.: "what's new?". Newly minted terms can be highly attitudinal but volatile in polarity and often out of known vocabulary.)

###### Bayesian statistics

###### Naives Bayes for text classification

###### NLP

###### NLP techniques