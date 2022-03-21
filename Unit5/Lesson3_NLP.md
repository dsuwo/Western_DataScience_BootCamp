<!---
Plan:

- Google N-Grams
- TF/IDF



--->

# Natural Language Processing (NLP), a.k.a. Code is Perfect and English is Awful

Numbers are highly structured and lend themselves well to caclulus. Human language, on the other hand, is just insane. For instance, consider the following two phrases:

> It's hot outside, yes?
> It's hot outside, no?

The two sentences mean the same thing, even though "yes" and "no" have completely opposite meanings. We, as humans, can understand the words in a broader context, but there is absolutely no way to tell a computer how this works. Code works according to perfect rules, English grammar takes rules as a challenge.

How, then, could we possibly hope to process natural language?


# The Basics: N-Grams

The most basic way to analyse text is by looking at a word and seeing what word (or words) are most likely to come after it. 

A unigram is a single word, a bigram is two words, and a trigram is three words. For two authors, it might be informative which author uses "democrat" or "republican" more, or who's more likely to end the phrase "climate change ..." with the word "hoax". On a finer scale, maybe you want to know whether "ei" is more common than "ie" in the speech that we actually use simply by how often the letter bigrams occur. This would be an imperfect metric, but it would be a good start!



This technique is very much just an exploratory analysis. There is no notion of a "population of all bigrams" that Homer sampled from in order to write The Illyad. We don't make assumptions about the sampling distribution of the word "aardvark". At best, we could look make a predictive keyboard that looks at the last word typed and tries to find the most common word to come after it. Even this last case isn't a prediction, it's just looking at what's in our data.



## Stop Words

Almost certainly, the most common bigram is going to be something like "and the". This is not very useful. To remedy this, we should remove words like "and" and "the" from our analysis before finding bigrams (or doing any other NLP task). The words "and" and "the" are considered **stop words**. 

A stop word is any word that gets filtered out prior to analysis. There are several standard stop word dictionaries that can be used for general purpose analyses, but the choice of which words are stop words is unique to each analysis. 

For example, if we had a corpus (a **corpus** is like a data set, but with text data) of interviews with data scientists about data science, we might want to remove "data science" from the data before we do data science on it. If, instead, the interviews were with medical professionals, their use of the phrase "data science" might be very informative!


# Comparative Exploration: TF-IDF

TF-IDF is short for Term Frequence - Inverse Document Frequency and it's a comparison of what words are more likely to be in one document than the others. 

Before we get too far, we should talk a bit more about what a corpus is. The full text of Homer's Illyad might be considered a single corpus. However, classical literature might also be a corpus, where each book is a separate set of words. Corpuses can either be *labelled*, such as by author, or *unlabelled*, such as a single work. The labels don't have to be unique - authors write more than one book, and 2nd or 3rd editions will be slightly different even though they have the same title! (Note that the word "corpuses" should not be confused with the word "porpoises" as that would severely confuse the dolphins). 

TF-IDF is only useful for labelled porpoises. In a labelled corpus, we can ask questions such as "what words or phrases (n-grams) make this document unique from the others?" 

Basically, we're looking at the number of times a word occurs in 1 document versus how many times it occurs in all of our documents. This can be very useful for:

- Finding which words characterize an author
- Determining which document discusses a given topic
- Determining how important a document is for search results



# Getting Sentimental


Sentiment analysis is used to determine the emotionality of language. Generally, we can assume that words like "good" or "happy" are positive words. If we somehow decide on a scale for positivity, such as "happy" = 1, "good" = 0.7, etc., then we can simply add up all the sentiments from a document and be done with it.

Some people have already done this, such as the [NRC Word-Emotion Association Lexicon](http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm), aka EmoLex. My favourite example of this use-case is [this analysis, showing that Trump used an Android while his aides use an iPhone](http://varianceexplained.org/r/trump-tweets/) (sorry, it's in R, not Python).

However, this isn't always useful for one, assigning sentiment to words will lead us to believe that "not bad" is bad, since we'll miss all of the negations. Again, English is a not entirely un-wild language and sometimes it's not not nonesense (if your first thought was to just look for "not"s and alter them, you'd miss both of those double negatives).

Instead, we can learn the way certain words are used in a given context. As an example, [this Kaggle](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data) competition has movie reviews and their score from 0-4 (1-5, but zero-indexed). The point of this is to learn what words *and* sentence structure corresponds to a bad review. This is done in a completely non-linear fashion, usually using dimensionalty reduction and/or neural networks. 



# Words as Numbers?!?

So let's say "and" is 1, "the" is 2, "it" is 3,... This is absolutely not useful in any way.

We need to do it smarter. In fact, we've already shown one way to represent words as numbers - sentiment analysis! Each word is assigned a value based on the strength and direction of its sentiment. So we can say that "love" and "cuddling" are close to each other, according to sentiment.

What if we add another dimension, say, action-ness. Now, the word "Love" has two directions (i.e, it has a **vector** assigned to it): Sentiment and action-ness. "Love" does not describe an action, but "cuddling" does, so in this sense they are not close. For a more general idea of closeness, we can calculate the distance between the two vectors.

```py
import numpy as np
import matplotlib.pyplot as plt

Love = np.array([2, 0.3])
Cuddle = np.array([1.7, 1.8])

plt.plot()
plt.annotate("", xy = Love, xytext = [0, 0], 
    arrowprops = dict(headwidth=10, width=1, 
    color = "red", headlength=10))
plt.annotate("", xy = Cuddle, xytext = [0, 0], 
    arrowprops = dict(headwidth=10, width=1, 
    color = "blue", headlength=10))
plt.annotate("", xy = Cuddle, xytext = Love, 
    arrowprops = dict(headwidth=1, width=1, 
    color = "black", headlength=1))
plt.plot(Love[0], Love[1], "ro")
plt.plot(Cuddle[0], Cuddle[1], "bo")
plt.text(Love[0], Love[1], "Love", size = 15)
plt.text(Cuddle[0], Cuddle[1], "Cuddle", size = 15)
plt.xlim([0,2.25]); plt.ylim([0,2.1])
plt.xlabel("Sentiment"); plt.ylabel("Action-ness")
plt.title("Similarity in two dimensions")
plt.show()
```

<img src="figs/vecsim.png">

In the plot above, the length of the black line can be used to determine the similarity of the words.

Just like with sentiment analysis, we run into the problem of labelling every single one of the words in the English language. What do we do when we're too lazy to assign labels ourself? We turn to neural networks!



# Word2Vec

Instead of manually labelling words, we just look at how close two words tend to be within sentences. This "closeness" represents the similarity of their use. This method *cannot* tell us whether two words have similar meanings (like the semantics or the action-ness in the previous examples), but it can tell us which words are related to other words.

Word2Vec is a two layer neural net first converts a word to a vector representation, then uses that to determine similar words. Unlike the neural nets from the previous lesson, it does *not* work based on the properties of a word itself. It determines the vector representation based on context, and this process of representing a word as a vector of numbers is called **embedding**. There are two ways it does this.

## 1. Continuous Bag of Words (CBOW)

Consider the sentence:

> The quick brown fox jumps over the lazy dog.

A CBOW would see the word "the" with nothing before it, and just move on. "Quick" has one word in front of it, so our data looks like this so far:

| Input 1 | Input 2 | Target |
|---|---|---|
| | the | quick |

When the CBOW moves on to "brown", the input looks like:

| Input 1 | Input 2 | Target |
|---|---|---|
| | the | quick |
| the | quick | brown| 

For "fox" and "jumps", we get:

| Input 1 | Input 2 | Target |
|---|---|---|
| | the | quick |
| the | quick | brown| 
| quick | brown | fox |
| brown | fox | jumps |


There's no restriction that the Target word needs to go at the *end* of a window. The following data would work just as well:

| Input 1 | Input 2 | Target |
|----|---|---|
| | quick | The |
| The | brown | quick |
| quick | fox | brown |
| brown | jumps | fox |
| fox | over | jumps |

In this data, the two inputs are the words before and after the target word.



## 2. Skipgrams


A skipgram would look at "the" and see that ["quick", "brown"] are nearby words, then add that to the data. Next, it would see that ["the", "brown", "fox"] are nearby to "quick". Next, ["the", "quick", "fox", "jumps"] are near "brown". Then, ["quick", "brown", "jumps", "over"] are near "fox". And so on. The resulting data will look like this:

| Source | Target |
|---|---|
| the | quick |
| the | brown |
| quick | the |
| quick | brown|
| quick | fox|
| brown | the |
| brown | quick |
| brown | fox |
| brown | jumps |
| fox | quick |
| fox | brown |
| fox | jumps |
| fox | over |
| ... | ... |

## Training the Model

In both CBOW and SkipGrams we have a collection of input words and target words. So how do we get these vectors that I promised you? At first, they're assigned randomly. We choose the number of directions (i.e. the **size of the embedding**), but otherwise they start out "dumb". 

For CBOW, the input words each have their own vectors associated with them. To predict the similarity to the target word, the vectors are added together and the resulting vector is compared to all other words. Whichever word has the closest vector to the sum of the input words is the predicted word. 

Backpropagation tries to make it so that the labelled target word is the best prediction, and in doing so it assigns the weights in the neural net. Having done this, you can put in any collection of input words (of the same size as your training data) and get a word prediction! This is actually how your phone's keyboard predicts the next word.

Note that, for CBOW, the collection of input words is a collection. All of those words, taken together, will try to predict the target.

Skipgrams work similarly, but also kinda backwards. If you train a skipgram using 4 input words per target, the data preparation step splits this into 4 rows, and each input is associated with a target. The neural net tries to make it so that each input correctly predicts the target, but there's a catch. Each target is tied to several *separate* inputs! The prediction takes this into account - the target should be predicted by each of the inputs individually, not taken together like CBOW. Because of this, Skipgrams identifies words that share a *context*. 


# Negative Samples

If I told you that many rich people gets up before 6am, would you guess that getting up before 6am causes richness? Well no, there are hundreds of millions of people who get up before 6am and are not rich. To make good conclusions about positive cases, we must know about the negative cases!

For the data sets above, add another column that just contains 1s (because they're all positive cases). For every target word, we add some number of random input values and put a 0 in the new column. The problem now reduces to predicting whether a given input is related to the output - this is just logistic regression!

# Examples

The following example of word2vec using skipgrams was taken from here: [cambridgespark.com](https://blog.cambridgespark.com/tutorial-build-your-own-embedding-and-use-it-in-a-neural-network-e9cde4a81296)

We begin by importing the `nltk` (Natural Language ToolKit) module and the `Word2Vec()` function from `gensim`. You may need to install these packages first (I'm not sure if they're pre-installed by Anaconda). `multiprocessing` allows us to use multiple cores on our computer to do this calculations.

```py
import nltk
from gensim.models import Word2Vec
import multiprocessing
```

Next, we can download some text. The `brown` corpus contains 1 million words and was [first compiled at Brown university](https://www.nltk.org/book/ch02.html). This is downloaded by the `nltk.download()` function, which opens up a text-based interface. Typing `d brown` will download the data, then `q` will exit.

```py
nltk.download()
```

Now that it's downloaded, we can import it:

```py
from nltk.corpus import brown
# Create an object with just the sentences
sentences = brown.sents()
```

Now we define the embedding size (the number of vectors used to represent a word). It's pretty standard to use 300, as this is a balance between the insane ways that language can be used and the amount of processing power that NLP needs.

We also define the window. In the examples above, the window for skipgrams was 1 since we looked at 1 word before and 1 word after. 

```py
EMB_DIM = 300

w2v = Word2Vec(sentences,
    vector_size = EMB_DIM,
    sg = 0, # 1 for skipgrams, 0 for CBOW
    window = 5,
    min_count = 5, # If a word is used <5 times, ignore it
    negative = 15, # Number of negative samples
    workers = 3 # Number of CPU cores to use
)
```

Go get a coffee, this will be a few minutes.

**Coffee Break**

And we're back!

Since this is a neural network, we don't exactly have model parameters that we can check. In fact, there's not a lot we *can* check! Instead, we'll just make sure the predictions seem reasonable.

```py
w2v.wv.similar_by_word("Saturday")
```

```
[('Monday', 0.9538821578025818),
 ('Sunday', 0.9456729888916016),
 ('Friday', 0.932210385799408),
 ('Tuesday', 0.9234191179275513),
 ('fourth', 0.9217803478240967),
 ('Wednesday', 0.9129100441932678),
 ('ending', 0.909375786781311),
 ('winter', 0.9027897715568542),
 ('December', 0.898737907409668),
 ('afternoon', 0.8976615071296692)]
```

Yep, that seems reasonable!

```py
w2v.wv.similar_by_word("money")
```

```
[('work', 0.8885102868080139),
 ('job', 0.8835934996604919),
 ('care', 0.8616563081741333),
 ('fear', 0.8477988243103027),
 ('trouble', 0.8448494076728821),
 ('way', 0.8342838883399963),
 ('chance', 0.8299768567085266),
 ('freedom', 0.8280826210975647),
 ('future', 0.824268639087677),
 ('others', 0.8190582990646362)]
```

Also reasonable!

```py
w2v.wv.similar_by_word("Canadian")
```

```
[('bold', 0.9846794009208679),
 ('theaters', 0.9838402271270752),
 ('Convention', 0.9834485650062561),
 ('recognizing', 0.9824867844581604),
 ('cerebral', 0.9820525646209717),
 ('stretches', 0.9806843996047974),
 ("Ruth's", 0.9806720018386841),
 ('contrasting', 0.9801533222198486),
 ('Barnett', 0.9800900816917419),
 ('projected', 0.9798555970191956)]
```

That's... not as reasonable. Try it with "Canada", and you get better results. Keep in mind that the corpus was compiled in the 60s and 1 million words isn't all that many. 

To play around with this, I've added it as a [Colab Notebook](https://colab.research.google.com/drive/17lLxZhffCX_gsCroG5hfK_rs4r6NtGmS?usp=sharing)
