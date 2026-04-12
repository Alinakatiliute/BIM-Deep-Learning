# Deep Network for Text Classification
#
# Covers: embedding layers, padding, string tokenization, and pretrained
# sentiment libraries via an introductory example.
#
# Install dependencies (run once):
#   pip install vaderSentiment nrclex matplotlib pandas tensorflow

import re

import matplotlib.pyplot as plt
import pandas as pd
from nrclex import NRCLex
from tensorflow.keras import Sequential
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Dense, Embedding, Flatten, SimpleRNN
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def plot_history(history, title):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle(title)
    ax1.plot(history.history['loss'], label='train loss')
    ax1.plot(history.history['val_loss'], label='val loss')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax2.plot(history.history['acc'], label='train acc')
    ax2.plot(history.history['val_acc'], label='val acc')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    plt.tight_layout()
    plt.show()


# Sep 12, 2017 tweets after iPhone X release
t1 = (
    "I'm not a huge $AAPL fan but $160 stock closes down $0.60 for the day"
    " on huge volume isn't really bearish"
)
t2 = (
    "$AAPL $BAC not sure what more dissapointing: the new iphones or the"
    " presentation for the new iphones?"
)
t3 = "IMO, $AAPL animated emojis will be the death of $SNAP."
t4 = (
    "$AAPL get on board. It's going to 175. I think wall st will have issues"
    " as aapl pushes 1 trillion dollar valuation but 175 is in the cards"
)
t5 = (
    "In the AR vs. VR battle, $AAPL just put its chips behind AR in a big way."
)
tweets = [t1, t2, t3, t4, t5]

# --- Tokenize ---
# For the tokenization process, we specified num_words as 10.
# We want to use 10 of the most frequent words and ignore any others.
# Tokenizer automatically converts text into lowercase and removes punctuation.
tokenizer = Tokenizer(num_words=10)
tokenizer.fit_on_texts(tweets)

print("Word counts:", tokenizer.word_counts)
print("Index to word:", tokenizer.index_word)
print("Word to index:", tokenizer.word_index)

# These are most frequent words.
top3_by_index = {k: tokenizer.index_word[k]
                 for k in list(tokenizer.index_word)[:3]}
top3_by_word = {k: tokenizer.word_index[k]
                for k in list(tokenizer.word_index)[:3]}
print("Top 3 by index:", top3_by_index)
print("Top 3 by word:", top3_by_word)
print("Vocabulary size:", len(tokenizer.word_index))

# The following code is used to convert text into sequences of integers.
seq = tokenizer.texts_to_sequences(tweets)
print("Integer sequences:", seq)

# Padding.
# We have used pad_sequences so that all sequences are equal in length.
pad_seq = pad_sequences(seq, maxlen=5)
print("Padded sequences (pre, default):\n", pad_seq)

# Check and see differences.
pad_seq = pad_sequences(seq, maxlen=5, padding='post', truncating='post')
print("Padded sequences (post padding + truncating):\n", pad_seq)

# NRC Emotion Lexicon (NRCLex): a lookup table of ~14,000 words manually tagged
# by human annotators with 8 emotions (fear, anger, anticipation, trust,
# surprise, sadness, disgust, joy) plus positive/negative sentiment.
# It is NOT a predictive model — it simply looks up each word in the dictionary
# and returns the proportion of matched words belonging to each emotion category.
#
# R's syuzhet package uses the same NRC lexicon but lemmatizes words first
# (e.g. "pushes" -> "push"), catching more matches. Here we strip punctuation
# with regex as an approximation; results may differ slightly from R.

nrc = NRCLex()

for tweet in tweets:
    nrc.load_token_list(re.findall(r'\b[a-z]+\b', tweet.lower()))
    print(f"NRC emotions for '{tweet[:20]}...': {nrc.affect_frequencies}")

nrc.load_token_list(['anime'])
print(f"NRC emotions for 'anime': {nrc.affect_frequencies}")

nrc.load_token_list(['death'])
print(f"NRC emotions for 'death': {nrc.affect_frequencies}")

# Different sentiment lexicons via VADER (replaces syuzhet/bing/afinn).
analyzer = SentimentIntensityAnalyzer()
for tweet in tweets:
    score = analyzer.polarity_scores(tweet)
    print(f"VADER sentiment for '{tweet[:20]}...': {score}")


# --- IMDB Data ---

# 50K movie reviews (25K positive, 25K negative sentiment comments).
# Very balanced, very sanitary.
# Already converted to numbers, cleaned and so forth.

# The following are some examples of negative reviews from IMDb labeled as 0:
# A very, very, very slow-moving, aimless movie about a distressed, drifting
# young man.
# Not sure who was more lost-the flat characters or the audience, nearly half
# of whom walked out.
# Attempting artiness with black and white and clever camera angles, the movie

# The following are some examples of positive reviews from IMDb labeled as 1:
# The best scene in the movie was when Gerardo was trying to find a song that
# kept running through his head.
# Saw the movie today and thought it was a good effort, good messages for kids.
# Loved the casting of Jimmy Buffet as the science teacher.

# At this point take a look at IMDB data frame.
# Notice the length of the text in the lists.
(train_x, train_y), (test_x, test_y) = imdb.load_data(num_words=500)

# Padding
train_x = pad_sequences(train_x, maxlen=200, padding='post')
test_x = pad_sequences(test_x, maxlen=200, padding='post')

# --- Model ---

# Building a classification model.
model = Sequential([
    Embedding(input_dim=500, output_dim=16, input_length=200),
    Flatten(),
    Dense(units=16, activation='relu'),
    Dense(units=1, activation='sigmoid')
])

model.summary()

# Compile.
model.compile(optimizer='adamax',
              loss='binary_crossentropy',
              metrics=['acc'])

# Fit model.
# Don't forget to change batch size to 512 and try again.
history_1 = model.fit(train_x, train_y,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)

plot_history(history_1, 'Model 1 - Embedding + Dense')

# Prediction.
model.evaluate(train_x, train_y)
pred = model.predict(train_x).flatten()
print("\n--- Train Predictions ---")
print(pd.crosstab(pred, train_y, rownames=['Predicted'], colnames=['Actual']))

model.evaluate(test_x, test_y)
pred1 = model.predict(test_x).flatten()
print("\n--- Test Predictions ---")
print(pd.crosstab(pred1, test_y, rownames=['Predicted'], colnames=['Actual']))

# --- Possible options ---

# Playing with sequence length and optimizer: Adam, Adamax.
# I will do that in LSTMs, looks nicer there.
model2 = Sequential([
    Embedding(input_dim=500, output_dim=32),
    SimpleRNN(units=32),
    Dense(units=1, activation='sigmoid')
])
model2.summary()

# Compile model.
model2.compile(optimizer='rmsprop',
               loss='binary_crossentropy',
               metrics=['acc'])

history2 = model2.fit(train_x, train_y,
                      epochs=10,
                      batch_size=128,
                      validation_split=0.2)

plot_history(history2, 'Model 2 - Embedding + SimpleRNN')

model2.evaluate(train_x, train_y)
model2.evaluate(test_x, test_y)
