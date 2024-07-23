# Reddit Data Mining Project 📊

This project aims to extract actionable insights from Reddit posts in the "learnpython" subreddit using text mining and machine learning techniques. The project involves collecting data, preprocessing it, identifying actionable posts, extracting topics, and visualizing the results.

## Table of Contents 📚

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Methodology](#methodology)
   - [Data Collection](#data-collection)
   - [Text Preprocessing](#text-preprocessing)
   - [Classification](#classification)
   - [Topic Modeling](#topic-modeling)
   - [Visualization](#visualization)
5. [Results](#results)
6. [Contributing](#contributing)
7. [License](#license)

## Introduction 🎯

The purpose of this project is to analyze Reddit posts to identify actionable content, extract relevant topics, and visualize the distribution of these topics. This can help users to quickly find useful information and understand prevalent themes within the "learnpython" subreddit.

## Installation 🛠️

To run this project, you will need to install the following Python libraries:

```sh
pip install praw nltk gensim textblob wordcloud matplotlib
```

## Usage 🚀

1. Set up your Reddit API credentials:

   ```python
   reddit = praw.Reddit(
       client_id='YOUR_CLIENT_ID',
       client_secret='YOUR_SECRET_KEY',
       user_agent='MyAPI/0.0.1',
       username='YOUR_USERNAME',
       password='YOUR_PASSWORD'
   )
   ```

2. Run the main script:

   ```python
   python main.py
   ```

## Methodology 🧠

### Data Collection 📥

Posts are collected from the "learnpython" subreddit using the Reddit API with the PRAW library. The script fetches the top 1000 hot posts.

```python
subreddit = reddit.subreddit("learnpython")
hot_posts = subreddit.hot(limit=1000)
for post in hot_posts:
    post_list.append(post.selftext)
```

### Text Preprocessing 📝

The collected posts are preprocessed to remove noise and lemmatize the text.

```python
def remove_noise(post_list):
    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()

    def clean(doc):
        stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
        punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
        normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
        return normalized

    doc_clean = [clean(doc).split() for doc in post_list]
    return doc_clean
```

### Classification 🔍

A Naive Bayes classifier is used to identify actionable posts from the preprocessed text.

```python
training_corpus = [
    ("need help with for loop in Python", "actionable"),
    ("Using pandas to analyze data, need advice", "actionable"),
    # ... more training data ...
]

model = NBC(training_corpus)
for doc in doc_standard:
    result = model.classify(doc)
    if result == 'actionable':
        actionable_post.append(doc)
```

### Topic Modeling 📚

Latent Dirichlet Allocation (LDA) is used to extract topics from actionable posts.

```python
dictionary = corpora.Dictionary(actionable_list)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in actionable_list]
ldamodel = Lda(doc_term_matrix, num_topics=len(actionable_post), id2word=dictionary, passes=50)
raw_topic_list = ldamodel.print_topics(num_topics=len(actionable_post), num_words=1)
```

### Visualization 📊

The results are visualized using word clouds and bar charts.

```python
def vis_wordcloud(dict, topic_list):
    wordcloud = WordCloud(width=250, height=250).generate_from_frequencies(word_counts)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

def vis_bargraph(dict):
    plt.bar(topics, counts, color='skyblue', edgecolor='black')
    plt.xlabel("Topic")
    plt.ylabel("Number of Posts")
    plt.title("Distribution of Posts by Topic (Top 25)")
    plt.show()
```

## Results 📈

The project outputs actionable posts grouped by topic and visualizes the distribution of these topics through word clouds and bar charts.

# TEAM MEMBERS
## Aarya Ashok K
## Tejasv Goyal
