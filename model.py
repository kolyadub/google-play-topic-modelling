import numpy as np
import pandas as pd
from pprint import pprint

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

from google_play_scraper import Sort, reviews, app

import nltk

nltk.download("stopwords")
mallet_path = "mallet-2.0.8/bin/mallet"  # Folder path with Mallet

import emoji
import re
from nltk.corpus import stopwords
from pymystem3 import Mystem
import string
import json

num_topics = 5
number_of_reviews = 5

application_name = "ru.rostel"
scores = [1, 2]  # Review scores
count = 200  # The number of reviews
result = []
language = "ru"  # Review language
country = "ru"  # Review region

mystem = Mystem()
russian_stopwords = stopwords.words("russian")
russian_stopwords.append("это")
russian_stopwords.remove("не")

def get_gp_title(application_name):
    app_info = app(
    application_name,
    lang=language, # defaults to 'en'
    country=country # defaults to 'us'
    )
    app_title = app_info['title']
    return app_title

def get_data_gp():
    result = []
    for score in scores:
        try:
            result_current, continuation_token = reviews(
                application_name,
                lang=language,
                country=country,
                sort=Sort.NEWEST,  # defaults to Sort.MOST_RELEVANT
                count=int(count / len(scores)),  # defaults to 100
                filter_score_with=score,  # defaults to None(means all score)
            )
        except IndexError:
                result_current = [{'content' :'Empty content', 'at': datetime.datetime(2020, 6, 6, 13, 41, 46)}]
        result.extend(result_current)
    return result


def preprocess_text_rus(text):
    """
    Removing punctuation and lemmatization.
    """
    text = text.translate(
        str.maketrans(string.punctuation, " " * len(string.punctuation))
    )
    tokens = mystem.lemmatize(text.lower())
    tokens = [
        token.strip()
        for token in tokens
        if token not in russian_stopwords
        and token != " "
        and token not in emoji.UNICODE_EMOJI
    ]
    text = " ".join(tokens)
    # Verbs with не are more informative in the case of negative reviews
    text = text.replace("не ", "не_")

    return text


def df_from_preproc():
    result = get_data_gp()
    reviews_list = []
    date = []
    for i in range(len(result)):
        if result[i]["content"]:
            reviews_list.append(result[i]["content"])
            date.append(result[i]["at"])

    df = pd.DataFrame(data={"date": date, "text": reviews_list})

    df.date = pd.to_datetime(df.date).dt.normalize()

    df["text_preproc"] = df.text.apply(preprocess_text_rus)

    df = df[df["text_preproc"].apply(len) > 2].reset_index(drop=True)

    return df

def sent_to_words(sentences):
    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=False))


def keyword_review_merge(ldamodel, corpus, texts):
    # Init output
    topic_per_review_df = pd.DataFrame()

    # Find a topic for each review with the biggest percentage contribution
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        topic_num = row[0][0]
        prop_topic = row[0][1]
        # Get keywords for the topic
        topic_per_review_df = topic_per_review_df.append(
            pd.Series([int(topic_num), round(prop_topic, 4)]), ignore_index=True
        )
    topic_per_review_df.columns = ["dominant_topic", "perc_contribution"]

    # Add original text to the end of the output
    contents = pd.Series(texts)
    topic_per_review_df = pd.concat([topic_per_review_df, contents], axis=1)
    return topic_per_review_df


def topic_keywords():
    df = df_from_preproc()
    data_words = list(sent_to_words(df.text_preproc))
    id2word = corpora.Dictionary(data_words)

    corpus = [id2word.doc2bow(words) for words in data_words]

    best_model = gensim.models.wrappers.LdaMallet(
        mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word
    )

    data = df.text_preproc.values.tolist()

    review_with_topic_info_df = keyword_review_merge(
        ldamodel=best_model, corpus=corpus, texts=data
    )
    review_with_topic_info_df.columns = [
        "dominant_topic",
        "perc_contribution",
        "text_preproc",
    ]
    review_with_topic_info_df = review_with_topic_info_df.merge(
        df, right_index=True, left_index=True, how="left"
    ).drop(columns=["text_preproc_x", "text_preproc_y"])
    review_with_topic_info_df = review_with_topic_info_df.drop_duplicates(
        subset="text", keep="last"
    )
    output = {}
    topic_counts = review_with_topic_info_df["dominant_topic"].value_counts()

    for key, value in topic_counts.items():
        output[str(int(key))] = {"number_of_reviews": value}
        # output[str(int(key))]['reviews'] = review_with_topic_info_df[review_with_topic_info_df.dominant_topic == key].sort_values(
        #     'perc_contribution', ascending=False)[:number_of_reviews]['text'].to_list()
        texts = (
            review_with_topic_info_df[review_with_topic_info_df.dominant_topic == key]
            .sort_values("perc_contribution", ascending=False)[:number_of_reviews][
                "text"
            ]
            .to_list()
        )
        texts = [text.replace('"', "").replace("'", "") for text in texts]
        output[str(int(key))]["reviews"] = texts
        output[str(int(key))]["dates"] = (
            review_with_topic_info_df[review_with_topic_info_df.dominant_topic == key]
            .sort_values("perc_contribution", ascending=False)[:number_of_reviews][
                "date"
            ]
            .dt.strftime("%d-%m-%Y")
            .to_list()
        )
        words = [el[0].strip() for el in best_model.show_topic(int(key))]
        output[str(int(key))]["keywords"] = words

    # print(str(output))
    return output
