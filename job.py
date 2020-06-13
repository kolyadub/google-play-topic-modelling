import numpy as np
import pandas as pd
from pprint import pprint
import itertools
import datetime

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

from google_play_scraper import Sort, reviews, app

import nltk

nltk.download("stopwords")

import emoji
import re
from nltk.corpus import stopwords
from pymystem3 import Mystem
import string
import json

mallet_path = "mallet-2.0.8/bin/mallet" 

num_topics = 5
number_of_reviews = 5

application_name = "ru.rostel"
scores = [1, 2]  # Review scores
count = 200  # The number of reviews
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
                count=count_per_score,  # defaults to 100
                filter_score_with=score,  # defaults to None(means all score)
            )
        except IndexError:
                result_current = [{'content' :'Empty content', 'at': datetime.datetime(2020, 6, 6, 13, 41, 46)}]
        result.extend(result_current)
    print(scores, count, count_per_score, filename, "NUMBER: ", len(result))
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

def compute_coherence_values(id2word, corpus, texts, maximum, start=2, step=3):
    """
    Compute coherences for different topic number variations to find the best one

    Returns:
    model_list - list of models for the given topic number range
    coh_values - values of coherences for these models
    """
    coh_values = []
    model_list = []
    for num_topics in range(start, maximum, step):
        model = gensim.models.wrappers.LdaMallet(
            mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model_list.append(model)
        coherencemodel = CoherenceModel(
            model=model, texts=texts, dictionary=id2word, coherence='c_v')
        coh_values.append(coherencemodel.get_coherence())

    return model_list, coh_values
    
def topic_keywords():
    df = df_from_preproc()
    data_words = list(sent_to_words(df.text_preproc))
    id2word = corpora.Dictionary(data_words)

    corpus = [id2word.doc2bow(words) for words in data_words]

    start = 3
    maximum = 7
    step = 2

    model_list, coh_values = compute_coherence_values(
        id2word=id2word, corpus=corpus, texts=data_words, start=start, maximum=maximum, step=step)

    best_model_index = coh_values.index(max(coh_values))
    best_model = model_list[best_model_index]

    # best_model = gensim.models.wrappers.LdaMallet(
    #     mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word
    # )

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

if __name__ == "__main__":

    path = "app_list/"
    file_list = ['top_free.txt','top_paid.txt']
    app_ids = []
    for file in file_list:
        with open(path + file) as fh:
            for line in fh:
                app_ids.extend(line.split(sep=','))
    
    scores_to_permute = [[1,2],[3,4],[5]]
    counts_to_permute = [100, 200, 500]
    # permutations = [[app_ids[4]], [[1,2],[5]], [100, 200]]
    permutations = [app_ids[26:], scores_to_permute, counts_to_permute]
    # permutations = [['ru.yandex.yandexnavi'], [[5]], [200]]
    for perm in list(itertools.product(*permutations)):
        application_name = perm[0]
        scores = perm[1] 
        count = perm[2]
        count_per_score = int(count/len(scores))
        app_title = get_gp_title(application_name)
        filename = "{0}_{1}_{2}".format(perm[0], re.sub("[^0-9]", "", str(perm[1])), str(perm[2]))+".txt"

        model_data = topic_keywords()

        data = {'model_data' : model_data,
            'app_title' : app_title,
            'count' : count,
            'scores' : scores}
        
        with open('app_list/' + filename, 'w+') as file:
            file.write(json.dumps(data))