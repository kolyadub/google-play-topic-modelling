import numpy as np
import pandas as pd
from pprint import pprint
import os

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

from google_play_scraper import Sort, reviews, app

ROOT_DIR = os.path.abspath(os.curdir)
print(ROOT_DIR)

os.environ.update({'MALLET_HOME':r''+ROOT_DIR+'/mallet-2.0.8/'})
mallet_path = ROOT_DIR + "/mallet-2.0.8/bin/mallet"  # Folder path with Mallet

eng_words = []
with open('mallet-2.0.8/eng_words.txt', 'r') as file:
    eng_words=file.read().splitlines() 
eng_words

import emoji
import re
# from nltk.corpus import stopwords
from pymystem3 import Mystem
import string
import json
import datetime

num_topics = 5
number_of_reviews = 5

application_name = "ru.rostel"
scores = [1, 2]  # Review scores
count = 200  # The number of reviews
result = []
language = "ru"  # Review language
country = "ru"  # Review region


russian_stopwords = ['c', 'а', 'алло', 'без', 'белый', 'близко', 'более', 'больше', 'большой', 'будем', 'будет', 'будете', 'будешь', 'будто', 'буду', 'будут', 'будь', 'бы', 'бывь', 'в', 'вам', 'вами', 'вас', 'ваш', 'ваша', 'ваше', 'ваши', 'вверх', 'вдали', 'вдруг', 'ведь', 'везде', 'весь', 'взять', 'вид', 'вместе', 'вне', 'во', 'вокруг', 'вон', 'вообще', 'восемнадцатый', 'восемнадцать', 'восемь', 'восьмой', 'вот', 'впрочем', 'времени', 'все', 'все', 'еще', 'всегда', 'всего', 'всем', 'всеми', 'всему', 'всех', 'всею', 'всю', 'всюду', 'вся', 'всё', 'второй', 'вы', 'выйти', 'г', 'где', 'главный', 'глаз', 'говорил', 'говорит', 'год', 'года', 'году', 'да', 'давать', 'давно', 'даже', 'далекий', 'далеко', 'дальше', 'даром', 'два', 'двадцатый', 'двадцать', 'две', 'двенадцатый', 'двенадцать', 'двух', 'девятнадцатый', 'девятнадцать', 'девятый', 'девять', 'действительно', 'дел', 'делал', 'делаю', 'дело', 'день', 'деньги', 'десятый', 'десять', 'для', 'до', 'довольно', 'долго', 'должен', 'должно', 'должный', 'другая', 'другие', 'других', 'друго', 'другое', 'другой', 'е', 'его', 'ее', 'ей', 'ему', 'если', 'есть', 'еще', 'ещё', 'ею', 'её', 'ж', 'же', 'жизнь', 'за', 'занят', 'занята', 'занято', 'заняты', 'затем', 'зато', 'зачем', 'здесь', 'знать', 'значит', 'и', 'иди', 'идти', 'из', 'или', 'им', 'имеет', 'имел', 'именно', 'ими', 'иногда', 'их', 'к', 'каждая', 'каждое', 'каждые', 'каждый', 'кажется', 'казаться', 'как', 'какая', 'какой', 'кем', 'книга', 'когда', 'кого', 'ком', 'кому', 'конечно', 'которая', 'которого', 'которой', 'которые', 'который', 'которых', 'кроме', 'кто', 'куда', 'лет', 'ли', 'лицо', 'лишь', 'лучше', 'м', 'маленький', 'мало', 'между', 'меля', 'менее', 'меньше', 'меня', 'мимо', 'мне', 'много', 'многочисленная', 'многочисленное', 'многочисленные', 'многочисленный', 'мной', 'мною', 'мог', 'могу', 'могут', 'мож', 'может', 'можно', 'можхо', 'мои', 'мой', 'мор', 'мочь', 'моя', 'моё', 'мы', 'на', 'наверху', 'над', 'надо', 'назад', 'наиболее', 'найти', 'наконец', 'нам', 'нами', 'нас', 'наш', 'наша', 'наше', 'наши', 'него', 'недавно', 'недалеко', 'нее', 'ней', 'некоторый', 'нельзя', 'нем', 'немного', 'нему', 'непрерывно', 'нередко', 'несколько', 'нет', 'нею', 'неё', 'ни', 'нибудь', 'ниже', 'низко', 'никакой', 'никогда', 'никто', 'никуда', 'ним', 'ними', 'них', 'ничего', 'ничто', 'но', 'новый', 'нога', 'ночь', 'ну', 'нужно', 'нужный', 'нх', 'о', 'об', 'оба', 'обычно', 'один', 'одиннадцатый', 'одиннадцать', 'однажды', 'однако', 'одного', 'одной', 'оказаться', 'окно', 'около', 'он', 'она', 'они', 'оно', 'опять', 'особенно', 'от', 'откуда', 'отовсюду', 'отсюда', 'очень', 'первый', 'перед', 'по', 'под', 'подойди', 'пожалуйста', 'позже', 'пока', 'пор', 'пора', 'после', 'последний', 'посреди', 'потом', 'потому', 'почему', 'почти', 'правда', 'прекрасно', 'при', 'про', 'просто', 'против', 'пятнадцатый', 'пятнадцать', 'пятый', 'пять', 'раз', 'разве', 'рано', 'раньше', 'ряд', 'рядом', 'с', 'с', 'кем', 'сам', 'сама', 'сами', 'самим', 'самими', 'самих', 'само', 'самого', 'самой', 'самом', 'самому', 'саму', 'самый', 'свет', 'свое', 'своего', 'своей', 'свои', 'своих', 'свой', 'свою', 'сделать', 'сеаой', 'себе', 'себя', 'сегодня', 'седьмой', 'сейчас', 'семнадцатый', 'семнадцать', 'семь', 'сих', 'сколько', 'слишком', 'слово', 'сначала', 'снова', 'со', 'собой', 'собою', 'совсем', 'спасибо', 'сразу', 'старый', 'т', 'та', 'так', 'такая', 'также', 'таки', 'такие', 'такое', 'такой', 'там', 'твои', 'твой', 'твоя', 'твоё', 'те', 'тебе', 'тебя', 'тем', 'теми', 'теперь', 'тех', 'то', 'тобой', 'тобою', 'тогда', 'того', 'тоже', 'только', 'том', 'тому', 'тот', 'тою', 'третий', 'три', 'тринадцатый', 'тринадцать', 'ту', 'туда', 'тут', 'ты', 'тысяч', 'у', 'уж', 'уже', 'хороший', 'хорошо', 'хотел', 'бы', 'хотеть', 'хоть', 'хотя', 'хочешь', 'час', 'часто', 'часть', 'чаще', 'чего', 'чем', 'чему', 'через', 'четвертый', 'четыре', 'четырнадцатый', 'четырнадцать', 'что', 'чтоб', 'чтобы', 'чуть', 'шестнадцатый', 'шестнадцать', 'шестой', 'шесть', 'эта', 'эти', 'этим', 'этими', 'этих', 'это', 'этого', 'этой', 'этом', 'этому', 'этот', 'эту', 'я']

def get_gp_title(application_name):
    """
    Getting application title from application code name
    """
    app_info = app(
    application_name,
    lang=language, # defaults to 'en'
    country=country # defaults to 'us'
    )
    app_title = app_info['title']
    return app_title

def get_data_gp():
    """
    Retreiving reviews and date from Google Play
    """
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
                result_current = [{'content' :'Empty content Пустой контент', 'at': datetime.datetime(2020, 6, 6, 13, 41, 46)}]
        result.extend(result_current)
    return result


def preprocess_text_rus(text, mystem):
    """
    Removing punctuation and lemmatization.
    """
    text = text.translate(
        str.maketrans(string.punctuation, " " * len(string.punctuation))
    )
    tokens = mystem.lemmatize(text.lower())
    # tokens = [word.lower() for word in text.split()]
    tokens = [
        token.strip()
        for token in tokens
        if token not in russian_stopwords
        and token != " "
        and token not in emoji.UNICODE_EMOJI
        and token not in eng_words
    ]
    text = " ".join(tokens)
    # Verbs with не are more informative in the case of negative reviews
    text = text.replace("не ", "не_")

    return text


def df_from_preproc():
    """
    Preprocessing data and creation of a Dataframe for LDA model
    """
    result = get_data_gp()
    reviews_list = []
    date = []
    for i in range(len(result)):
        if result[i]["content"]:
            reviews_list.append(result[i]["content"])
            date.append(result[i]["at"])

    df = pd.DataFrame(data={"date": date, "text": reviews_list})

    df.date = pd.to_datetime(df.date).dt.normalize()

    try:
        mystem = Mystem()
    except FileExistsError:
        print("Dierctory exists")

    df["text_preproc"] = df.text.apply(preprocess_text_rus, mystem=mystem)

    df = df[df["text_preproc"].apply(len) > 2].reset_index(drop=True)

    return df

def sent_to_words(sentences):
    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=False))


def keyword_review_merge(ldamodel, corpus, texts):
    """
    Merging reviews with corresponding keywords and contributions
    """
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
    """
    Main data pipeline for porcessing, getting topics, keywords and most relevant reviews
    """
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

    return output
