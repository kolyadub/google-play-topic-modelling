# coding=utf8
from flask import Flask, render_template, render_template_string, request, Response, redirect, url_for
import json
import model
import os.path
import re
import requests
import redis
import time

app = Flask(__name__, static_url_path="/static/", static_folder='/Users/nd/Documents/GitHub/google-play-topic-modelling/static/')
req_form = {}

@app.route("/", methods=["GET", "POST"])
def index():
   return render_template("index.html")

@app.route("/loading", methods=["GET", "POST"])
def loading():
    data = request.form
    return render_template("loading.html", data=data)
    
@app.route("/report", methods=["GET", "POST"])
def report():
    app_link, app_title, count, scores = '', '', 0, []
    app_link = request.form['search']
    if '&' in app_link:
        app_link = app_link[app_link.index('=') + 1:app_link.index('&')]
    else:
        app_link = app_link[app_link.index('=') + 1:]
    model.application_name = app_link

    app_title = model.get_gp_title(app_link)

    count = int(request.form['count'])
    model.count = count

    if request.form['scores'] == 'Excellent':
        scores = [5]
    elif request.form['scores'] == 'Bad':
        scores = [1,2]
    else:
        scores = [3,4]
    model.scores = scores

    # model_data = model.topic_keywords()
    # data = {'model_data' : model_data,
    #         'app_title' : app_title,
    #         'count' : count,
    #         'scores' : scores}

    # with open('file2.txt', 'w+') as file:
    #     file.write(json.dumps(data))

    path = "app_list/"
    filename = "{0}_{1}_{2}".format(app_link, re.sub("[^0-9]", "", str(scores)), str(count)) + ".txt"
    print(filename)
    if os.path.isfile(path + filename):
        commands = ''
        with open(path + filename) as fh:
            for line in fh:
                commands += line
        data = json.loads(commands)
    else:
        model_data = model.topic_keywords()
        data = {'model_data' : model_data,
            'app_title' : app_title,
            'count' : count,
            'scores' : scores}
        with open(path + filename, 'w+') as file:
            file.write(json.dumps(data))


        # commands = ''
        # filename = 'file2.txt'
        # with open(filename) as fh:
        #     for line in fh:
        #         commands += line
        # data = json.loads(commands)

    return render_template("report.html", data=data)


# @app.errorhandler(Exception)
# def handle_error(e):
#     print(e)
#     # original = getattr(e, "google_play_scraper.exceptions.ExtraHTTPError", None)

#     # if original is None:
#     #     # direct 500 error, such as abort(500)
#     #     return render_template("error.html"), 500

#     # # wrapped unhandled error
#     # return render_template("error.html", e=original), 500

#     return render_template("error.html")

if __name__ == "__main__":
    app.run(debug=True) 
