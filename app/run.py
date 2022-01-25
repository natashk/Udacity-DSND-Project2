import sys
sys.path.append( '../models' )
from tokenization import tokenize

import json
import plotly
import pandas as pd

from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar
from plotly.graph_objs import Box
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    category_counts = list(df.iloc[:,-36:].sum(axis=0))
    category_names = list(df.columns[6:].str.replace('_',' '))
    
    genre_counts = list(df.iloc[:,3:6].sum(axis=0))
    genre_names = list(df.columns[3:6])

    direct_len = list(df[df['direct']==1].message.str.len())
    news_len = list(df[df['news']==1].message.str.len())
    social_len = list(df[df['social']==1].message.str.len())

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts,
                    texttemplate='%{y:.2s}',
                    text=category_counts,
                    textposition='outside'
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count",
                    'automargin': 1,
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': 60,
                    'categoryorder':'total descending'
                },
                'height': 500,
                'margin': {
                    't': 50,
                    'b': 150
                }

            }
        },
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    texttemplate='%{y:.2s}',
                    text=category_counts,
                    textposition='inside'
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                },
                'margin': {
                    't': 50,
                    'b':100
                }

            }
        },
        {
            'data': [
                Box(
                    name='direct',
                    x=direct_len,
                ),
                Box(
                    name='news',
                    x=news_len,
                ),
                Box(
                    name='social',
                    x=social_len,
                )
            ],
            'layout': {
                'title': 'Length of Message by Genre',
                'height': 700
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[6:], classification_labels))

    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001)
    #app.run()


if __name__ == '__main__':
    main()