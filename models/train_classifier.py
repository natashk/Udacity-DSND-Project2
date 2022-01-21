# download necessary NLTK data
import nltk
nltk.download(['punkt', 'wordnet'])
#nltk.download('averaged_perceptron_tagger')
#nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')

import time
import sys
import sqlite3
import pandas as pd
import re
#from nltk import pos_tag, ne_chunk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
#from sklearn.metrics import f1_score
#from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle




def load_data(database_filepath):
    conn = sqlite3.connect(database_filepath)

    # load data
    df = pd.read_sql("SELECT * FROM messages", con=conn)

    conn.commit()
    conn.close()

    if debug:
        print(f'\nShape: {df.shape}')
        print(df.info())
        print(df.head())

    category_names = list(df.columns[6:])
    X = df['message'].values
    Y = df[category_names].values
    if debug:
        print(f'{len(category_names)} category names:\n{category_names}' )
        print(len(X),'\n',len(Y))
        print(Y)
    return X, Y, category_names


def tokenize(text):
    # tokenize text
    #tokens = ne_chunk(pos_tag(word_tokenize(text)))
    stop_words = stopwords.words("english")

    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stop words and lemmatize
    tokens = [WordNetLemmatizer().lemmatize(token) for token in tokens if token not in stop_words]

    return tokens

def build_model():
    """
    3. Build a machine learning pipeline
    This machine pipeline should take in the message column as input and output classification
    results on the other 36 categories in the dataset. You may find the MultiOutputClassifier helpful
    for predicting multiple target variables.

    6. Improve your model
    Use grid search to find better parameters.


    # text processing and model pipeline


    # define parameters for GridSearchCV


    # create gridsearch object and return as final model pipeline

    """
    # text processing and model pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'clf__estimator__n_estimators': [5, 10],
        'clf__estimator__min_samples_split': [3, 4],
        'clf__estimator__max_depth': [None, 25]
    }
    pipeline = GridSearchCV(pipeline, param_grid=parameters)

    return pipeline



def evaluate_model(model, X_test, Y_test, category_names):
    """
    5. Test your model
    Report the f1 score, precision and recall for each output category of the dataset. You can do
    this by iterating through the columns and calling sklearn's classification_report on each.
    """
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names, zero_division=0))
    print(f'Accuracy: {(y_pred==Y_test).mean()}')
    print("\nBest Parameters:", model.best_params_)



def save_model(model, model_filepath):
    """
    9. Export your model as a pickle file
    """
    with open(model_filepath, 'wb') as file:  
        pickle.dump(model, file)

def test_tokenize(X):
    # test out function
    for message in X[:5]:
        tokens = tokenize(message)
        print(message)
        print(tokens, '\n')


def main():
    if len(sys.argv) >= 3:
        database_filepath, model_filepath = sys.argv[1:3]
        if len(sys.argv) == 4 and sys.argv[3] == 'debug':
            global debug
            debug = True

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)

        if debug:
            test_tokenize(X)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2) 
        print(X_train[:10])
        print(Y_train[:10])
        
        print('Building model...')
        start = time.time()
        model = build_model()
        end = time.time()
        if debug:
            print(f'{int((end - start)//60)} min {int((end - start)%60)} sec ({end - start} sec)')
        
        print('Training model...')
        start = time.time()
        model.fit(X_train, Y_train)
        end = time.time()
        if debug:
            print(f'{int((end - start)//60)} min {int((end - start)%60)} sec ({end - start} sec)')
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)
        """
        7. Test your model
        Show the accuracy, precision, and recall of the tuned model.

        Since this project focuses on code quality, process, and pipelines, there is no minimum
        performance metric needed to pass. However, make sure to fine tune your models for accuracy,
        precision and recall to make your project stand out - especially for your portfolio!

        8. Try improving your model further. Here are a few ideas:
            try other machine learning algorithms
            add other features besides the TF-IDF
        """

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Missing arguments.\n\n'\
            'Usage: python train_classifier.py DB_PATH MODEL_PATH \n\n'\
            '   DB_PATH      the filepath of the disaster messages database\n'
            '   MODEL_PATH   the filepath of the pickle file to save the model to\n\n\n'\
            'Example: python train_classifier.py '\
            '../data/DisasterResponse.db classifier.pkl')


debug = False
if __name__ == '__main__':
    main()