from tokenization import tokenize
import time
import sys
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle


def load_data(database_filepath):
    '''
    INPUT:
    database_filepath - string, the filepath of the database with data

    OUTPUT:
    X - array of messages 
    Y - array of arrays with values for categories
    category_names - list of category names
    '''

    # load data
    conn = sqlite3.connect(database_filepath)
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


def build_model():
    '''
    OUTPUT:
    pipeline - machine learning pipeline, which will take in the message column as input
               and output classification results
    '''

    # text processing and model pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # define parameters for GridSearchCV
    parameters = {
        'clf__estimator__n_estimators': [5, 10],
        'clf__estimator__min_samples_split': [3, 4],
        'clf__estimator__max_depth': [None, 25]
    }
    # create gridsearch object and return as final model pipeline
    pipeline = GridSearchCV(pipeline, param_grid=parameters)

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT:
    model - machine learning pipeline
    X_test - array of messages from test data
    Y_test - array of arrays with values for categories from test data
    category_names - list of category names

    OUTPUT:
    Reports:
        f1 score, precision and recall for each output category of the dataset
        accuracy of the model
        the best parameters found using GridSearch
    '''

    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names, zero_division=0))
    print(f'Accuracy: {(y_pred==Y_test).mean()}')
    print("\nBest Parameters:", model.best_params_)


def save_model(model, model_filepath):
    '''
    INPUT:
    model - machine learning pipeline
    model_filepath - the filepath of the pickle file to save the model to

    OUTPUT:
    Saves the model as a pickle file
    '''

    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def test_tokenize(X):
    '''
    INPUT:
    X - array of messages

    OUTPUT:
    Prints first 5 pairs of messages and tokens from them
    '''

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