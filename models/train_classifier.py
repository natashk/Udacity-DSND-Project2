import sys
import sqlite3
import pandas as pd


def load_data(database_filepath):
    conn = sqlite3.connect(database_filepath)

    # get a cursor
    cur = conn.cursor()

    # load data
    df = pd.read_sql("SELECT * FROM messages", con=conn)

    conn.commit()
    conn.close()

    if debug:
        print(f'\nShape: {df.shape}')
        print(df.info())
        print(df.head())

    X, Y, category_names = (None,None,None)
    return X, Y, category_names


def tokenize(text):
    pass


def build_model():
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) >= 3:
        database_filepath, model_filepath = sys.argv[1:3]
        if len(sys.argv) == 4 and sys.argv[3] == 'debug':
            global debug
            debug = True

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        return
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
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