# download necessary NLTK data
import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')

import sys
import sqlite3
import pandas as pd
import re
from nltk import pos_tag, ne_chunk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split


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

    category_names = list(df.columns[6:])
    X = df['message'].values
    Y = df[category_names]
    if debug:
        print(f'{len(category_names)} category names:\n{category_names}' )
        print(len(X))
        print(Y.head(3),'\n',Y.shape)
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


    # text processing and model pipeline


    # define parameters for GridSearchCV


    # create gridsearch object and return as final model pipeline

    """
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    """
    5. Test your model
    Report the f1 score, precision and recall for each output category of the dataset. You can do
    this by iterating through the columns and calling sklearn's classification_report on each.
    """
    pass


def save_model(model, model_filepath):
    """
    9. Export your model as a pickle file
    """
    pass


def main():
    if len(sys.argv) >= 3:
        database_filepath, model_filepath = sys.argv[1:3]
        if len(sys.argv) == 4 and sys.argv[3] == 'debug':
            global debug
            debug = True

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)

        # test out function
        for message in X[:5]:
            tokens = tokenize(message)
            print(message)
            print(tokens, '\n')

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        return
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)
        """
        6. Improve your model
        Use grid search to find better parameters.

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