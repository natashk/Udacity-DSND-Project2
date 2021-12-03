import sys
import pandas as pd


def load_data(messages_filepath, categories_filepath):
    '''
    INPUT:
    messages_filepath - string, the filepath of the messages dataset
    categories_filepath - string, the filepath of the categories dataset

    OUTPUT:
    df - DataFrame, joined data from both datasets
    '''
    df_msg = pd.read_csv(messages_filepath)
    df_cat = pd.read_csv(categories_filepath)
    df = df_msg.join(df_cat.set_index('id'), on='id')

    if verbose:
        print(f'\n        messages shape: {df_msg.shape}\n        categories shape: {df_cat.shape}\n        joined shape: {df.shape}\n')

    return df


def clean_data(df):
    pass


def save_data(df, database_filename):
    pass  


def main():
    if len(sys.argv) >= 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:4]
        if len(sys.argv) == 5 and sys.argv[4] == 'verbose':
            global verbose
            verbose = True

        print('\nLoading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Missing arguments.\n\n'\
            'Usage: python process_data.py MSG_PATH CAT_PATH DATABASE\n\n'\
            '   MSG_PATH    the filepath of the messages dataset\n'
            '   CAT_PATH    the filepath of the categories dataset\n'
            '   DATABASE    the filepath of the database to save the cleaned data\n\n\n'
            'Example: python process_data.py '\
            'disaster_messages.csv disaster_categories.csv DisasterResponse.db')


verbose = False
if __name__ == '__main__':
    main()