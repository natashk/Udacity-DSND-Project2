import sys
import pandas as pd
import sqlite3


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
    #df = df_msg.join(df_cat.set_index('id'), on='id')
    #df = pd.concat([df_msg, df_cat], axis=1)
    df = df_msg.join(df_cat, how='outer', rsuffix='_cat')

    if debug:
        print(f'\n        messages shape: {df_msg.shape}\n        categories shape: {df_cat.shape}\n        joined shape: {df.shape}\n')
    return df


def clean_duplicates(df):
    print('    duplicates...')

    if debug:
        diff_ids = df[df['id']!=df['id_cat']].shape[0]
        print(f'\n{diff_ids} rows with different id and id_cat\n')
    # id == id_cat in all rows, so we can keep only id
    df = df.drop(columns=['id_cat'])
    
    if debug:
        print(f'New joined shape: {df.shape}\n')
        double = df[df.duplicated(keep=False)]
        print(f'Full duplicates:\n {double}\n')
        #double.to_csv('double.csv')
        #print(df_msg[df_msg['id']==24779])
        #print(df_cat[df_cat['id']==24779])

    # Drop full duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    if debug:
        print(f'\nNo full duplicates shape: {df.shape}\n')
    
    df_dupl_id = df[df.duplicated(subset=['id'], keep=False)]
    if debug:
        print('Duplicates in id:')
        print(df_dupl_id)
        print('Duplicates, the difference only in "categories":')
        print(df[df.duplicated(subset=['id','message','original','genre'], keep=False)])


    # Duplicates, the difference only in "categories"


    #if debug:
        #dupl_ids = list(df_dupl_id['id'])
        #print(f'\n{len(dupl_ids)}:  {dupl_ids}')
        #print(f'\n{len(list(set(dupl_ids)))}:  {dupl_ids}')

    return df

def clean_missing_values(df):
    print('    missing values...')
    if debug:
        print(df.isnull().sum())

    return df    

def create_dummies(df):
    print('    categorical variables (creating dummies)...')
    df = pd.concat([df, pd.get_dummies(df['genre'])], axis=1).drop(columns=['genre'])
    if debug:
        print(df.head())
 
    categories = df.loc[0,'categories'].replace('-0', '').replace('-1', '').split(';')
    if debug:
        print(f'\n{len(categories)} categories:\n{categories}\n')
    df_cat = df['categories'].str.split(';', expand=True)
    col_dict = {i:categories[i] for i in range(36)}
    if debug:
        print(f'\nDictionary of category columns:\n{col_dict}\n')
    df_cat = df_cat.rename(columns=col_dict)
    if debug:
        print(df_cat.head())

    for col in df_cat.columns:
        df_cat[col] = df_cat[col].str[-1]
    if debug:
        print(df_cat.head())
    df = pd.concat([df, df_cat], axis=1).drop(columns=['categories'])
    if debug:
        print(df.head())
    return df    

def clean_data(df):
    df = clean_duplicates(df)
    df = clean_missing_values(df)
    df = create_dummies(df)

    return df


def save_data(df, database_filename):
    conn = sqlite3.connect(database_filename)
    df.to_sql('messages', con=conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()


def main():
    if len(sys.argv) >= 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:4]
        if len(sys.argv) == 5 and sys.argv[4] == 'debug':
            global debug
            debug = True

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


debug = False
if __name__ == '__main__':
    main()
