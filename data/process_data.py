import sys

import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loads and merges the data from both csv files containing messages and categories
    :param messages_filepath:  path to the messages file
    :param categories_filepath: path to the categories file
    :return:
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return messages.merge(categories, how='left', on='id')


def clean_data(df):
    """
    Cleans the categories and turns them into dummy variables
    :param df: dataFrame of the messages and their categories
    :return: cleaned DataFrame
    """
    columns = [w[:-2] for w in df['categories'][0].split(';')]
    categories = df['categories'].str.split(';', expand=True).transform(lambda x:[w[-1] for w in x])
    categories.columns = columns
    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df,categories], axis=1)
    return df


def save_data(df, database_filename):
    """
    Saves the DataFrame in a database
    :param df: DataFrame of the messages
    :param database_filename: path of the database file
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('messages', engine, index=False, if_exists='replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
