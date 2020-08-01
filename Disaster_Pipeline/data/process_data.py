import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    This function loads a data set using a specified file path.

    :param messages_filepath: File path for the messages data set
    :param categories_filepath: File path for the categories data set
    :return: Returns the merged data set
    """
    # Load the messages data set
    messages = pd.read_csv(messages_filepath)

    # Load the categories data set
    categories = pd.read_csv(categories_filepath)

    return pd.merge(messages, categories, on='id')


def clean_data(df):
    """
    This function takes a DataFrame object and performs various cleaning steps on it, such as:
        1. extracting and mapping column names
        2. fixing the values of the "categories" columns to be one-hot encoded values
        3. dropping nulls and removing duplicates
    :param df: DataFrame object to be cleaned
    :return: a cleaned DataFrame ready to be input into a model pipeline
    """

    # Splitting categories column into separate columns in a separate DataFrame
    categories = df['categories'].str.split(";", expand=True)
    row = categories[:1]

    # Extracting the column names for the categories
    category_columns = row.applymap(lambda x: x[:-2]).iloc[0, :].tolist()
    categories.columns = category_columns

    # Fixing the values of categories to be either 0 or 1
    categories = categories.applymap(lambda x: int(x[-1]))

    # Dropping the original "categories" column and concatenating the new category columns
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)

    # Removing duplicate values, dropping nulls, and replacing any value that is not 0 or 1
    df.drop_duplicates(subset='message', inplace=True)
    df.dropna(subset=category_columns, inplace=True)
    df.related.replace(2, 0, inplace=True)
    
    return df


def save_data(df, database_filename):
    """
    This function saves the newly cleaned DataFrame into a sqlite database for extracting for our model
    :param df: DataFrame object to be saved
    :param database_filename: Filename for our sqlite database
    """
   
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('clean_msg', engine, index=False, if_exists='replace')
    engine.dispose()


def main():
    """
    This function runs through the steps of our data cleaning process and saves the cleaned DataFrame to
    a sqlite database
    """
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