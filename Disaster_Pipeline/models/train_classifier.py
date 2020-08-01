import sys

import joblib
import nltk
import pandas as pd
from sqlalchemy import create_engine
import string

from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
STOPWORDS = nltk.corpus.stopwords.words("english")
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
PUNCTUATION_TABLE = str.maketrans('', '', string.punctuation)


def load_data(database_filepath):
    """
    Loads cleaned data from a sqlite database and splits it into X, y values
    :param database_filepath: File path for the sqlite database
    :return: Returns our data split into X, y values and a list of column names
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('clean_msg', engine)
    engine.dispose()

    X = df['message']
    Y = df.drop(['message', 'genre', 'id', 'original'], axis=1)
    cat_names = Y.columns.tolist()

    return X, Y, cat_names


def tokenize(text):
    """
    Takes text and transforms the words into their lemmatized version.

    :param text: Text to be transformed
    :return: Returns a list of lemmatized words with stop words removed
    """

    text = text.translate(PUNCTUATION_TABLE).lower()

    tokens = nltk.word_tokenize(text)

    lemmatized = [lemmatizer.lemmatize(word) for word in tokens if word not in STOPWORDS]

    return lemmatized


def build_model():
    """
    Builds our model pipeline and trains it.
    The model pipeline utilizes GridSearch to find a set of hyperparameters that maximize the model's performance
    :return: A trained model object
    """
    clf = RandomForestClassifier(n_estimators=100)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
        ('best', TruncatedSVD(n_components=100)),
        ('clf', MultiOutputClassifier(clf))
    ])

    param_grid = {
        'tfidf__ngram_range': ((1, 1), (1, 2)),
        'tfidf__max_df': [0.8, 1.0],
        'tfidf__max_features': [None, 10000],
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__min_samples_split': [2, 4]
    }

    cv = GridSearchCV(pipeline, param_grid, cv=3, verbose=10, n_jobs=-1)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates our model on metrics such as accuracy, precision, recall, and F1 score

    :param model: trained model object
    :param X_test: X-values from our test set to assess the model on
    :param Y_test: y-values from our test set to evaluate the model on
    :param category_names: List of column names for our "category" columns that will display with each set of metrics
    :return: Prints the metrics of our model
    """
    Y_pred = model.predict(X_test)

    print(classification_report(Y_test, Y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """
    Saves our model as a pickle file to be used by the web app.

    :param model: model object that will be saved
    :param model_filepath: File path to save our model's pickle file
    :return: Saves the model at the specified file path
    """
    joblib.dump(model, model_filepath)


def main():
    """
    Executes our model pipeline

    :return: prints success message if the model is successfully saved, otherwise will print
    an alternate message if not enough args are specified from the command line
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
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
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()