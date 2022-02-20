import sys
import pandas as pd
import re

import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import re

from sklearn.pipeline import Pipeline, FeatureUnion
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
import pickle
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report


def load_data(database_filepath):
    """
    Loads data from a database
    :param database_filepath: path to the database file
    :return: training features, training targets, and the labels of the categories
    """
    engine = create_engine(f'sqlite:///{database_filepath}')

    df = pd.read_sql("SELECT * FROM messages", engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    return X.values, Y.values, list(Y.columns)


def tokenize(text):
    """
    Tokenizes a string
    :param text: the text
    :return: cleaned tokens
    """
    text = re.sub(r"[^a-zA-Z0-9]", ' ', text)
    tokens = [w for w in word_tokenize(text.lower()) if w not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Builds the workflow pipeline
    :return: model
    """
    pipeline = Pipeline([
        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
        ])),

        ('moc', MultiOutputClassifier(RandomForestClassifier()))
    ])
    # specify parameters for grid search
    parameters = {
        'text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        'text_pipeline__vect__max_features': (None, 5000, 10000),
        'text_pipeline__tfidf__use_idf': (True, False),
        'moc__estimator__criterion': ("gini", "entropy"),
        'moc__estimator__n_estimators': (50, 100, 150)
    }

    # create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates the model on the testing dataset
    :param model: the workflow pipeline
    :param X_test: The testing features
    :param Y_test: the testing targets
    :param category_names: teh categories labels
    """
    y_pred = model.predict(X_test)

    for i in range(len(category_names)):
        print(f"results for {category_names[i]}")
        print(classification_report(y_pred[:, i], Y_test[:, i]))


def save_model(model, model_filepath):
    """
    Saves the pre-trained model in a pickle file
    :param model: the workflow pipeline
    :param model_filepath: path to the model file
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
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