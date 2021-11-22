import os
import warnings
import sys

import pandas as pd
import numpy as np
from joblib import dump, load

from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, f1_score

from preproccessing import *

import logging


mlflow.set_tracking_uri("http://localhost:5000")


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    f1 = f1_score(actual, pred)
    acc = accuracy_score(actual, pred)
    return f1, acc


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # Read data
    try:
        df = pd.read_csv('IMDB_dataset.csv')
        df['target'] = df['sentiment'].apply(
            lambda x: 1 if x == 'positive' else 0)
    except Exception as e:
        logger.exception(
            "Unable to get training & test CSV, check that data is in the current folder. Error: %s",
            e)

    print('Data loaded')
    df = preproccess(df, 'review')

    print('data preproccessed')

    train_x, test_x, train_y, test_y = train_test_split(df['final_text'],
                                                        df['target'],
                                                        test_size=0.2,
                                                        stratify=df['target'])

    C = float(sys.argv[1]) if len(sys.argv) > 1 else 1

    print('start training model... ')
    with mlflow.start_run():

        tf_idf = TfidfVectorizer(ngram_range=(1, 2))
        tf_idf.fit(train_x)

        train_x = tf_idf.transform(train_x)
        test_x = tf_idf.transform(test_x)

        model = LinearSVC(C=C).fit(train_x, train_y)

        predicted_qualities = model.predict(test_x)

        (f1, acc) = eval_metrics(test_y, predicted_qualities)

        print("LinearSVC model (C=%f):" % (C))
        print("f1_score: %s" % f1)
        print("accuracy_score: %s" % acc)

        mlflow.log_param("C", C)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy_score", acc)

        dump(model, '../inference/LinerSVC.joblib')
        dump(tf_idf, '../inference/TfIdfVectorizer.joblib')
        print('model saved: ../inference/LinerSVC.joblib')
        print('vectorizer saved: ../inference/TfIdfVectorizer.joblib')

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(
                model, "model", registered_model_name="LinerSVC")
        else:
            mlflow.sklearn.log_model(model, "model")
