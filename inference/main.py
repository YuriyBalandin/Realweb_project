from preproccessing import *
import gradio as gr
from joblib import dump, load
from sklearn.svm import LinearSVC

model = load('LinearSVC.joblib')
tfidf = load('TfIdfVectorizer.joblib')


def get_preiction(text):
    text = pd.DataFrame([text])
    text.columns = ['review']

    text = preproccess(text, 'review')

    text = tfidf.transform(text['final_text'])

    result = model.predict(text)

    if result[0] == 0:
        return 'negtive'
    else:
        return 'positive'


if __name__ == "__main__":
    iface = gr.Interface(
        fn=get_preiction,
        inputs="text",
        outputs="text",
        title='Get review sentiment')
    iface.launch()
