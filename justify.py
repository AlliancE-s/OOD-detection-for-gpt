from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from textblob import TextBlob
import dtwalign
import numpy as np


def add_answer(answer, answer_list):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    answer_list.append(model.encode(answer))
    return answer_list

def compute_similarity(list1, list2):
    if len(list1) == 0 or len(list2) == 0:
        return None
    else:
        list1_array = np.array(list1)
        list2_array = np.array(list2)
        dtw_distance = dtwalign.dtw(list1_array, list2_array)
        similarity = 1 / (1 + dtw_distance.distance)
        return similarity

def evaluate_answer_base(question, answer):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    question_vec = model.encode(question)
    answer_vec = model.encode(answer)
    sbert_similarity = 1 - cosine(question_vec, answer_vec)

    question_sentiment = TextBlob(question).sentiment
    answer_sentiment = TextBlob(answer).sentiment

# match = 1, not match = 0
    if question_sentiment.polarity == answer_sentiment.polarity:
        sentiment_match = 1
    else:
        sentiment_match = 0

    return sbert_similarity, sentiment_match

def evaluate_answer_dtw(answer_list):
    if (len(answer_list)==0):
        return None
    else:
        time_series_similarity = compute_similarity(answer_list[:-1], answer_list)
        return time_series_similarity