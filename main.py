import sys
from openai import OpenAI
import justify
from joblib import load
import pandas as pd
import os
from dotenv import load_dotenv

# Specify the path to the .env file
dotenv_path = 'OPENAI_API_KEY.env'

# Loading .env files
load_dotenv(dotenv_path=dotenv_path)

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def evaluate(question, answer, answer_list):
    # Evaluation Answers
    sbert_similarity, sentiment_match = justify.evaluate_answer_base(question, answer)

    answer_list = justify.add_answer(answer, answer_list)

    time_series_similarity = justify.evaluate_answer_dtw(answer_list)
    if time_series_similarity == None:
        time_series_similarity = sbert_similarity

    # Loading Models
    model = load('model/new_model.joblib')

    # Build the data
    data = {
        'sbert_similarity': sbert_similarity,
        'sentiment_match': sentiment_match,
        'time_series_similarity': time_series_similarity
    }
    # Creating a DataFrame
    df = pd.DataFrame([data])

    # Prediction using models
    prediction = model.predict(df)

    if prediction == 0:
        prediction_text = 'Not OOD'
    else:
        prediction_text = 'OOD'

    return answer_list, sbert_similarity, sentiment_match, time_series_similarity, prediction_text
def continue_conversation(showdata = 'n',frist_question= '', background='You need to answer the following question.' ):
    answer_list = []


    messages = [
        {"role": "system", "content": background},
        {"role": "user", "content": frist_question}
    ]

    initial_question = messages[-1]["content"]

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    initial_response_content = completion.choices[0].message.content
    print("Response:", initial_response_content)

    # Getting questions and answers
    question = initial_question
    answer = initial_response_content

    # Evaluation Answers
    answer_list, sbert_similarity, sentiment_match, time_series_similarity, prediction = evaluate(question, answer,
                                                                                                  answer_list)

    print('OOD prediction:', prediction)
    if showdata == 'y':
        print('sbert_similarity:', sbert_similarity)
        print('sentiment_match:', sentiment_match)
        print('time_series_similarity:', time_series_similarity)


    while True:
        user_input = input("text your next question (type 'quit' to quit): ")
        if user_input.lower() == 'quit':
            break

        messages.append({"role": "user", "content": user_input})

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )

        response_content = completion.choices[0].message.content
        print("Response:", response_content)

        # Getting questions and answers
        question = messages[-2]["content"]
        answer = response_content

        # Evaluation Answers
        answer_list, sbert_similarity, sentiment_match, time_series_similarity, prediction = evaluate(question,answer,answer_list)

        print('OOD prediction:', prediction)
        if showdata == 'y':
            print('sbert_similarity:', sbert_similarity)
            print('sentiment_match:', sentiment_match)
            print('time_series_similarity:', time_series_similarity)



# Calling a function to start an ongoing dialogue

background = 'You need to answer the following question.'
showdata = input(f'Do you want to see the detail data? (y or n)')
first_input = input("text your first question (type 'quit' to quit): ")
if first_input.lower() == 'quit':
    sys.exit()
first_question = first_input
continue_conversation(showdata,first_question, background)
