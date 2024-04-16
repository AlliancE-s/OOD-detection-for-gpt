import json
import justify

# load data
split = 'validation'
with open(f'dataset/{split}.json', 'r') as file:
    data = json.load(file)

# Preprocessing data and feature calculation
processed_data = []
count = 0

for dialogue in data:
    if count == 1:
        print(processed_data)
    if count > 49:
        print(f'----------stop-----------')
        break
    print(f"Processing NO.{count} ---- Dialogue ID: {dialogue['dialogue_id']}")
    answer_list = []
    for i in range(0, len(dialogue['turns']), 2):
        if i + 1 < len(dialogue['turns']):
            question = dialogue['turns'][i]['utterance']
            answer = dialogue['turns'][i+1]['utterance']
            sbert_similarity, sentiment_match = justify.evaluate_answer_base(question, answer)
            answer_list = justify.add_answer(answer, answer_list)
            time_series_similarity = justify.evaluate_answer_dtw(answer_list)
            if time_series_similarity == None:
                time_series_similarity = sbert_similarity
            processed_data.append({
                'sbert_similarity': sbert_similarity,
                'sentiment_match': sentiment_match,
                'time_series_similarity': time_series_similarity,
                'ood': dialogue['turns'][i+1]['ood']
            })
    count += 1



with open(f'dataset/{split}_c.json', 'w') as f:
    json.dump(processed_data, f, indent=4)