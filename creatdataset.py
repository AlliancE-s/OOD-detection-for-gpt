import random
import json
from datasets import load_dataset


def get_random_response(responses, current_dialogue_id):
    other_ids = list(responses.keys())
    other_ids.remove(current_dialogue_id)
    if not other_ids:
        return None
    random_dialogue_id = random.choice(other_ids)
    if responses[random_dialogue_id]:
        return random.choice(responses[random_dialogue_id])
    return None


def replace_responses_and_mark_ood(dialogue_id, speakers, utterances, responses, replacement_chance=0.2):
    modified_utterances = utterances.copy()
    ood_flags = [0] * len(utterances)  # Initialisation ood is marked as 0

    for i, (speaker, utterance) in enumerate(zip(speakers, utterances)):
        if speaker == 1 and random.random() < replacement_chance:
            new_response = get_random_response(responses, dialogue_id)
            if new_response:
                modified_utterances[i] = new_response
                ood_flags[i] = 1  # Mark ood as 1 after replacement

    return modified_utterances, ood_flags


def process_dataset(dataset,split):
    modified_dataset = []
    all_responses = {example['dialogue_id']: [utterance for speaker, utterance in
                                              zip(example['turns']['speaker'], example['turns']['utterance']) if
                                              speaker == 1]
                     for example in dataset[split]}

    for example in dataset[split]:
        dialogue_id = example['dialogue_id']
        speakers = example['turns']['speaker']
        utterances = example['turns']['utterance']
        new_utterances, ood_flags = replace_responses_and_mark_ood(dialogue_id, speakers, utterances, all_responses,
                                                                   replacement_chance=0.2)

        modified_turns = []
        for turn_id, speaker, utterance, ood in zip(example['turns']['turn_id'], speakers, new_utterances, ood_flags):
            modified_turns.append({
                'turn_id': turn_id,
                'speaker': speaker,
                'utterance': utterance,
                'ood': ood
            })

        modified_entry = {
            'dialogue_id': example['dialogue_id'],
            'services': example['services'],
            'turns': modified_turns
        }
        modified_dataset.append(modified_entry)

    with open(f'dataset/{split}.json', 'w') as f:
        json.dump(modified_dataset, f, indent=4)


# Load the dataset with 'trust_remote_code=True' to avoid the warning
dataset = load_dataset("multi_woz_v22", trust_remote_code=True)
split = 'validation'
process_dataset(dataset,split)
