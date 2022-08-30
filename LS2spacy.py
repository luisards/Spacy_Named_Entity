######
# Convert Label Studio completions to train, dev, and test data,
# in the json format required for spacy v2 train CLI
#
# Completions are NOT included if:
#   - there is more than one completion for the task, or
#   - the task was skipped (cancelled)
######

import spacy
import srsly
from spacy.gold import docs_to_json, biluo_tags_from_offsets, spans_from_biluo_tags
import argparse
from zipfile import ZipFile
import json
import random


##
# Return a list of spacy docs, converted from Label Studio completions.
# Skip cancelled tasks and tasks with multiple completions.
##
def ls_to_spacy_json(ls_completions):
    nlp = spacy.load('en_core_web_sm')

    # Load the Label Studio completions
    with ZipFile(ls_completions, 'r') as zip:
        result_file = zip.read('result.json')
        label_studio_json = json.loads(result_file)

    gold_docs = []
    entity_cnt = 0
    entity_values = {}
    misaligned = 0
    for task in label_studio_json:
        completions = task['completions']

        # don't include skipped tasks or tasks with multiple completions
        if len(completions) == 1:
            completion = completions[0]
            if 'was_cancelled' in completion:
                continue

            raw_text = task['data']['reddit']
            annotated_entities = []
            for result in completion['result']:
                ent = result['value']
                start_char_offset = ent['start']
                end_char_offset = ent['end']
                ent_label = ent['labels'][0]
                entity = (start_char_offset, end_char_offset, ent_label)
                annotated_entities.append(entity)
                ent_text = ent['text']
                entity_values[ent_text] = entity_values.get(ent_text, 0) + 1

            doc = nlp(raw_text)
            tags = biluo_tags_from_offsets(doc, annotated_entities)
            entities = spans_from_biluo_tags(doc, tags)
            doc.ents = entities
            gold_docs.append(doc)
            entity_cnt += len(annotated_entities)
            misaligned += (len(annotated_entities) - len(doc.ents))

    print("{} entities in {} docs ({} misaligned)".format(str(entity_cnt), len(gold_docs), str(misaligned)))
    print("{} entity values: {}".format(str(len(entity_values)), entity_values))
    return gold_docs


##
# Split the docs into training, dev and test.
# Try to get the right percentage of entity examples in each dataset.
##
def split_docs(gold_docs, split_train, split_dev, split_test):
    train_docs = []
    dev_docs = []
    test_docs = []

    train_entities = 0
    dev_entities = 0
    test_entities = 0
    total_entities = 0

    # count the total number of entities
    for doc in gold_docs:
        total_entities += len(doc.ents)

    # shuffle the gold docs
    random.seed(27)
    random.shuffle(gold_docs)

    dev_ratio = split_dev / 100
    test_ratio = split_test / 100
    cur_train_ratio = -1
    cur_dev_ratio = -1
    cur_test_ratio = -1
    for doc in gold_docs:
        num_entities = len(doc.ents)
        if cur_test_ratio < test_ratio:
            test_docs.append(doc)
            test_entities += num_entities
            cur_test_ratio = test_entities / total_entities
        elif cur_dev_ratio < dev_ratio:
            dev_docs.append(doc)
            dev_entities += num_entities
            cur_dev_ratio = dev_entities / total_entities
        else:
            train_docs.append(doc)
            train_entities += num_entities
            cur_train_ratio = train_entities / total_entities

    print("{} train entities in {} docs ({} %)".format(str(train_entities), str(len(train_docs)), str(int(cur_train_ratio*100))))
    print("{} dev entities in {} docs ({} %)".format(str(dev_entities), str(len(dev_docs)), str(int(cur_dev_ratio*100))))
    print("{} test entities in {} docs ({} %)".format(str(test_entities), str(len(test_docs)), str(int(cur_test_ratio*100))))
    return train_docs, dev_docs, test_docs


##
# Save training and dev files
##
def save_data(train_docs, dev_docs, test_docs, train_file, dev_file, test_file):
    srsly.write_json(train_file, [docs_to_json(train_docs)])
    srsly.write_json(dev_file, [docs_to_json(dev_docs)])
    srsly.write_json(test_file, [docs_to_json(test_docs)])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("ls_completions", help="completions exported from Label Studio")
    parser.add_argument("split_train", type=int, help="percentage of data to use for training")
    parser.add_argument("split_dev", type=int, help="percentage of data to use for development")
    parser.add_argument("split_test", type=int, help="percentage of data to use for testing")
    parser.add_argument("train_file", help="file to save training data")
    parser.add_argument("dev_file", help="file to save dev data")
    parser.add_argument("test_file", help="file to save testing data")
    return parser.parse_args()


def main(args):
    gold_docs = ls_to_spacy_json(args.ls_completions)
    train_docs, dev_docs, test_docs = split_docs(gold_docs, args.split_train, args.split_dev, args.split_test)
    save_data(train_docs, dev_docs, test_docs, args.train_file, args.dev_file, args.test_file)


if __name__ == '__main__':
    main(parse_args())
