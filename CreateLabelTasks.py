import spacy
from spacy.pipeline import EntityRuler
import json
import sys

####
# Create labeling tasks for Label Studio. Labeling tasks
# include at least one 'ING' entity found by
# spaCy's EntitRuler pattern matcher.
#
# https://spacy.io/api/entityruler
# https://spacy.io/usage/rule-based-matching#entityruler
####

if len(sys.argv) < 3:
    sys.exit('Too few arguments, please speciify the input reddit data and the file name to store labeling tasks')

inFile = sys.argv[1]
outFile = sys.argv[2]
# Load the reddit comments
with open(inFile, 'r', encoding="utf-8") as f:
  redditJson = json.load(f)

# Extract the texts from json
# Key could be 'selftext', 'body'
texts = list()
for entry in redditJson:
    if 'selftext' in entry:
        texts.append(entry['selftext'])
    if 'body' in entry:
        texts.append(entry['body'])

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_sm")

# List of some common ingredients
conditions = [
    'depression', 'anxiety', 'allergies', 'diabetes', 'insomnia', 'hernia', 'endometriosis',
    'migraine', 'hypertension', 'flu', 'acne', 'hemorrhoids', 'asthma', 'uti', 'adhd', 'ptsd', 'ocd', 'ibs', 'hiv', 'pcos'
]

# Ingredient pattern
condition_match = {
    'POS': 'NOUN',
    'LEMMA': { 'IN': conditions }
}

# Entity Patterns

# Just the ingredient
condition_pattern = {
    'label': 'COND',
    'pattern': [ condition_match ]
}

# Noun followed by an ingredient
noun_cond_pattern = {
    'label': 'COND',
    'pattern': [
        {
            'POS': 'NOUN',
        },
        condition_match
    ]
}

# Adj followed by an ingredient
adj_cond_pattern = {
    'label': 'COND',
    'pattern': [
        {
            'POS': 'ADJ',
        },
        condition_match
    ]
}

patterns = [condition_pattern, noun_cond_pattern, adj_cond_pattern]

# Create an Entity Ruler and add patterns
ruler = EntityRuler(nlp, overwrite_ents=True)
ruler.add_patterns(patterns)

# Add the Entity Ruler to the nlp pipeline
nlp.add_pipe(ruler)

# Process texts with the Entity Ruler in the pipelne
# Create a labeling task for each doc that has at least
# one entity of type 'COND'

LABELING_DATA = []
for text in texts:
    doc = nlp(text)

    # get list of labels for this doc
    labels = [ent.label_ for ent in doc.ents if ent.label_ == 'COND']

    # if the doc has at least one 'ING' entity,
    # add it to the labeling task list
    if labels:
        # Append doc.text to the Label Studio labeling tasks
        task = {}
        task['reddit'] = doc.text
        LABELING_DATA.append(task)

print("{} tasks created from {} docs.".format(len(LABELING_DATA), len(texts)))

with open(outFile, 'w', encoding="utf-8") as f:
    f.write(json.dumps(LABELING_DATA, indent=4))
