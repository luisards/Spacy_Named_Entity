import spacy
from spacy.pipeline import EntityRuler
from spacy import displacy
import scispacy
from scispacy.abbreviation import AbbreviationDetector
import json
import sys
import random

if len(sys.argv) < 2:
    sys.exit('Too few arguments, please speciify the input file')

filename = sys.argv[1]
# Load the reddit comments
with open(filename, 'r', encoding="utf-8") as f:
  redditJson = json.load(f)

# Extract the texts from json
texts = list()
for entry in redditJson:
    if 'selftext' in entry:
        texts.append(entry['selftext'])
    if 'body' in entry:
        texts.append(entry['body'])

# Select n random elements from texts
n = 300
seed = 27
random.seed(seed)
randomTexts = random.sample(texts, n)

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_sm")

abbreviation_pipe = AbbreviationDetector(nlp)
nlp.add_pipe(abbreviation_pipe)

# Patterns
condition_pattern = {
    'label': 'COND',
    'pattern': [
        {
            'LOWER': { 'IN': doc._.abbreviations},
            'POS': 'NOUN'
        }
    ]
}

patterns = [condition_pattern]

# Create an Entity Ruler and add patterns
ruler = EntityRuler(nlp, overwrite_ents=True)
ruler.add_patterns(patterns)

# Add the Entity Ruler to the nlp pipeline
nlp.add_pipe(ruler, after="ner")

# Process texts with the Entity Ruler in the pipelne
# Process the texts one at a time because if nlp.pipe(randomTexts) is used,
# displacy doesn't work

docs = []
for text in randomTexts:
    doc = nlp(text)
    [print(ent.label_, ent.text) for ent in doc.ents if ent.label_ in ['COND']]

    hasING = False
    for ent in doc.ents:
        if ent.label_ == 'COND':
            hasING = True
            break

    if hasING:
        docs.append(doc)

displacy.serve(docs, style="ent", options = {"ents": ["COND"]})
displacy.serve(docs, style="dep")
