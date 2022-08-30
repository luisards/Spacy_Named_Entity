"""
 Create training data using developed patterns. 
 Save the training data to a file. 
"""
import spacy
from spacy.matcher import Matcher
import json
import sys
import random

if len(sys.argv) < 2:
    sys.exit('Please speciify the input file') #askdocs_posts.json

filename = sys.argv[1]

# Load the reddit comments
with open(filename, 'r', encoding="utf-8") as f:
  redditJson = json.load(f)

# Extract the texts from json
# Key could be 'selftext', 'body'
texts = list()
for entry in redditJson:
    if 'selftext' in entry:
        texts.append(entry['selftext'])
    if 'body' in entry:
        texts.append(entry['body'])

#Consider 600 texts
data = texts[0:600]

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_sm")

# Common illnesses and conditions
condition_pattern = [
    {
        'LEMMA': { 'IN': ['depression', 'anxiety', 'allergies', 'diabetes', 'insomnia', 'hernia', 'endometriosis', 
                          'migraine', 'hypertension', 'flu', 'acne', 'hemorrhoids', 'asthma']},
        'POS': 'NOUN'
    }
]

# Illnesses referred to as acronyms. (The model doesn't seem to recognize most of them)
acronym_pattern = [
    {
        'LOWER': { 'IN': ['uti', 'adhd', 'ptsd', 'ocd', 'ibs', 'hiv', 'pcos']},
        'POS': 'NOUN'
    }
]

# All case-insensitive mentions of a herniated disc
compound_pattern3 = [{"LOWER": "herniated"}, {"LOWER": "disc"}]

# All case-insensitive mentions of a herniated disc
compound_pattern4 = [{"POS": "NOUN", "OP": "?"}, {"LOWER": "cancer"}]

# Create a Matcher
matcher = Matcher(nlp.vocab, validate=True)

# Add the patterns to the matcher
matcher.add("COND", None, condition_pattern)
matcher.add("COND", None, acronym_pattern)
matcher.add("COND", None, compound_pattern3)
matcher.add("COND", None, compound_pattern4)

TRAINING_DATA = []

# Process texts and run the matcher
for doc in nlp.pipe(data):
    # Match on the doc and create a list of matched spans
    spans = [doc[start:end] for match_id, start, end in matcher(doc)]
    
    # Get (start character, end character, label) tuples of matches
    entities = [(span.start_char, span.end_char, "COND") for span in spans]
    
    # Format the matches as a (doc.text, entities) tuple
    training_example = (doc.text, {"entities": entities})
    
    # Append the example to the training data
    TRAINING_DATA.append(training_example)

#Save training data to json file
json_file = json.dumps(TRAINING_DATA, indent = 4)

with open("training_data_askdocs.json", "w") as outfile:
    outfile.write(json_file)
