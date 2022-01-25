import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("I have not been able to do this in years.")

for token in doc:
    print(token.dep_)