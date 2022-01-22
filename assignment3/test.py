import spacy
import textacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

ngrams = list(textacy.extract.basics.ngrams(doc, 2, min_freq=2))

print(ngrams)
