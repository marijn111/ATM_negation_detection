import spacy
from spacy.tokens import DocBin
from stanza.utils.conll import CoNLL

doc = CoNLL.conll2doc("./dataset/SEM-2012-SharedTask-CD-SCO-training-simple.v2.conll")

print(*[f'word: {word.text}\tupos: {word.upos}\txpos: {word.xpos}\tfeats: {word.feats if word.feats else "_"}' for sent in doc.sentences for word in sent.words], sep='\n')


# nlp = spacy.load("en_core_web_sm")
#
# doc_bin = DocBin().from_disk("./dataset/SEM-2012-SharedTask-CD-SCO-training-simple.v2.spacy")
#
# docs = list(doc_bin.get_docs(nlp.vocab))
#
# i = 0
#
# for doc in docs:
#     print(doc.text)
#     if i == 30:
#         exit()
#
#     i += 1


# Pandas do with columns. Since sentence ID is present, can be splitted into words. Then do all the things of lemma etc. Add it to the corresponding column
# Go from there.