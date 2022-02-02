import pandas as pd
import spacy
from spacy.tokens import Doc
pd.options.mode.chained_assignment = None  # default='warn'


class DataProcessing:
    """
    Features:

    POS: part of speech tag
    LEMMA: lemma (lowercased)
    TAG: Detailed part of speech tag
    DEP: syntactic dependency
    STOP: whether the token is a stopword
    NER: Named entity tag
    AFFIX: whether an affixal negation cue is present in the token
    CONTR: Whether a contracted negation cue is in the token
    EXPR: Whether the token matches one of the highly probable negation cue expressions

    """
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.contracted_negation_cues = ["'t", "not", "n't"]
        self.affixal_negation_cues = ['de', 'dis', 'il', 'im', 'in', 'ir', 'mis', 'non', 'un', 'anti']
        self.negation_expressions = ['nor', 'neither', 'without', 'nobody', 'none', 'nothing', 'never', 'not', 'no', 'nowhere', 'non']
        self.df = pd.DataFrame()

    def load_data(self, path):
        df = pd.read_csv(path, sep='\s+', names=['document', 'sentence ID', 'token ID', 'token', 'cue', 'POS', 'LEMMA', 'TAG', 'DEP', 'STOP', 'NER', 'AFFIX', 'CONTR', 'EXPR'], header=None)
        self.df = df

    def save_data(self, path):
        self.df.to_csv(path, sep='\t', header=False, index=False, na_rep='0')

    def process_corpus(self):
        # Get all different documents in the corpus
        documents = self.df['document'].unique()

        for document in documents:
            print(f'Processing document: {document}')
            # Get all sentences in the document
            doc_df = self.df.loc[self.df['document'] == document, :]
            sentences = doc_df['sentence ID'].unique()

            for sentence_id in sentences:
                # Do the preprocessing on the sentence level
                sentence_df = doc_df.loc[doc_df['sentence ID'] == sentence_id, :]
                sentence = ' '.join(sentence_df['token'].tolist())

                # TODO: we already have tokenized text, so maybe we can use the pre-tokenized text feature of spacy?
                #   https://spacy.io/usage/linguistic-features#own-annotations
                # sentence_word_list = sentence_df['token'].tolist()
                # doc = Doc(self.nlp.vocab, words=words)

                # Get linguistic features through Spacy pipeline
                doc = self.nlp(sentence)

                # Extract all features and add them to the df
                self.label_linguistic_features(doc, sentence_df)
                self.label_named_entities(doc, sentence_df)
                self.label_affixal_negation(doc, sentence_df)
                self.label_contracted_negation(doc, sentence_df)
                self.label_probable_negation_expression(doc, sentence_df)

    def label_linguistic_features(self, doc, sentence_df):
        for token in doc:
            # Get the df row for this specific token
            token_df = sentence_df.loc[sentence_df['token'] == token.text, :]

            # Add linguistic features as columns
            token_df['POS'] = token.pos_
            token_df['LEMMA'] = token.lemma_
            token_df['TAG'] = token.tag_
            token_df['DEP'] = token.dep_
            token_df['STOP'] = token.is_stop

            self.df.update(token_df)

    def label_named_entities(self, doc, sentence_df):
        for ent in doc.ents:
            # Get the df row for this specific token
            token_df = sentence_df.loc[sentence_df['token'] == ent.text, :]

            # Apply NER tag to token row
            token_df['NER'] = ent.label_

            self.df.update(token_df)

    def label_contracted_negation(self, doc, sentence_df):
        for token in doc:
            # Get the df row for this specific token
            token_df = sentence_df.loc[sentence_df['token'] == token.text, :]

            token_df['CONTR'] = False

            # Loop over the contracted negation cues and check if the token ends with it, and if so assign true
            for contr_cue in self.contracted_negation_cues:
                if token.text.endswith(contr_cue):
                    token_df['CONTR'] = True
                    self.df.update(token_df)
                    break

            self.df.update(token_df)

    def label_affixal_negation(self, doc, sentence_df):
        for token in doc:
            # Get the df row for this specific token
            token_df = sentence_df.loc[sentence_df['token'] == token.text, :]

            token_df['AFFIX'] = False

            for affix in self.affixal_negation_cues:
                if token.text.startswith(affix):
                    token_df['AFFIX'] = True
                    self.df.update(token_df)
                    break

            self.df.update(token_df)

    def label_probable_negation_expression(self, doc, sentence_df):
        for token in doc:
            # Get the df row for this specific token
            token_df = sentence_df.loc[sentence_df['token'] == token.text, :]

            token_df['EXPR'] = False

            for expr in self.negation_expressions:
                if token.text == expr:
                    token_df['EXPR'] = True
                    self.df.update(token_df)
                    break

            self.df.update(token_df)


def preprocess_data(input_file, output_file):
    data_processor = DataProcessing()
    data_processor.load_data(input_file)
    data_processor.process_corpus()
    data_processor.save_data(output_file)


def main(input_path, output):
    preprocess_data(input_path, output)


if __name__ == '__main__':
    input_file = './dataset/SEM-2012-SharedTask-CD-SCO-test-circle.txt'
    output_file = './dataset/processed_corpus_test_set_circle.csv'
    main(input_file, output_file)


