#! /usr/bin/python
import spacy
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES


class Language:
    def __init__(self):
        self.nlp = spacy.load('en')
        self.lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)

    def get_top_nouns(self, text):
        doc = self.nlp(unicode(text, 'utf-8'))
        ordered_keys = []
        counts = {}
        for token in doc:
            if token.pos_ == 'NOUN':
                lemmatized_noun = self.lemmatizer(token.text, token.pos_)[0]
                if lemmatized_noun in counts:
                    counts[lemmatized_noun] += 1
                else:
                    counts[lemmatized_noun] = 1
                    ordered_keys.append(lemmatized_noun)
        result = []
        for key in ordered_keys:
            if len(result) == 0:
                result.append(key)
            else:
                inserted = False
                for i, compare_key in enumerate(result):
                    if counts[compare_key] < counts[key]:
                        result.insert(i, key)
                        inserted = True
                        break
                if not inserted:
                    result.append(key)
        return [str(word) for word in result]


if __name__ == '__main__':
    print(Language().get_top_nouns('Hello friend, you are a cat and I am a bird, and cat\'s and birds are my best friends!'))
