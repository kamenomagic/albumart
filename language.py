#! /usr/bin/python
import io
import spacy
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES


class Language:
    def __init__(self):
        self.nlp = spacy.load('en')
        self.lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)
        with io.open('cuss_words.txt', 'r', encoding='utf-8', errors='ignore') as cuss_words_file:
            self.cuss_words = cuss_words_file.read().replace('\n', '')

    def lemmatize(self, token):
        return self.lemmatizer(token.text, token.pos_)[0]

    def get_top_nouns(self, text):
        doc = self.nlp(unicode(text, 'utf-8'))
        ordered_keys = []
        counts = {}
        for token in doc:
            if token.text in self.cuss_words:
                continue
            if token.pos_ == 'NOUN':
                word_key = (token.text, self.lemmatize(token))
                inserted = False
                for word in counts.keys():
                    if word[1] == word_key[1]:
                        counts[word] += 1
                        inserted = True
                        break
                if not inserted:
                    counts[word_key] = 1
                    ordered_keys.append(word_key)
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
        return [str(word[0]) for word in result]


if __name__ == '__main__':
    print(Language().get_top_nouns('Hello friend, you are a cat and I am a bird, and birds and turtles are my best friends!'))
