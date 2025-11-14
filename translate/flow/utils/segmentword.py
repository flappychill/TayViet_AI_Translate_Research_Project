from collections import Counter, defaultdict
import math
import re
from config import CORPUS
class CorpusProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.corpus = self._load_corpus()
    def _load_corpus(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            return [line.strip().lower() for line in f if line.strip()]
    def get_corpus(self):
        return self.corpus
class PhraseExtractor:
    def __init__(self, corpus, max_ngram=5, min_freq=2, min_pmi=3):
        self.corpus = corpus
        self.max_ngram = max_ngram
        self.min_freq = min_freq
        self.min_pmi = min_pmi
        self.phrase_dict = self._build_phrase_dict()
    def _build_phrase_dict(self):
        ngram_counts = defaultdict(Counter)
        total_ngrams = defaultdict(int)
        word_counts = Counter()
        for sentence in self.corpus:
            words = re.findall(r'\w+', sentence)
            word_counts.update(words)
            for n in range(1, self.max_ngram + 1):
                if len(words) >= n:
                    ngrams = zip(*(words[i:] for i in range(n)))
                    ngram_list = list(ngrams)
                    ngram_counts[n].update(ngram_list)
                    total_ngrams[n] += len(ngram_list)
        phrase_dict = {}
        for n in range(2, self.max_ngram + 1):
            for ngram, count in ngram_counts[n].items():
                if count < self.min_freq:
                    continue
                p_ngram = count / total_ngrams[n]
                p_indep = math.prod(word_counts[w] / total_ngrams[1] for w in ngram)
                if p_indep <= 0:
                    continue
                pmi = math.log2(p_ngram / p_indep)
                if pmi >= self.min_pmi:
                    phrase_text = " ".join(ngram)
                    phrase_join = "_".join(ngram)
                    phrase_dict[phrase_text] = phrase_join
        return phrase_dict
    def get_phrases(self):
        return self.phrase_dict
class TextSegmenter:
    def __init__(self):
        corpus_file = CORPUS
        corpus_processor = CorpusProcessor(corpus_file)
        corpus = corpus_processor.get_corpus()
        phrase_extractor = PhraseExtractor(corpus, max_ngram=3, min_freq=2, min_pmi=5)
        phrase_dict = phrase_extractor.get_phrases()
        self.phrase_dict = phrase_dict
    def segment(self, sentence):
        sentence_lower = sentence.lower()
        for phrase, replacement in sorted(self.phrase_dict.items(), key=lambda x: -len(x[0])):
            pattern = rf"\b{re.escape(phrase)}\b"
            sentence_lower = re.sub(pattern, replacement, sentence_lower)
        return sentence_lower.split()
