from utils.classification import VietnameseTextAnalyzer
from utils.reconstruct_sentence import reconstructSentenceBatch
from utils.segmentword import TextSegmenter
from utils.translator import Translate_Model
from utils.search import SearchTranslator
from utils.best_candidate import BestCandidateSelector
from difflib import SequenceMatcher
from config import DICTIONARY_URL
class Translator:
    def __init__(self, classification_model, translator_model, selector_model, solr_url):
      
        self.analyzer = VietnameseTextAnalyzer(dictionary_url=DICTIONARY_URL, model_name=classification_model) 
        self.text_segmenter = TextSegmenter()
        self.solr_url = solr_url
        self.search_translator = SearchTranslator(solr_url)
        self.translator = Translate_Model(translator_model)
        self.selector = BestCandidateSelector(selector_model)
    def translate(self, sentence):
        test_sentence = sentence
        non_foreign_words, remaining_sentence = self.analyzer.analyze_sentence(test_sentence)
        segmented_text = self.text_segmenter.segment(remaining_sentence)
        words = self.analyzer.normalize_words(segmented_text)
        processed_results = self.processSentenceBatch(words, f'{self.solr_url}/select?indent=true&q.op=OR&q=')
        output_sentence = reconstructSentenceBatch(processed_results, non_foreign_words)
        return output_sentence
    def processSentenceBatch(self, words, solr_url):
     
        search_results = self.search_translator.search(words)                                   
        sentence = ''
        processed_results = []                                  
        i = 0
        non_dict_words = []                                     
        while i < len(words):
            word = words[i]
            if word.startswith('<') and word.endswith('>'):
                if non_dict_words:
                    temp_combined = ' '.join(non_dict_words)                                            
                    temp_translation = self.translator.translate(temp_combined.strip())
                    processed_results.append(temp_translation)
                    sentence += temp_translation + ' '
                    non_dict_words = []                                                              
                processed_results.append(word)
                sentence += word + ' '
                i += 1
                continue
            best_candidate = None
            best_match_length = 0
            best_combined_word = None
            for j in range(i, min(i + 4, len(words))):
                combined_word = ' '.join(words[i:j + 1])
                candidates = self.findRelatedCandidates(combined_word, search_results)
                if candidates:
                    best_candidate = self.selector.choose_best_candidate(sentence, candidates)
                    best_match_length = j - i + 1                               
                    best_combined_word = combined_word
            if best_candidate:
                if non_dict_words:
                    temp_combined = ' '.join(non_dict_words)                                            
                    temp_translation = self.translator.translate(temp_combined.strip())
                    processed_results.append(temp_translation)
                    sentence += temp_translation + ' '
                    non_dict_words = []                                                              
                processed_results.append(best_candidate)
                sentence += best_candidate + ' '
                i += best_match_length                          
            else:
                non_dict_words.append(word)
                i += 1
        if non_dict_words:
            temp_combined = ' '.join(non_dict_words)                                            
            temp_translation = self.translator.translate(temp_combined.strip())
            processed_results.append(temp_translation)
            sentence += temp_translation + ' '
        return processed_results
    def similarity_ratio(self, a, b):
     
        a = a.replace('_', ' ')                         
        b = b.replace('_', ' ')                         
        return SequenceMatcher(None, a, b).ratio()
    def findRelatedCandidates(self, word, search_results):
        
        related_candidates = []
        for result in search_results:
            if 'tay' in result and 'vietnamese' in result:
                tay_phrase = result['tay']
                vietnamese_candidates = result['vietnamese']
                similarity = self.similarity_ratio(word, tay_phrase)
                if similarity >= 0.85:                                    
                    related_candidates.extend(vietnamese_candidates)
        return related_candidates
