import spacy
import re
import requests
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import os
class VietnameseTextAnalyzer:
    def __init__(self, dictionary_url=None, model_name="undertheseanlp/vietnamese-ner-v1.4.0a2", dictionary_folder="data"):
      
        os.makedirs(dictionary_folder, exist_ok=True)
        dictionary_file = os.path.join(dictionary_folder, "vietnamese_words.xlsx")
        if dictionary_url:
            self.download_vietnamese_dictionary(dictionary_url, dictionary_file)
        self.vietnamese_dict = self.load_vietnamese_dictionary(dictionary_file)
        self.nlp_spacy = spacy.blank("vi")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.ner_model = pipeline("ner", model=self.model, tokenizer=self.tokenizer, aggregation_strategy="simple")
    def download_vietnamese_dictionary(self, url, file_path):
  
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(response.content)
        else:
            print(f"Không thể tải từ điển. Mã lỗi: {response.status_code}")
    def load_vietnamese_dictionary(self, file_path):
     
        df = pd.read_excel(file_path, usecols=[0], header=None)
        return set(df[0].str.strip().str.lower().dropna())
    def is_special_character(self, word):
      
        return bool(re.match(r"[#$%&()*+,-./:;<=>?@\[\]^`{}~]", word))
    def is_number(self, word):
      
        return word.isdigit() or bool(re.match(r"^0*\d+(\.\d+)?$", word))
    def is_date(self, word):
     
        date_formats = [            r"^\d{1,2}/\d{1,2}/\d{4}$",            r"^\d{1,2}-\d{1,2}-\d{4}$",            r"^\d{4}/\d{1,2}/\d{1,2}$",            r"^\d{4}-\d{1,2}-\d{1,2}$",        ]
        return any(re.match(date_format, word) for date_format in date_formats)
    def is_vietnamese_word(self, word):
     
        return word.lower() in self.vietnamese_dict
    def analyze_sentence(self, sentence):
     
        doc = self.nlp_spacy(sentence)
        tokens = [token.text for token in doc]
        results = []
        for word in tokens:
            if self.is_special_character(word):
                results.append((word, "Ký tự đặc biệt"))
            elif self.is_number(word):
                results.append((word, "Số"))
            elif self.is_date(word):
                results.append((word, "Ngày tháng năm"))
            elif self.is_vietnamese_word(word):
                results.append((word, "Tiếng Việt"))
            else:
                results.append((word, "Ngôn ngữ khác"))
        non_foreign_words = [word for word, category in results if category != "Ngôn ngữ khác"]
        remaining_sentence = " ".join([f"<word>" if category != "Ngôn ngữ khác" else word                                       for word, category in results])
        return non_foreign_words, remaining_sentence
    def normalize_words(self, word_list):
      
        return [word.replace('_', ' ').lower() for word in word_list]
