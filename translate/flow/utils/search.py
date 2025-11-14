import json
import pandas as pd
from urllib3 import PoolManager
from urllib.parse import quote
from config import WORD_URL
class SolrClient:
 
    def __init__(self, solr_url):
        self.solr_url = solr_url.rstrip('/')
        self.http = PoolManager()
    def delete_all_documents(self):
       
        delete_query = '<delete><query>*:*</query></delete>'
        headers = {"Content-Type": "text/xml"}
        response = self.http.request('POST', f'{self.solr_url}/update?commit=true', body=delete_query, headers=headers)
        if response.status == 200:
            print("All data on Solr has been deleted.")
        else:
            print(f"❌ Lỗi khi xóa dữ liệu: {response.status}, {response.data.decode('utf-8')}")
    def upload_documents(self, data):
       
        headers = {'Content-Type': 'application/json'}
        response = self.http.request('POST', f'{self.solr_url}/update?commit=true', body=json.dumps(data).encode('utf-8'), headers=headers)
        if response.status == 200:
            print("The data has been successfully uploaded to Solr!")
        else:
            print(f"❌ Lỗi khi tải dữ liệu lên Solr: {response.data.decode('utf-8')}")
    def search_tay_words(self, words):
      
        or_query = " OR ".join([f'tay:"{quote(word)}"' for word in words])
        search_url = f'{self.solr_url}/select?indent=true&q.op=OR&q=({or_query})&rows=1000&fl=tay,vietnamese&wt=json'
        response = self.http.request('GET', search_url)
        try:
            data = json.loads(response.data.decode('utf-8'))
        except json.JSONDecodeError:
            return []
        if 'response' not in data:
            return []
        results = {}
        for doc in data['response']['docs']:
            tay_word = doc.get('tay', [''])[0]
            vietnamese_word = doc.get('vietnamese', [''])[0]
            if tay_word and vietnamese_word:
                if tay_word not in results:
                    results[tay_word] = []
                results[tay_word].append(vietnamese_word)
        final_results = [{"tay": k, "vietnamese": list(set(v))} for k, v in results.items()]
        return final_results
class GoogleSheetsClient:
   
    def __init__(self, sheet_url):
        self.sheet_url = sheet_url
    def read_csv(self):
       
        df = pd.read_csv(self.sheet_url)
        return df[['tiếng tay', 'tiếng việt']].rename(columns={'tiếng tay': 'tieng_tay', 'tiếng việt': 'tieng_viet'})
class SearchTranslator:
   
    def __init__(self, solr_url):
        solr_client = SolrClient(solr_url)
        google_sheets_client = GoogleSheetsClient(WORD_URL)
        df = google_sheets_client.read_csv()
        solr_client.delete_all_documents()
        documents = [{"tay": row["tieng_tay"], "vietnamese": row["tieng_viet"]} for _, row in df.iterrows()]
        solr_client.upload_documents(documents)
        self.solr_client = solr_client
        self.solr_url = solr_url
    def deleteQuery(self, url = 'http://localhost:8983/solr/mycore/update?commit=true'):
        http = PoolManager()
        r = http.request('POST', url, body=b'<delete><query>*:*</query></delete>', headers={'Content-Type': 'text/xml'})
        return
    def search(self, words):
      
        self.deleteQuery(self.solr_url)
        return self.solr_client.search_tay_words(words)
