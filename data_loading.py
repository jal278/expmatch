import json
import numpy as np

class ashoka_data:
    raw_file_name = 'FINAL_processed_ashoka_full.json'
    embedding_file_name = f"data/batch_job_results_ashoka_embeddings_emblarge.jsonl"

    def __init__(self):
        # load in the raw data
        with open(self.raw_file_name, 'r') as f:
            self.json_data = json.load(f)

        records = []
        with open(self.embedding_file_name, 'r') as f:
            for idx,line in enumerate(f):
                # Parsing the JSON string into a dict and appending to the list of results
                json_object = json.loads(line.strip())
                # turn the embedding into a numpy array
                json_object['response']['body']['data'][0]['embedding'] = np.array(json_object['response']['body']['data'][0]['embedding'])

                # grab the task id and extract key
                task_id = json_object['custom_id']
                # Getting index from task id
                index = int(task_id.split('-')[-2])
                exp_num = int(task_id.split('-')[-1])
                _storage_key = (index,exp_num)
                emb = json_object['response']['body']['data'][0]['embedding']

                records.append((_storage_key, emb))

        self.records = records
        self.fellows_key = list(self.json_data.keys())
    
    def lookup_by_key(self,fellow_idx,exp_idx):
        fellow = self.fellows_key[fellow_idx]
        exp = self.json_data[fellow][exp_idx]
        return fellow, exp

    def get_string(self,fellow_idx,exp_idx):
        fellow, exp = self.lookup_by_key(fellow_idx,exp_idx)
        return f"Person: {fellow}\nInflection Summary: {exp}"
    
    def get_dict(self,fellow_idx,exp_idx):
        fellow, exp = self.lookup_by_key(fellow_idx,exp_idx)
        return {'Person':fellow, 'Inflection Summary':exp}
    
    def populate_search(self,es):
        for record in self.records:
            _storage_key, emb = record
            es.add_embedding(_storage_key, emb)

class storycorp_data:
    raw_file_name = 'transcript_database_v2.json'
    embedding_file_name = f"data/batch_job_results_life_circumstances_embeddings_emblarge.jsonl"

    def __init__(self):
        with open(self.raw_file_name, 'r') as f:
            self.json_data = json.load(f)

        records = []
        with open(self.embedding_file_name, 'r') as f:
            for line in f:
                json_object = json.loads(line.strip())
                json_object['response']['body']['data'][0]['embedding'] = np.array(json_object['response']['body']['data'][0]['embedding'])

                task_id = json_object['custom_id']
                index = int(task_id.split('-')[-2])
                exp_num = int(task_id.split('-')[-1])
                _storage_key = (index,exp_num)
                emb = json_object['response']['body']['data'][0]['embedding']
                records.append((_storage_key, emb))
        
        self.keys = list(self.json_data.keys())
        self.records = records

    def lookup_by_key(self,story_idx,exp_idx):
        url = self.keys[story_idx]
        data = self.json_data[url]
        exp = data['llm_life_circumstances'][exp_idx]
        return url, data, exp

    def get_string(self,story_idx,exp_idx):
        url, data, exp = self.lookup_by_key(story_idx,exp_idx)
        return f"URL: {url}\nStory: {data['llm_summary']}\nLife Circumstance: {exp}"
    
    def get_dict(self,story_idx,exp_idx):
        url, data, exp = self.lookup_by_key(story_idx,exp_idx)
        return {'URL':url, 'Title':data['title'], 'Summary':data['llm_summary'], 'Life Circumstance':exp}
    
    def populate_search(self,es):
        for record in self.records:
            _storage_key, emb = record
            es.add_embedding(_storage_key, emb)