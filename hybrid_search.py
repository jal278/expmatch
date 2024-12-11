# %%
import numpy as np
from rank_bm25 import BM25Okapi


class bm25_search:
    def __init__(self, documents):
        self.documents = documents
        #todo: can improve tokenization
        self.tokenized_corpus = [doc.split() for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
    
    def search(self, query, top_k=10):
        # todo: can improve tokenization
        tokenized_query = query.split()
        doc_scores = self.bm25.get_scores(tokenized_query)

        max_score = np.max(doc_scores)
        if max_score == 0:
            max_score = 1
        doc_scores = doc_scores/max_score

        top_k_indices = np.argsort(doc_scores)[::-1][:top_k]
        return doc_scores, [(i, doc_scores[i]) for i in top_k_indices]
    
class embedding_search:
    def __init__(self):
        self.embeddings = []

    def add_embedding(self, key, embedding):
        self.embeddings.append((key,embedding))

    def search_embedding(self,embedding,top_k=10):
        # calculate cosine similarity
        def cosine_similarity(a, b):    
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        similarities = [(e[0],cosine_similarity(embedding, e[1])) for e in self.embeddings]
        # find the top 10 most similar embeddings
        top_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
        # return the keys and similarity scores
        return similarities,top_similarities

class hybrid_search:
    def __init__(self, bm25, embedding_search):
        self.bm25 = bm25
        self.embedding_search = embedding_search
    
    def search(self, query, query_emb, top_k=10,weights=[1.0,0.25]):
        bm_scores, bm25_results = self.bm25.search(query, top_k)
        embedding_scores, embedding_results = self.embedding_search.search_embedding(query_emb)

        combined_scores = []
        for i,emb_score in embedding_scores:
            emb_score = emb_score*weights[0]
            bm_score = bm_scores[i[0]]*weights[1]
            combined_scores.append((i,emb_score+bm_score))

        combined_scores = sorted(combined_scores, key=lambda x: x[1], reverse=True)
        return combined_scores,combined_scores[:top_k]


# %%
if __name__ == "__main__":

    text_query = "An existential crisis"
    embedding_query = client.embeddings.create(input=text_query, model=model).data[0].embedding

    # %%
    es = embedding_search()
    for res in results[:]:
        task_id = res['custom_id']
        # Getting index from task id
        index = int(task_id.split('-')[-2])
        exp_num = int(task_id.split('-')[-1])
        _storage_key = (index,exp_num)
        result = res['response']['body']['data'][0]['embedding']

        es.add_embedding(_storage_key, result)
    # %%
    docs = [transcript_database[key]['transcript'] for key in keys]
    bm25 = bm25_search(docs)
    # %%
    bm_scores, bm25_results = bm25.search(text_query, 10)
    embedding_scores, embedding_results = es.search_embedding(embedding_query)
    # %%
    combined_scores = []
    weights = [1.0,0.25] #embedding, bm25
    for i,emb_score in embedding_scores:
        emb_score = emb_score*weights[0]
        bm_score = bm_scores[i[0]]*weights[1]
        combined_scores.append((i,emb_score+bm_score))
    # %%
    # sort by score
    combined_scores = sorted(combined_scores, key=lambda x: x[1], reverse=True)
    # %%
    for i,score in combined_scores[:10]:
        index,exp_num = i
        print(f"Index: {index}, Experience Number: {exp_num}, Score: {score}")
        key = keys[index]
        llm_summary = transcript_database[key]['llm_summary']
        exp = transcript_database[key]['llm_life_circumstances'][exp_num]
        url = transcript_database[key]['url']   
        print(f"LLM Summary: {llm_summary}")
        print(f"Life Circumstances: {exp}")
        print(f"URL: {url}")
        print("\n")
    # %%
