import streamlit as st
import json
import os
import numpy as np
from openai import OpenAI
from hybrid_search import embedding_search, bm25_search, hybrid_search

os.environ["OPENAI_API_KEY"] = st.secrets["OpenAI_key"]
extract_system_prompt = """The goal of this app is to help people find stories that are relevant to their life circumstances (problems or questions or transitions in life they may be facing).

Your goal is to analyze a transcript of a StoryCorps interview, and to create a short summary of it, and to list 3 life circumstances or life problems where hearing this story would be helpful and relevant, in order of increasing generality.

You will be provided with a transcript of the interview, and you will output a JSON object with the following format:
{
    "summary": str, // a one-sentence summary of the story
    "life_circumstances": list[str], // array of one-sentence life circumstances in order of increasing generality
}

The summary should be a short summary of what the story is about.
The life circumstances should be 3 life circumstances or life problems where hearing this story would be helpful and relevant, in order of increasing generality. For example ["A hispanic woman who is unexpectedly pregnant, very young, and without a father in the picture", "A woman who is unexpectedly pregnant","A person facing an unplanned scary situation without much support"]
"""

print("App running")
# get list of files in current directory
print("Current directory:")
print(os.listdir())
# get list of files in data directory
print("Data directory:")
print(os.listdir("data"))


client = OpenAI()

#model="text-embedding-3-small"
model="text-embedding-3-large"
model_short = "emblarge"
gen_model = "gpt-4o-mini"
    
if 'transcript_database' not in st.session_state.keys():
    with open('transcript_database_v2.json', 'r') as f:
        transcript_database = json.load(f)
    st.session_state['transcript_database'] = transcript_database
transcript_database = st.session_state['transcript_database']
keys = list(transcript_database.keys())

if 'embeddings' not in st.session_state.keys():
    results = []
    result_file_name = f"data/batch_job_results_life_circumstances_embeddings_{model_short}.jsonl"

    # load in embeddings
    with open(result_file_name, 'r') as file:
        for line in file:
            # Parsing the JSON string into a dict and appending to the list of results
            json_object = json.loads(line.strip())
            # turn the embedding into a numpy array
            json_object['response']['body']['data'][0]['embedding'] = np.array(json_object['response']['body']['data'][0]['embedding'])
            results.append(json_object)

    st.session_state['embeddings'] = results

embeddings = st.session_state['embeddings']

if 'es' not in st.session_state.keys():
    es = embedding_search()
    for res in embeddings[:]:
        task_id = res['custom_id']
        # Getting index from task id
        index = int(task_id.split('-')[-2])
        exp_num = int(task_id.split('-')[-1])
        _storage_key = (index,exp_num)
        result = res['response']['body']['data'][0]['embedding']
        es.add_embedding(_storage_key, result)
    print(f"Number of embeddings: {len(es.embeddings)}")
    docs = [transcript_database[key]['transcript'] for key in keys]
    bm25 = bm25_search(docs)
    hybrid = hybrid_search(bm25, es)
    st.session_state['es'] = es
    st.session_state['bm25'] = bm25
    st.session_state['hybrid'] = hybrid

es = st.session_state['es']
bm25 = st.session_state['bm25']
hybrid = st.session_state['hybrid']
    
st.write("""
         # Experience Search
         """)

text = st.text_area("Enter life circumstance")

# add a slider for weighting embedding and bm25
embedding_weight = st.slider("Semantic search weight", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
bm25_weight = 1 - embedding_weight

keys = list(transcript_database.keys())

if st.button('Search'):
    if text:
        embedding = client.embeddings.create(input=text, model=model).data[0].embedding
        #search_results = es.search_embedding(embedding)
        search_results = hybrid.search(text, embedding, 10, [embedding_weight, bm25_weight])
        for r in search_results:
            index,exp_num = r[0]
            score = r[1]

            #print(f"Index: {index}, Experience Number: {exp_num}, Score: {score}")
            st.write(f"Score: {score}")
            key = keys[index]
            title = transcript_database[key]['title']
            llm_summary = transcript_database[key]['llm_summary']
            exp = transcript_database[key]['llm_life_circumstances'][exp_num]
            url = transcript_database[key]['url']

            st.write(f"Title: {title}")
            st.write(f"LLM summary: [{llm_summary}]({url})")
            st.write(f"Matching life Circumstance: {exp}")
            st.write(f"***")


if st.button('debug'):
    print(len(es.embeddings))
    st.write(f"Number of embeddings: {len(es.embeddings)}")
    print(len(transcript_database))
    print(keys[0])
    print(keys[0] in transcript_database)

st.write("# Life circumstance extractor")

prompt = st.text_area("System prompt", value=extract_system_prompt, height=600)
gen_model = st.radio("Model", ["gpt-4o-mini", "gpt-4o"], index=0)
key = st.selectbox("Select a transcript", keys)

if st.button('Extract'):
    if prompt:
        _transcript = transcript_database[key]

        completion = client.chat.completions.create(
            model=gen_model,
            response_format={'type': 'json_object'},
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": _transcript['transcript']}],
            temperature=0.1,
            max_tokens=200)
        content = completion.choices[0].message.content
        st.write(completion.choices[0].message.content)
            