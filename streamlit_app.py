import streamlit as st
import json
import os
import numpy as np
from openai import OpenAI
from hybrid_search import embedding_search, bm25_search, hybrid_search
from data_loading import ashoka_data, storycorp_data

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

chat_model = "gpt-4o"

if 'ashoka' not in st.session_state.keys():
    st.session_state['ashoka'] = ashoka_data()
ashoka = st.session_state['ashoka']

if 'storycorp' not in st.session_state.keys():
    st.session_state['storycorp'] = storycorp_data()
storycorp = st.session_state['storycorp']

if 'es_ashoka' not in st.session_state.keys():
    es_ashoka = embedding_search()
    ashoka.populate_search(es_ashoka)

    es_storycorp = embedding_search()
    storycorp.populate_search(es_storycorp)

    print(f"Number of embeddings: {len(es_ashoka.embeddings)}")
    print(f"Number of embeddings: {len(es_storycorp.embeddings)}")
    #docs = [transcript_database[key]['transcript'] for key in keys]
    #bm25 = bm25_search(docs)
    #hybrid = hybrid_search(bm25, es)
    st.session_state['es_ashoka'] = es_ashoka
    st.session_state['es_storycorp'] = es_storycorp
    #st.session_state['bm25'] = bm25
    #st.session_state['hybrid'] = hybrid

es_ashoka = st.session_state['es_ashoka']
es_storycorp = st.session_state['es_storycorp']
#bm25 = st.session_state['bm25']
#hybrid = st.session_state['hybrid']

def intro():
    st.write("# Welcome to Streamlit! 👋")
    st.sidebar.success("Select a demo above.")

def exp_search():
    st.write("""
            # Experience Search
            """)
    
    # make a radio button to select the dataset
    dataset = st.radio("Select a dataset", ["Ashoka","StoryCorps"])
    if dataset == "Ashoka":
        es = es_ashoka
    else:
        es = es_storycorp

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    stages = ["Enter life circumstance", "Reply", "Done"]
    if 'stage' not in st.session_state.keys():
        st.session_state['stage'] = 0
    #text = st.text_area("Enter life circumstance")

    # add a slider for weighting embedding and bm25
    #embedding_weight = st.slider("Semantic search weight", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    #bm25_weight = 1 - embedding_weight

    #st.write(f"Semantic search weight: {embedding_weight}, plain-text search weight: {bm25_weight}")
    #st.write(f"Current stage: {st.session_state['stage']}")

    #keys = list(transcript_database.keys())

    stage_text = stages[st.session_state['stage']]
    #text = st.chat_input(stage_text)

    if text:= st.chat_input(stage_text):

        with st.chat_message("user"):
            st.markdown(text)
        st.session_state.messages.append({"role": "user", "content": text})

        if text and st.session_state['stage'] == 0:
            embedding = client.embeddings.create(input=text, model=model).data[0].embedding
            _,search_results = es.search_embedding(embedding)
            #search_results = hybrid.search(text, embedding, 10, [embedding_weight, bm25_weight])

            response = ""
            for idx, r in enumerate(search_results):
                index,exp_num = r[0]
                score = r[1]

                #print(f"Index: {index}, Experience Number: {exp_num}, Score: {score}")
                response += f"*Rank {idx+1}, Score: {score}*\n\n"

                if dataset == "Ashoka":
                    data = ashoka.get_dict(index,exp_num)
                else:
                    data = storycorp.get_dict(index,exp_num)

                #st.write(f"Score: {score}")
                for k,v in data.items():
                    response += f"**{k}:** {v}\n\n"
                response += f"***\n\n"
            # now we should query the LLM to ask if there are any follow-up questions that are relevant to the user's life circumstance
            prompt = f"The user has entered the following life circumstance: {text}. \
                Here are {len(search_results)} stories that are relevant to this life circumstance. \
                Please generate a follow-up question useful to get more information about the user's life circumstance to best re-rank the stories to return the most relevant ones."
            
            prompt += f"\n\nHere are the stories:\n{response}"
            prompt += "\n\nNow respond with the follow-up question."

            st.session_state['search_results'] = response


            completion = client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "system", "content": prompt}, {"role": "user", "content": ""}],
                temperature=0.1,
                max_tokens=200)
            chat_response = completion.choices[0].message.content

            with st.chat_message("assistant"):
                st.markdown(response)
            with st.chat_message("assistant"):
                st.markdown(chat_response)       
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.messages.append({"role": "assistant", "content": chat_response})
            st.session_state['stage'] += 1

        elif text and st.session_state['stage'] == 1:
            # now rerank the results
            # now create llm prompt to rerank the results
            chat_response = st.session_state.messages[-1]['content']
            prompt = f"The user first entered the following life circumstance: {text}. \
                Then we asked a follow-up question to improve the search result: {chat_response}. \
                The user then replied with the following: {text}. \
                Please re-rank the stories based on the user's reply to the follow-up question, and return the 3 most relevant stories, make sure to include the URLs."

            prompt += f"\n\nHere are the stories:\n{st.session_state['search_results']}"

            completion = client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "system", "content": prompt}, {"role": "user", "content": ""}],
                temperature=0.1,
                max_tokens=600)
            chat_response = completion.choices[0].message.content

            with st.chat_message("assistant"):
                st.markdown(chat_response)       
            
            st.session_state.messages.append({"role": "assistant", "content": chat_response})
            st.session_state['stage'] += 1

    if st.button('Clear'):
        st.session_state.messages = []
        st.session_state['stage'] = 0


def prompt_playground():
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


page_names_to_funcs = {
    "—": intro,
    "Experience Search": exp_search,
    "Prompt Playground": prompt_playground,
}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()