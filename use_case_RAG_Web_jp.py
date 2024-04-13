"""
author: Elena Lowery and Catherine Cao

This code sample shows how to invoke Large Language Models (LLMs) deployed in watsonx.ai.
Documentation: # https://ibm.github.io/watson-machine-learning-sdk/foundation_models.html#
You will need to provide your IBM Cloud API key and a watonx.ai project id (any project)
for accessing watsonx.ai
This example shows a Question and Answer use case for a provided web site


# Install the wml api your Python env prior to running this example:
# pip install ibm-watson-machine-learning

# Install chroma
# pip install chromadb

IMPORTANT: Be aware of the disk space that will be taken up by documents when they're loaded into
chromadb on your laptop. The size in chroma will likely be the same as .txt file size
"""

# For reading credentials from the .env file
import os
from dotenv import load_dotenv

from langchain.text_splitter import HTMLHeaderTextSplitter, RecursiveCharacterTextSplitter
from chromadb.utils import embedding_functions
from udfs import *

# WML python SDK
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods

import chromadb


# These global variables will be updated in get_credentials() functions
watsonx_project_id = ""
api_key = ""
url = ""

def get_credentials():

    load_dotenv()

    # Update the global variables that will be used for authentication in another function
    globals()["api_key"] = os.getenv("api_key", None)
    globals()["watsonx_project_id"] = os.getenv("project_id", None)
    globals()["url"] = os.getenv("url", None)

# The get_model function creates an LLM model object with the specified parameters

def get_model(model_type, max_tokens, min_tokens, decoding, temperature, stop_seq):
    generate_params = {
        GenParams.MAX_NEW_TOKENS: max_tokens,
        GenParams.MIN_NEW_TOKENS: min_tokens,
        GenParams.DECODING_METHOD: decoding,
        GenParams.TEMPERATURE: temperature,
        #GenParams.TOP_K: top_k,
        #GenParams.TOP_P: top_p,
        GenParams.STOP_SEQUENCES: stop_seq
    }

    model = Model(
        model_id=model_type,
        params=generate_params,
        credentials={
            "apikey": api_key,
            "url": url
        },
        project_id=watsonx_project_id
    )

    return model


def get_model_test(model_type, max_tokens, min_tokens, decoding, temperature):
    generate_params = {
        GenParams.MAX_NEW_TOKENS: max_tokens,
        GenParams.MIN_NEW_TOKENS: min_tokens,
        GenParams.DECODING_METHOD: decoding,
        GenParams.TEMPERATURE: temperature
    }

    model = Model(
        model_id=model_type,
        params=generate_params,
        credentials={
            "apikey": api_key,
            "url": url
        },
        project_id=watsonx_project_id
    )

    return model


def extract_text_and_chunking(url):
    try:
        headers_to_split_on = [
            ("h1", "Header 1"),
            ("h2", "Header 2"),
            ("h3", "Header 3"),
            ("h4", "Header 4"),
            ("h5", "Header 5"),
            ("h6", "Header 6"),
            ]

        html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        html_header_splits = html_splitter.split_text_from_url(url)
        chunk_size = 500 # 100 = 86 tokens
        chunk_overlap = 20
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n"])
        splits = text_splitter.split_documents(html_header_splits)

        return splits

    except Exception as e:
        print(f"An error occurred: {str(e)}")


def create_embedding(url, collection_name):
    extracted_chunks = extract_text_and_chunking(url)
    embedding_function_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="intfloat/multilingual-e5-large")
    client = chromadb.Client()
    collection = client.get_or_create_collection(collection_name, embedding_function = embedding_function_ef, metadata={"hnsw:space": "cosine"})
    
    passage_prefix = "passage: "

    # Upload text to chroma
    collection.upsert(
        documents=[passage_prefix + i.page_content for i in extracted_chunks],
        metadatas=[i.metadata if hasattr(i, 'metadata') and i.metadata else {"Header": "N/A"} for i in extracted_chunks],
        ids=[str(i) for i in range(1, len(extracted_chunks) + 1)],
    )

    return collection


def create_prompt(url, question, collection_name):
    # Create embeddings for the text file
    collection = create_embedding(url, collection_name)
    
    query_prefix = "query: "

    # query relevant information
    relevant_chunks = collection.query(
        query_texts=[query_prefix + question],
        n_results=5,
        where={"Header 2": {
        '$ne': '脚注[編集]'}
        },
    #where_document={"$contains":"抗生物質"},
    )

    context = "\n\n".join(relevant_chunks['documents'][0])
    

    # Please note that this is a generic format. You can change this format to be specific to llama
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。常に日本語で回答してください。提供されたコンテキスト情報を使用して回答します。十分な情報が提供されていない場合は、「提供された情報に基づいて回答できません」と述べます。"
    final_prompt = "{b_inst} {system}{prompt} {e_inst} ".format(b_inst=B_INST,
                                                            system=f"{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}",prompt=f"コンテキストに基づいて質問に答えてください: \n```\n{context}\n``` \n\n質問： {question} \n\n答え:",e_inst=E_INST)


    return final_prompt


def main():

    # Get the API key and project id and update global variables
    get_credentials()

    # Try diffrent URLs and questions
    url = "https://ja.wikipedia.org/wiki/%E6%8A%97%E7%94%9F%E7%89%A9%E8%B3%AA"

    #question = "抗生物質とは何ですか?"
    question = "ペニシリンは誰が発見したのですか？"
    collection_name = "test_web_RAG"

    answer_questions_from_web(api_key, watsonx_project_id, url, question, collection_name)


def answer_questions_from_web(request_api_key, request_project_id, url, question, collection_name):

    # Retrieve variables for invoking llms
    get_credentials()

    # Update the global variable
    globals()["api_key"] = request_api_key
    globals()["watsonx_project_id"] = request_project_id

    # Specify model parameters
    model_type = "elyza/elyza-japanese-llama-2-7b-instruct"
    max_tokens = 100
    min_tokens = 50
    decoding = DecodingMethods.GREEDY
    temperature = 0.7
    #top_k = 50
    #top_p = 1
    stop_seq = ["。", "\n"]
    
    # Get the watsonx model = try both options
    model = get_model(model_type, max_tokens, min_tokens, decoding, temperature, stop_seq)

    # Get the prompt
    complete_prompt = create_prompt(url, question, collection_name)

    # Let's review the prompt
    print("----------------------------------------------------------------------------------------------------")
    print("*** Prompt:" + complete_prompt + "***")
    print("----------------------------------------------------------------------------------------------------")

    generated_response = model.generate(prompt=complete_prompt)
    response_text = generated_response['results'][0]['generated_text']

    # Remove trailing white spaces
    response_text = response_text.strip()

    # print model response
    print("--------------------------------- Generated response -----------------------------------")
    print(response_text)
    print("*********************************************************************************************")

    return response_text


# Invoke the main function
if __name__ == "__main__":
    main()
