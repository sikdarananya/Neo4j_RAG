#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('pip', 'install llama-index-llms-openai')
get_ipython().run_line_magic('pip', 'install llama-index-graph-stores-neo4j')
get_ipython().run_line_magic('pip', 'install llama-index-embeddings-openai')
get_ipython().run_line_magic('pip', 'install llama-index-llms-azure-openai')


# In[6]:


get_ipython().run_line_magic('pip', 'install llama-index-embeddings-azure-openai')
get_ipython().run_line_magic('pip', 'install llama-index-llms-azure-openai')


# In[12]:


pip install llama-index-core


# In[17]:


import os
import openai
from langchain_community.graphs import Neo4jGraph
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain.callbacks.base import BaseCallbackHandler
from threading import Thread
from queue import Queue, Empty
from collections.abc import Generator
from sse_starlette.sse import EventSourceResponse
import json
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.llms.openai import OpenAI
from IPython.display import Markdown, display
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import (
      VectorStoreIndex,
      SimpleDirectoryReader,
      KnowledgeGraphIndex,
)

from llama_index.core import ServiceContext

import logging
import sys

logging.basicConfig(
stream = sys.stdout, level=logging.INFO)

logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# In[22]:


api_key = "3289261e6cc84fa8aef58d38e2264fa9"
openai.api_key = api_key
openai.api_base = 'https://openai-demo-mb-001.openai.azure.com/'
openai.api_type = 'azure'
openai.api_version = '2023-05-15'
deployment_name = 'openaidemomb001'
deployment_name_embeddings = 'openaidemomb002'

os.environ["OPENAI_API_TYPE"] = openai.api_type
os.environ["OPENAI_API_VERSION"] = openai.api_version
os.environ["OPENAI_OPENAI_BASE"] = openai.api_base
os.environ["OPENAI_API_KEY"] = "3289261e6cc84fa8aef58d38e2264fa9"


# In[23]:


llm = AzureOpenAI(
    deployment_name = deployment_name,
    model_name = deployment_name,
    cache = False,
    api_key = api_key,
    azure_endpoint = openai.api_base,
    temperature = 0.1)


# In[24]:


embedding_llm = AzureOpenAIEmbedding(
     deployment_name = deployment_name_embeddings,
     model_name = deployment_name_embeddings,
     api_key = openai.api_key,
     api_base = openai.api_base,
     api_type = openai.api_type,
     api_version = openai.api_version,
     azure_endpoint = openai.api_base
)


# In[25]:


service_context = ServiceContext.from_defaults(
llm=llm,
embed_model = embedding_llm,
)


# In[26]:


username = "neo4j"
password = "AyN7ybIusHSm8PKpU1K2N80Gsfu1tytd3HtRMPxFKDg"
url = "neo4j+s://dc9b32df.databases.neo4j.io"
embed_dim = 1536
database = "neo4j"


# In[27]:


space_name = "llamaindex"
edge_types, rel_prop_names = ["relationship"], [
    "relationship"
]
tags = ["entity"]


# In[28]:


graph_store = Neo4jGraphStore(
     username = username,
     password = password,
     url = url,
     database = database,
     space_name = space_name,
     edge_types = edge_types,
     rel_prop_names = rel_prop_names,
     tags = tags,
)


# In[29]:


storage_context = StorageContext.from_defaults(graph_store = graph_store)


# In[30]:


from llama_index.core import SimpleDirectoryReader

reader = SimpleDirectoryReader(
     input_files = ["data.txt"]
)

documents = reader.load_data()


# In[32]:


index = KnowledgeGraphIndex.from_documents(
     documents,
     storage_context = storage_context,
     show_progress = True,
     service_context = service_context,
     max_triplets_per_chunk = 10,
     space_name = space_name,
     edge_types = edge_types,
     rel_prop_names = rel_prop_names,
     tags = tags,
     include_embeddings = True,
)


# In[33]:


query_engine = index.as_query_engine(include_text = False, response_mode = "tree_summarize", storage_context = storage_context,
                                    service_context = service_context, verbose = True, embedding_mode = 'hybrid', similarity_top_k = 10)


# In[34]:


query_engine.query("Tell me about Aarush and Ananya?")


# In[ ]:




