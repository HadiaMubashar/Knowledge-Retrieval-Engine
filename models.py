from transformers import pipeline
import wikipediaapi
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel
import os
from dotenv import load_dotenv
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
import torch

# Load environment variables from .env file
load_dotenv()
hf_token = os.getenv('HUGGINGFACE_API_KEY')
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

user_agent = 'MyWikipediaApp/1.0 (myemail@example.com)'
wiki_wiki = wikipediaapi.Wikipedia(user_agent, 'en')

summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device= device)

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', token = hf_token)
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', token = hf_token).to(device)

# FAISS index setup
index = faiss.IndexFlatL2(384)

def fetch_wikipedia_page(link):
    page_name = link.split("/")[-1]
    page = wiki_wiki.page(page_name)
    if not page.exists():
        return None, None
    return page.title, page.text

def chunk_text(text, chunk_size=3500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def summarize_full_text(text, chunk_size=3500):
    text_chunks = chunk_text(text, chunk_size=chunk_size)
    
    summaries = []
    for chunk in text_chunks:
        summary = summarizer(chunk, max_length=30, min_length =10, do_sample=False)
        summaries.append(summary[0]['summary_text'])

    final_summary = " ".join(summaries)
    return final_summary

# embed text
def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(device)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy()

# index content in FAISS
def index_content(content):
    chunks = chunk_text(content)
    vectors = np.array([embed_text(chunk) for chunk in chunks])
    index.add(vectors)
    return chunks

# search content
def search(query, chunks, top_k=5):
    query_vector = embed_text(query)
    D, I = index.search(np.array([query_vector]), top_k)
    return [chunks[i] for i in I[0]]

# Set up LLM 
repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceHub(repo_id=repo_id, 
                     huggingfacehub_api_token=hf_token)

# prompt template 
prompt_template = """Your question: {question}

Answer using the given context: """

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

def generate_answer_llm(query, chunks):
    # Search for relevant chunks based on the query
    retrieved_chunks = search(query, chunks)

    # Combine retrieved chunks for context
    context = ' '.join(retrieved_chunks)

    # Generate answer using prompt
    result = llm(prompt.format(context=context, question=query))

    answer = result.strip() or "Sorry, I don't know."

    last_period_index = answer.rfind('.')
    if last_period_index != -1:
        answer = answer[:last_period_index + 1].strip()  
    return answer 

def process_wikipedia_page(link):
    title, content = fetch_wikipedia_page(link)

    summary = summarize_full_text(content)
    chunks = index_content(content)
    return title, summary, chunks