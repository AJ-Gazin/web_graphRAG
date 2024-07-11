import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from pyvis.network import Network
import spacy
from tqdm import tqdm
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Load the spaCy model for entity recognition
spacy.require_cpu()
nlp = spacy.load("en_core_web_sm")

def process_website(website, api_key):
    print("Downloading website content...")
    response = requests.get(website)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text()

    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_text(text)

    print("Creating LangChain documents...")
    documents = [Document(page_content=chunk) for chunk in docs]

    print("Initializing OpenAI embeddings and VectorStore...")
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vector_db = FAISS.from_documents(documents, embeddings)

    print("Creating Knowledge Graph...")
    knowledge_graph = Network(directed=True)

    # Limit the number of nodes and edges for testing
    max_nodes = 50
    max_edges = 100
    nodes_set = set()
    edges_set = set()

    # First, add nodes
    for doc in tqdm(documents, desc="Adding nodes"):
        doc_nlp = nlp(doc.page_content)
        for ent in doc_nlp.ents:
            if len(nodes_set) >= max_nodes:
                break
            if ent.text not in nodes_set:
                knowledge_graph.add_node(ent.text, label=ent.text)
                nodes_set.add(ent.text)

    # Then, add edges
    for doc in tqdm(documents, desc="Adding edges"):
        doc_nlp = nlp(doc.page_content)
        for i, ent1 in enumerate(doc_nlp.ents):
            for j, ent2 in enumerate(doc_nlp.ents):
                if len(edges_set) >= max_edges:
                    break
                if i != j and (ent1.text, ent2.text) not in edges_set and ent1.text in nodes_set and ent2.text in nodes_set:
                    knowledge_graph.add_edge(ent1.text, ent2.text)
                    edges_set.add((ent1.text, ent2.text))

    print("Saving Knowledge Graph as HTML...")
    knowledge_graph_html = os.path.join(os.getcwd(), "knowledge_graph.html")
    knowledge_graph.write_html(knowledge_graph_html, notebook=False)

    # Log the content of the knowledge graph for debugging
    print(f"Knowledge Graph saved to {knowledge_graph_html}")

    with open(knowledge_graph_html, 'r') as file:
        knowledge_graph_content = file.read()

    return knowledge_graph_content, vector_db

def query_gpt_with_rag(website, api_key, query, vector_db):
    client = OpenAI(api_key=api_key)

    print("Querying the vector database...")
    search_results = vector_db.similarity_search(query)

    print("Generating prompt with retrieved documents...")
    retrieved_docs_text = "\n\n".join([doc.page_content for doc in search_results])
    prompt = f"{query}\n\nUse the following context to answer the query:\n\n{retrieved_docs_text}"

    print("Querying GPT-3.5...")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )

    response_text = response.choices[0].message.content.strip()

    print("Creating response-specific Knowledge Graph...")
    response_graph = Network(directed=True)
    response_text_nlp = nlp(response_text)

    nodes_set = set()
    edges_set = set()

    # First, add nodes
    for ent in response_text_nlp.ents:
        if len(nodes_set) >= 50:
            break
        if ent.text not in nodes_set:
            response_graph.add_node(ent.text, label=ent.text)
            nodes_set.add(ent.text)

    # Then, add edges
    for i, ent1 in enumerate(response_text_nlp.ents):
        for j, ent2 in enumerate(response_text_nlp.ents):
            if len(edges_set) >= 100:
                break
            if i != j and (ent1.text, ent2.text) not in edges_set and ent1.text in nodes_set and ent2.text in nodes_set:
                response_graph.add_edge(ent1.text, ent2.text)
                edges_set.add((ent1.text, ent2.text))

    print("Saving response-specific Knowledge Graph as HTML...")
    response_graph_html = os.path.join(os.getcwd(), "response_graph.html")
    response_graph.write_html(response_graph_html, notebook=False)

    # Log the content of the response-specific knowledge graph for debugging
    print(f"Response-specific Knowledge Graph saved to {response_graph_html}")

    with open(response_graph_html, 'r') as file:
        response_graph_content = file.read()

    return response_text, response_graph_content
