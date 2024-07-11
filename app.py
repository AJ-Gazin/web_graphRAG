import gradio as gr
from logic import process_website, query_gpt_with_rag

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def generate_knowledge_graph(website, api_key):
    status = "Downloading website content..."
    yield status, "", None
    knowledge_graph_content, vector_db = process_website(website, api_key)
    status = "Knowledge Graph generated."
    yield status, gr.HTML(value=knowledge_graph_content), vector_db

def query_gpt(website, api_key, query, vector_db):
    status = "Querying GPT-3.5..."
    yield status, "", None
    response, response_graph_content = query_gpt_with_rag(website, api_key, query, vector_db)
    status = "Response received."
    yield status, response, gr.HTML(value=response_graph_content)

with gr.Blocks() as demo:
    gr.Markdown("# RAG with Knowledge Graph Demo")

    with gr.Row():
        website = gr.Textbox(label="Website URL", placeholder="Enter the website URL")
        api_key = gr.Textbox(label="OpenAI API Key", placeholder="Enter your OpenAI API key", type="password")

    with gr.Row():
        generate_btn = gr.Button("Generate Knowledge Graph")

    status_output = gr.Textbox(label="Status", value="", interactive=False)
    knowledge_graph_output = gr.HTML()
    vector_db_output = gr.State()

    generate_btn.click(generate_knowledge_graph, inputs=[website, api_key], outputs=[status_output, knowledge_graph_output, vector_db_output])

    with gr.Row():
        query = gr.Textbox(label="Query", placeholder="Enter your query")
        query_btn = gr.Button("Send Query")

    response_output = gr.Textbox(label="GPT-3.5 Response", interactive=False)
    response_graph_output = gr.HTML()

    query_btn.click(query_gpt, inputs=[website, api_key, query, vector_db_output], outputs=[status_output, response_output, response_graph_output])

demo.launch()
