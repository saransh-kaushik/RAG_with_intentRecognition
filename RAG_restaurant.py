#importing docs
from llama_index.core import (
    VectorStoreIndex,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
    SimpleDirectoryReader,
)
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.memory import ChatMemoryBuffer
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
import warnings
warnings.filterwarnings("ignore")

#initializing memory
memory = ChatMemoryBuffer.from_defaults(token_limit=2000)

def RAG(user_input, memory):  # defining RAG function
    """
    Retrieves and generates a response for the user's query using RAG.
    
    Parameters:
    - user_input (str): The query input from the user.

    Returns:
    - str: The generated response from the chatbot.
    """
    # Load and prepare documents
    reader = SimpleDirectoryReader(input_files=["./your_files_here"])
    documents = reader.load_data()
    
    # Chunking
    text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=128)
    nodes = text_splitter.get_nodes_from_documents(documents, show_progress=True)
    
    # Embedding and LLM setup
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = Groq(model="llama-3.2-3b-preview", api_key="GROQ_API_KEY")
    service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm)
    
    # Create Vector Store Index
    vector_index = VectorStoreIndex.from_documents(documents, show_progress=True, service_context=service_context)
    
    # Set up memory and chat engine
    chat_engine = vector_index.as_chat_engine(
        chat_mode="context",
        memory=memory,
        system_prompt=(
            "Your prompt "
        ),
    )
    
    # Get the response from the chat engine
    response = chat_engine.chat(user_input)
    return response

def load_trained_model(): ##load model
    # Load the fine-tuned model and tokenizer
    model = DistilBertForSequenceClassification.from_pretrained("./intent_model")
    tokenizer = DistilBertTokenizer.from_pretrained("./intent_model")
    return model, tokenizer

def predict_intent(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(probs).item()
    intents = ["Menu Inquiry", "Order Request", "Pricing Inquiry"]
    print(f"Intent detected: {intents[predicted_class]} - Confidence: {probs[0][predicted_class]:.2f}")
    return predicted_class

def RAG_intent(user_input, model, tokenizer): 
    intent = predict_intent(user_input, model, tokenizer)
    if intent in [0, 1, 2]:             # Only activates RAG for menu inquiries, order requests, or pricing inquiries
        print("RAG processing required.")
        return True
    else:
        print("RAG not required.")
        return False

# Load trained model and tokenizer
model, tokenizer = load_trained_model()

# Sample usage

query = input("Enter your query: ")
if RAG_intent(query, model, tokenizer):
    print("Calling RAG function...")
    response = RAG(query, memory)
    print(response)
    while query != "exit":
        query = input("Enter your query: ")
        if query != "exit":
            response = RAG(query, memory)
            print(response)
        
## this is just for sample usage, to reset chat use: chat_engine.reset()





else:
    print("No RAG required. redierecting to AI...")
