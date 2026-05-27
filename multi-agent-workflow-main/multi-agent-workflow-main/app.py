import numpy as np
import faiss
import transformers import pipeline
from sentence_transformers import SentenceTransformer   


with open('documents.txt', 'r', encoding='utf-8') as f:
    documents = f.read().split("\n")

documents=[doc.strip() for doc in documents if doc.strip()]


#create embeddings
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
document_embeddings = embedding_model.encode(documents)

#build faiss index
dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(document_embeddings)).astype('float32')

#Retrieval agents

def retrieve_documents(query, top_k=2):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding).astype('float32'), top_k)
    retrieved_docs = [documents[i] for i in indices[0]]
    return retrieved_docs

#Hugging Face Agents

#Summarizer agent
summarizer_agent=pipeline(
    task='summarization',
    model='facebook/bart-large-cnn'
)

#QnA Agent
qa_agent=pipeline(
    task='question-answering',
    model='distilbert/distilbert-base-cased-distilled-squad'
)

#Sentiment Analysis Agent
sentiment_agent=pipeline(
    task='text-classification',
    model='distilbert/distilbert-base-uncased-finetuned-sst-2-english'   
)

#text generation
generator_agent=pipeline(
    task='text-generation',
    model='gpt2'
)

#Coordination Agent

def coordinator(query):
    print('User Query': query)

    #retrive doc
    retrieved_docs=retrieve_documents(query)
    context=" ".join(retrieved_docs)

    print('RETRIEVED DOC')
    print(context)

    summary=summarizer_agent(
        context,
        max_length=60,
        min_length=20,
        do_sample=False
    )[0]['summary_text']
    
    qa_result=qa_agent(
        question=query,
        context=context
    )

    sentiment=sentiment_agent(context)
    generated=generator_agent(
        f"Context:{context}\nQuestion: {query}",
        max_new_tokens=50,
        do_samplle=True,
        temperature=0.7
    )[0]['generated_text']

    print('SUMMARY AGENT OUTPUT')
    print(summary)

    print('SENTIMENT AGENT OUTPUT')
    print(sentiment)
    print('QA AGENT OUTPUT')
    print(qa_result)

    print('TEXT GENERATION AGENT OUTPUT')
    print(generated)

query=input('Enter your question: ')
coordinator(query)