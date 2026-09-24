import streamlit as st
import pandas as pd 
import os
from langchain_community.document_loaders.pdf import PyPDFLoader
from sentence_transformers import SentenceTransformer
import chromadb
import uuid
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
load_dotenv()
api_key=os.getenv("GROQ_API_KEY")
from langchain_groq import ChatGroq

st.header("Quick Answer RAG")
pdf=st.file_uploader("Upload a PDF",type=["pdf"])
if pdf:
    st.success("pdf uploaded successfully")
    def load_all_pdfs():
        num_docs=0
        all_docs=[]
        with open("temp.pdf","wb") as f:
            f.write(pdf.getbuffer())
        loader=PyPDFLoader("temp.pdf")
        doc=loader.load()
        all_docs.extend(doc)
        num_docs+=1
        return all_docs  
    all_pdf_documents=load_all_pdfs()
else:
    st.write("pdf no found")
    st.stop()
     
def split_doc(documents,chunk_size=500,chunk_overlap=50):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunked_docs=text_splitter.split_documents(documents)
    return chunked_docs
chunks=split_doc(all_pdf_documents)

class EmbeddingManager:
    def __init__(self, model_name="all-MiniLM-L6-v2"): #loading embedding model
        self.model_name=model_name
        self.model=SentenceTransformer(self.model_name)
      

    def generate_embeddings(self,text): #generate embedding
        embeddings=self.model.encode(text,show_progress_bar=True)
        return embeddings
embedding_manager=EmbeddingManager()

class vectorstoreManager:
    def __init__(self,persist_directory="data/vector_store",collection_name="pdf_documents"):
        self.collection_name=collection_name
        self.persist_directory=persist_directory
        self.collection=None
        self.client=None
        self._initialize_store()

    def _initialize_store(self):
        # creating directory 
        os.makedirs(self.persist_directory,exist_ok=True)

        # create a client
        self.client=chromadb.PersistentClient(path=self.persist_directory)

        # create the collection
        self.collection=self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description":"vector store collection for pdf embeddings in RAG"} 
        )
     
    def add_documents(self, documents, embeddings): # here documents=chunks
        if len(documents) != len(embeddings):
            raise ValueError("num of documents does not match num of embeddings")


        # store => ids, embedding, document, metadata
        ids = []
        all_metadata = []
        documents_content = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4()}"
            ids.append(doc_id)

            metadata = dict(doc.metadata)
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)
            all_metadata.append(metadata)

            documents_content.append(doc.page_content)

            embeddings_list.append(embedding.tolist())

            self.collection.add(
                ids=ids,
                metadatas=all_metadata,
                documents=documents_content,
                embeddings=embeddings_list
            )


vector_store=vectorstoreManager()
texts=[doc.page_content for doc in chunks]
embedding=embedding_manager.generate_embeddings(texts)
vector_store.add_documents(chunks,embedding)

class RAGRetriever:
    def __init__(self,embedding_manager,vector_store):
        self.embedding_manager=embedding_manager
        self.vector_store=vector_store
        
    def retrieve(self,query,top_k=5,score_threshold=0.0): #here cosine similarity min value that should be consider called score_threshold
        
        # query =>embedding
        query_embeddings=self.embedding_manager.generate_embeddings([query])[0]
       
        # semantic search
        results=self.vector_store.collection.query(
            query_embeddings=[query_embeddings.tolist()],
            n_results=top_k
        )
        
        # cosine similarity
        retrieved_docs=[]
        if results["documents"] and results["documents"][0]:
            ids=results["ids"][0]
            metadatas=results["metadatas"][0]
            documents=results["documents"][0]
            distances=results["distances"][0]

            for i,(doc_id,metadata,document,distance) in enumerate(zip(ids,metadatas,documents,distances)):
                similarity_score=1-distance

                if similarity_score>=score_threshold:
                    retrieved_docs.append({
                        "id":doc_id,
                        "document":document,
                        "metadata":metadata,
                        "distance":distance,
                        "similarity_score":similarity_score,
                        "rank":i+1
                    })
                print(f"retrieved {len(retrieved_docs)} documents")
        else:
            print("no document found")
        return retrieved_docs
rag_retriever=RAGRetriever(embedding_manager,vector_store)

llm=ChatGroq(
    groq_api_key=api_key,
    model="openai/gpt-oss-120b",
    temperature=0.1,
    max_tokens=1024 
)
def generate_output(query,rag_retriever,llm,top_k=3):
    results=rag_retriever.retrieve(query,top_k)
    context="\n".join([doc["document"] for doc in  results]) if results else""

    if not context:
        print("we found no relevant context for the given query")
    #context + query
    prompt=f""" use given context to generate the answer for the query
                context:{context}
                Query:{query}"""
    response=llm.invoke([prompt.format(context=context,query=query)]) 
    return response.content

question=st.text_input("Enter your question")
submit=st.button("Ask")
if submit:
    answer=generate_output(question,rag_retriever,llm)
    st.write(answer)
    

