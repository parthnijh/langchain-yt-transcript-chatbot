from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate



import os
from dotenv import load_dotenv
load_dotenv()
#1 Indexing
video_id="LPZh9BOjkQs"

transcript_list=YouTubeTranscriptApi().fetch(video_id=video_id,languages=["en"])
transcript_text=""
for item in transcript_list:
    transcript_text=transcript_text+ item.text +" "
splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)
splitted=splitter.create_documents([transcript_text])

embeddings=HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore=Chroma(
    persist_directory="chroma",
    collection_name="sample-yt",
    embedding_function=embeddings
)
vectorstore.add_documents(splitted)

#2 retriever

retriever=vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":4})


#3 augmentation
prompt=PromptTemplate(template='''You are a helpful assistant
                      Answer only from the context,if context is insufficent ,
                      just say you dont know
                      this is the {context},
                      this is the question : {question}''',input_variables=["context","question"])

question="is the topic of aliens discussed in this video? if yes what was discussed?"
context=retriever.invoke(question)
passable_context=''''''
for item in context:
    passable_context+=item.page_content
model=ChatGoogleGenerativeAI(model="gemini-2.5-flash",api_key=os.getenv("GOOGLE_API_KEY"))

chain=prompt | model
result=chain.invoke({"question":question,"context":passable_context})
print(result)