from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda
from langchain_core.output_parsers import StrOutputParser


import os
from dotenv import load_dotenv
load_dotenv()
#1 Indexing
video_id="LPZh9BOjkQs"
parser=StrOutputParser()
transcript_list=YouTubeTranscriptApi().fetch(video_id=video_id,languages=["en"])
transcript_text=""
for item in transcript_list:
    transcript_text=transcript_text+ item.text +" "

splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=300
)
splitted=splitter.create_documents([transcript_text])

embeddings=HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore=Chroma(
    persist_directory="chroma",
    collection_name="sample-yt",
    embedding_function=embeddings,
    # distance_metric="cos"
)
vectorstore.add_documents(splitted)

#2 retriever

retriever=vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":6})


#3 augmentation
prompt=PromptTemplate(template='''You are an AI assistant. Answer the question  using the provided context.
If the answer is not in the context, respond with "I don't know."

CONTEXT:
{context}

QUESTION:
{question}

Answer:''',input_variables=["context","question"])

# question="What a large language model (LLM) is"
# context=retriever.invoke(question)
# passable_context=''''''

def format(context):
    passable_context=''''''
    for item in context:
        passable_context+=item.page_content
    return passable_context

pal_chain=RunnableParallel({
    "context":retriever | RunnableLambda(format),
    "question":RunnablePassthrough()
})
# print(pal_chain.invoke("what is attention"))
# inputs = pal_chain.invoke("what is attention?")
# print("Inputs:", inputs)

# filled_prompt = prompt.format(**inputs)
# print("\n--- FINAL PROMPT ---\n")
# print(filled_prompt)
# print("\n--------------------\n")


model=ChatGoogleGenerativeAI(model="gemini-2.5-flash",api_key=os.getenv("GOOGLE_API_KEY"))
# result = model.invoke(filled_prompt)
# print("Model output:", result)

chain=pal_chain | prompt | model | parser
# chain=prompt | model
# filled_prompt = prompt.format(context=passable_context, question=question)
# print(filled_prompt)

result=chain.invoke("What a large language model (LLM) is")
print(result)