# --- Imports ---
# Standard library
import os
import asyncio


# Third-party libraries
import pandas as pd
from dotenv import load_dotenv


# LlamaIndex imports
from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec
from llama_index.readers.wikipedia import WikipediaReader 
from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow
from llama_index.core.tools import FunctionTool
from llama_index.readers.youtube_transcript import YoutubeTranscriptReader
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.schema import Document
from llama_index.core.node_parser import SentenceSplitter

# --- Environment Setup ---
load_dotenv()

# --- Constants ---
DEFAULT_API_URL = os.getenv("DEFAULT_API_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HF_KEY = os.getenv("HF_KEY")

# --- System Prompt ---
SYS_PROMPT = """You are a concise assistant. Your ONLY task is to provide the exact answer to the question.
IMPORTANT RULES:
0. If you are given data on a date or between two dates you have to be very precise
1. Your response MUST start with "FINAL ANSWER: "
2. After "FINAL ANSWER: " you MUST ONLY provide:
   - A single number (no words, no units, no commas)
   - OR a single word/phrase (no articles, no explanations)
   - OR a comma-separated list of numbers/words (no additional text)
3. DO NOT add any explanations, thoughts, or additional text
4. DO NOT use articles or abbreviations
5. DO NOT use units unless specifically requested

Example correct responses:
FINAL ANSWER: 42
FINAL ANSWER: Barcelona
FINAL ANSWER: 1,2,3,4,5
"""

# --- LLM Configuration ---
llm_model = GoogleGenAI(
    model_name="models/gemini-2.0-flash",
    api_key=GOOGLE_API_KEY,
    temperature=0.1,
)

# --- Tool Definitions ---
# File Operations Tools



def wikipedia_embed_retrieval(topic: str) -> dict:
    """
    Retrieves relevant Wikipedia chunks using Gemini embeddings.
    """
    try:
        loader = WikipediaReader()
        documents = loader.load_data(pages=[topic])

        if not documents:
            return {"error": "No Wikipedia article found"}

        # Usa Gemini para los embeddings
        Settings.embed_model = GoogleGenAIEmbedding(
            model_name="models/embedding-001",
            api_key=GOOGLE_API_KEY
        )

        # Configurar el splitter de oraciones
        text_splitter = SentenceSplitter(
            chunk_size=512,
            chunk_overlap=50
        )

        # Indexado y recuperación con chunks más pequeños para mejor precisión
        index = VectorStoreIndex.from_documents(
            documents,
            transformations=[text_splitter]
        )
        retriever = index.as_retriever(similarity_top_k=3)
        nodes = retriever.retrieve(topic)

        # Combina los chunks más relevantes
        relevant_chunks = "\n\n".join([n.text for n in nodes])
        return {"wikipedia_chunks": relevant_chunks}

    except Exception as e:
        return {"error": str(e)}

    



def load_video_transcript(video_link: str) -> str:
    """Loads transcript of the given video using the link."""
    try:
        loader = YoutubeTranscriptReader()
        documents = loader.load_data(ytlinks=[video_link])
        return {"video_transcript": documents[0].text_resource.text}
    except Exception as e:
        print("error", e)

# --- Tool Initialization ---
load_video_transcript_tool = FunctionTool.from_defaults(
    load_video_transcript,
    name="load_video_transcript",
    description=(
        "Given a YouTube video URL, fetch and return the full transcript text "
        "using the YouTube Transcript API."
    )
)

wiki_retriever_tool = FunctionTool.from_defaults(
    fn=wikipedia_embed_retrieval,
    name="wikipedia_embed_retrieval",
    description="Given a topic, retrieve the most relevant Wikipedia text chunks using embeddings"
)

# Search Tools
tool_search_duckduckgo = DuckDuckGoSearchToolSpec().to_tool_list()

# --- Agent Definitions ---
agent = FunctionAgent(
    name="Search Tools",
    description=(
        "Performs web searches using DuckDuckGo, Wikipedia and retrieves YouTube video "
        "transcripts to deliver comprehensive and accurate responses."
    ),
    tools=tool_search_duckduckgo+[load_video_transcript_tool,wiki_retriever_tool],
    llm=llm_model,
    system_prompt=SYS_PROMPT,

)

# --- Workflow Setup ---
workflow = AgentWorkflow(
    agents=[agent],
    root_agent="Search Tools",
)

# --- Main Execution ---
async def main():
    response = await workflow.run('How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)? You can use the latest 2022 version of english wikipedia.')
    print(str(response))

if __name__ == "__main__":
    asyncio.run(main())