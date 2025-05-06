# --- Imports ---
# Standard library
import os
import asyncio
import re
import requests

# Third-party libraries
import pandas as pd
from dotenv import load_dotenv
from markdownify import markdownify
from youtube_transcript_api import YouTubeTranscriptApi

# LlamaIndex imports
from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec
from llama_index.tools.wikipedia import WikipediaToolSpec
from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow
from llama_index.core.tools import FunctionTool
from llama_index.readers.youtube_transcript import YoutubeTranscriptReader
from llama_index.readers.whisper import WhisperReader
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.google_genai import GoogleGenAI

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
    max_tokens=500
)

# --- Tool Definitions ---
# File Operations Tools

def visit_webpage(url: str) -> str:
    """Visits a webpage at the given URL and returns its content as a markdown string."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        markdown_content = markdownify(response.text).strip()
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)
        return ' '.join(markdown_content.split()[:5000])
    except requests.RequestException as e:
        return f"Error fetching the webpage: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def load_video_transcript(video_link: str) -> str:
    """Loads transcript of the given video using the link."""
    try:
        loader = YoutubeTranscriptReader()
        documents = loader.load_data(ytlinks=[video_link])
        return {"video_transcript": documents[0].text_resource.text}
    except Exception as e:
        print("error", e)

# --- Tool Initialization ---
visit_webpage_tool = FunctionTool.from_defaults(
    visit_webpage,
    name="visit_webpage",
    description="Visits a webpage and returns its content as markdown."
)

load_video_transcript_tool = FunctionTool.from_defaults(
    load_video_transcript,
    name="load_video_transcript",
    description="Loads transcript of the given video using the link."
)

# Search Tools
tool_search_wikipedia = WikipediaToolSpec().to_tool_list()
tool_search_duckduckgo = DuckDuckGoSearchToolSpec().to_tool_list()

# --- Agent Definitions ---
agent = FunctionAgent(
    name="Search Tools",
    description="Search information on internet and wikipedia",
    tools=tool_search_duckduckgo + [visit_webpage_tool],
    llm=llm_model,
    system_prompt=SYS_PROMPT,
)

agent_youtube = FunctionAgent(
    name="Analyze youtube video",
    description="Analyze the transcription from a youtube video",
    tools=[load_video_transcript_tool],
    llm=llm_model,
    system_prompt=SYS_PROMPT,
)

# --- Workflow Setup ---
workflow = AgentWorkflow(
    agents=[agent, agent_youtube],
    root_agent="Search Tools",
)

# --- Main Execution ---
async def main():
    response = await workflow.run("How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)? You can use the latest 2022 version of english wikipedia.")
    print(str(response))

if __name__ == "__main__":
    asyncio.run(main())