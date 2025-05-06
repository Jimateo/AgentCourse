import os
import pandas as pd
import time
import nest_asyncio
import asyncio
import re
import tempfile

from youtube_transcript_api import YouTubeTranscriptApi
from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec
from llama_index.core.agent.workflow import ReActAgent, FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.readers.youtube_transcript import YoutubeTranscriptReader
from llama_index.readers.whisper import WhisperReader

from llama_index.core.memory import ChatMemoryBuffer
from dotenv import load_dotenv
from llama_index.llms.gemini import Gemini


nest_asyncio.apply()
load_dotenv()

# (Keep Constants as is)
# --- Constants ---
DEFAULT_API_URL = os.getenv("DEFAULT_API_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HF_KEY = os.getenv("HF_KEY")

memory = ChatMemoryBuffer.from_defaults()

SYS_PROMPT = "You are a helpful assistant tasked with answering questions using a set of tools. Now, I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.Your answer should only start with " + "FINAL ANSWER: " + ", then follows with the answer. If the answer is a number pls only respost with a number and not with a word"

# --- LLM ---
llm_model = Gemini(model_name="models/gemini-2.5-flash-preview-04-17")


# --- TOOLS ---
tool_search = DuckDuckGoSearchToolSpec().to_tool_list()

# --- Nueva Tool para YouTube ---

def load_video_transcript(video_link: str) -> str:
            try:
                loader = YoutubeTranscriptReader()
                documents = loader.load_data(
                    ytlinks=[video_link]
                )

                text = documents[0].text_resource.text

                return { "video_transcript": text }
            except Exception as e:
                print("error", e)





load_video_transcript_tool = FunctionTool.from_defaults(
    load_video_transcript,
    name="load_video_transcript",
    description="Loads transcript of the given video using the link.",
)

agent = FunctionAgent(
    tools= tool_search + [load_video_transcript_tool],
    llm = llm_model,
    system_prompt= SYS_PROMPT,
)

async def main():
    # Run the agent
    response = await agent.run("Examine the video at https://www.youtube.com/watch?v=1htKBjuUWec.What does Teal'c say in response to the question ""Isn't that hot?""")
    print(str(response))

# Run the agent
if __name__ == "__main__":
    asyncio.run(main())