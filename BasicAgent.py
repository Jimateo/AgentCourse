import os
import gradio as gr
import time

from typing import Optional

from agent import workflow
from api_GAIA import ApiClienteGAIA


class BasicAgent:
    def __init__(self):
        print("BasicAgent initialized.")
        self.agent = workflow
        
    async def __call__(self, question: str, task_file_path: Optional[str] = None) -> str:
        print(f"Agent received question (first 50 chars): {question[:50]}...")
        
        try:
            if task_file_path:
                try:
                    with open(task_file_path, 'r') as f:
                        file_content = f.read()
                    # Determine file type from extension
                    file_ext = os.path.splitext(task_file_path)[1].lower()
                    context = f"""
                                Question: {question}
                                This question has an associated file. Here is the file content:
                                ```{file_ext}
                                {file_content}
                                ```
                                Analyze the file content above to answer the question.
                                """ 
                    question = context
                except Exception as e:
                    print(f"Error {e}")

            answer = await self.agent.run(question)
            fixed_answer = str(answer).removeprefix("FINAL ANSWER:").removeprefix("FINAL ANSWER :").strip()
            print(f"Agent returning fixed answer: {str(fixed_answer)}")
            time.sleep(35)
            return fixed_answer
        
        except Exception as e:

            print(f"Error in agent execution: {str(e)}")
            return f"Error: {str(e)}"
        
