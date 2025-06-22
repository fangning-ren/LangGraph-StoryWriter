from langgraph.prebuilt import create_react_agent
from langchain_ollama import OllamaLLM
from langchain_together import Together, ChatTogether
from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseLLM

from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, MessageGraph, MessagesState, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod

import json

from logging import getLogger
import time
import dotenv
dotenv.load_dotenv()

from .utils import * 
from .prompts import *


class NovelState(TypedDict):
    messages: list
    current_prompt: str
    user_context: str
    revised_user_context: str
    story_elements: str
    chapter_outlines: dict[str, str]
    mega_outline: str

    chapters: dict[str, str]
    current_chapter_index: int
    current_chapter: str
    current_feedback: str
    last_chapter: str
    last_chapter_summary: str
    next_chapter_summary: str

    repeat: bool
    n_repeats: int = 0
    max_repeats: int = 0
    do_outline_generation: bool = True
    n_outline_reviews: int = 0
    max_outline_reviews: int = 0
    outline_complete: bool = False
    n_chapter_reviews: int = 0
    max_chapter_reviews: int = 0

    working_dir: str = "./stories/"
    output_dir: str = "./stories/story.txt"


def initialize_state(state:NovelState) -> NovelState:
    state["current_prompt"] = ""
    state["user_context"] = ""
    state["revised_user_context"] = ""
    state["story_elements"] = ""
    state["chapter_outlines"] = {"all": ""}
    state["chapters"] = {}
    state["current_chapter_index"] = 0
    state["current_chapter"] = ""
    state["current_feedback"] = ""
    state["last_chapter"] = ""

    state["repeat"] = False
    state["n_repeats"] = 0
    state["max_repeats"] = 1
    state["do_outline_generation"] = True
    state["n_outline_reviews"] = 0
    state["max_outline_reviews"] = 0
    state["outline_complete"] = False
    state["n_chapter_reviews"] = 0
    state["max_chapter_reviews"] = 0
    return state



class BasicNovelWriterAgent(ABC):
    def __init__(self, llm:OllamaLLM, 
                 system_prompt: str = WRITER_SYSTEM,
                 prompt_template: str = "",
                 personal_preference_prompt:str = PERSONAL_PREFERENCE_PROMPT,
                 minimum_response_length: int = 800,
                 maximum_response_length: int = 5000,
                 ) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
        self.personal_preference_prompt = personal_preference_prompt
        self.logger = NovelWritingLogger()

    @abstractmethod
    def __call__(self, state: NovelState) -> NovelState:
        raise NotImplementedError("Subclasses must implement __call__ method")

    def _add_personal_preference(self):
        # check if the personal preference prompt is already in the prompt template
        if self.personal_preference_prompt and self.personal_preference_prompt not in self.prompt_template:
            self.prompt_template = self.personal_preference_prompt + "\n" + self.prompt_template
    
    def _call_llm(self, state: NovelState, user_prompt: str, with_history: bool = False, filter_think: bool = True, format_json: bool = False, logfile_suffix = None) -> str:
        """
        Call the LLM
        """
        if with_history:
            state["messages"].append(
                {"role": "system", "content": self.system_prompt}
            )
            state["messages"].append(
                {"role": "user", "content": user_prompt}
            )
            response = self.llm.invoke(
                state["messages"],
            )
        else:
            temp_message_list = []
            temp_message_list.append(
                {"role": "system", "content": self.system_prompt}
            )
            temp_message_list.append(
                {"role": "user", "content": user_prompt}
            )
            response = self.llm.invoke(
                temp_message_list,
            )
        raw_response_len = len(response)
        if filter_think:
            response = remove_think(response)
        if format_json:
            response = strip_any_unnecessary_chars_for_json(response)
            if response.find("{") == -1 and response.find("[") == -1:
                print(response)
                raise ValueError("No valid JSON structure found in the response")
        new_response_len = len(response)
        state["messages"].append(
            {"role": "assistant", "content": response}
        )
        user_input_len = len(user_prompt)
        history_length = sum(
            len(message["content"]) for message in state["messages"]
        )
        history_number = len(state["messages"])
        message = f"User input: {user_input_len:>6d}; LLM response: {raw_response_len:>6d}; Filtered response: {new_response_len:>6d}; History number: {history_number:>6d}; History length: {history_length:>6d}"
        
        if logfile_suffix:
            logname = f"{self.__class__.__name__}-{logfile_suffix}"
        else:
            logname = f"{self.__class__.__name__}" + datetime.datetime.now().strftime("%H-%M-%S")
        self.logger.log(logname, message, user_message=user_prompt, assistant_response=response)
        return response

