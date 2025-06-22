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


from .prompts import *
from .utils import *
from .states import *

class GetImportantBaseInfoAgent(BasicNovelWriterAgent):
    def __init__(self, llm: OllamaLLM, 
                 system_prompt: str = WRITER_SYSTEM,
                 prompt_template: str = GET_IMPORTANT_BASE_INFO_PROMPT,
                 personal_preference_prompt: str = PERSONAL_PREFERENCE_PROMPT,
                 ) -> None:
        super().__init__(llm, system_prompt, prompt_template, personal_preference_prompt)

    def __call__(self, state: NovelState) -> NovelState:
        if not state.get("user_context"):
            state["user_context"] = state["messages"][-1]["content"]
        user_prompt = self.prompt_template.format(_Prompt=state["user_context"])
        response = self._call_llm(state, user_prompt)
        state["revised_user_context"] = state["user_context"] + "\n" + response
        state["user_context"] = state["revised_user_context"]
        return state
    
    

class GenerateStoryElementAgent(BasicNovelWriterAgent):
    def __init__(self, llm: OllamaLLM, 
                 system_prompt: str = WRITER_SYSTEM,
                 prompt_template: str = GENERATE_STORY_ELEMENTS_PROMPT,
                 personal_preference_prompt: str = PERSONAL_PREFERENCE_PROMPT,
                 ) -> None:
        super().__init__(llm, system_prompt, prompt_template, personal_preference_prompt)

    def __call__(self, state: NovelState) -> NovelState:
        self._add_personal_preference()
        user_prompt = self.prompt_template.format(
            _Prompt=state["revised_user_context"]
        )
        response = self._call_llm(state, user_prompt)
        state["story_elements"] = response
        log_response(user_prompt, response, f"{self.__class__.__name__}.md")
        return state

class WhetherWriteOutlineAgent(BasicNovelWriterAgent):
    def __init__(self, llm: OllamaLLM, 
                 system_prompt: str = WRITER_SYSTEM,
                 prompt_template: str = WHETHER_WRITE_OUTLINE_PROMPT,
                 personal_preference_prompt: str = PERSONAL_PREFERENCE_PROMPT,
                 ) -> None:
        super().__init__(llm, system_prompt, prompt_template, personal_preference_prompt)

    def __call__(self, state: NovelState) -> NovelState:
        self._add_personal_preference()
        user_prompt = self.prompt_template.format(
            _Prompt=state["revised_user_context"],
            _StoryElements=state["story_elements"],
        )
        response = self._call_llm(state, user_prompt, with_history=False, format_json=True)
        response_lines = response.split("\n")
        response_line = [line for line in response_lines if line.find("IsOutline") != -1][0].lower()
        if "true" in response_line:
            state["do_outline_generation"] = False
        elif "false" in response_line:
            state["do_outline_generation"] = True
        else:
            raise ValueError("No valid JSON structure found in the response: " + response)
        return state
    
def check_whether_write_outline(state: NovelState) -> bool:
    """
    Check if the user wants to write an outline.
    """
    if state["do_outline_generation"]:
        return True
    else:
        print("User's context is already a complete outline.")
        return False

# if user already has an outline, then extract the outline from the user context
class CleanOutlineAgent(BasicNovelWriterAgent):
    def __init__(self, llm: OllamaLLM, 
                 system_prompt: str = WRITER_SYSTEM,
                 prompt_template: str = CLEAN_OUTLINE_PROMPT,
                 personal_preference_prompt: str = PERSONAL_PREFERENCE_PROMPT,
                 ) -> None:
        super().__init__(llm, system_prompt, prompt_template, personal_preference_prompt)

    def __call__(self, state: NovelState) -> NovelState:
        self._add_personal_preference()
        user_prompt = self.prompt_template.format(
            _Prompt=state["revised_user_context"],
            _StoryElements=state["story_elements"],
            _Outline=state["chapter_outlines"]["all"],
        )
        response = self._call_llm(state, user_prompt)
        state["chapter_outlines"]["all"] = response
        return state
    
class CleanUserPromptAgent(BasicNovelWriterAgent):
    def __init__(self, llm: OllamaLLM, 
                 system_prompt: str = WRITER_SYSTEM,
                 prompt_template: str = CLEAN_USER_PROMPT_PROMPT,
                 personal_preference_prompt: str = PERSONAL_PREFERENCE_PROMPT,
                 ) -> None:
        super().__init__(llm, system_prompt, prompt_template, personal_preference_prompt)

    def __call__(self, state: NovelState) -> NovelState:
        self._add_personal_preference()
        user_prompt = self.prompt_template.format(
            _Prompt=state["revised_user_context"],
            _StoryElements=state["story_elements"],
            _Outline=state["chapter_outlines"]["all"],
        )
        response = self._call_llm(state, user_prompt)
        state["revised_user_context"] = response
        return state


def clean_outline(state: NovelState) -> NovelState:
    """
    Clean the outline from the user context.
    """
    # if the user has an outline, then clean the outline from the user context
    state["chapter_outlines"]["all"] = state["revised_user_context"]
    return state

# if the user does not have an outline, then generate an initial outline
class InitialOutlineAgent(BasicNovelWriterAgent):
    def __init__(self, llm: OllamaLLM, 
                 system_prompt: str = WRITER_SYSTEM,
                 prompt_template: str = INITIAL_OUTLINE_PROMPT,
                 personal_preference_prompt: str = PERSONAL_PREFERENCE_PROMPT,
                 ) -> None:
        super().__init__(llm, system_prompt, prompt_template, personal_preference_prompt)

    def __call__(self, state: NovelState) -> NovelState:
        self._add_personal_preference()
        user_prompt = self.prompt_template.format(
            _Prompt=state["revised_user_context"],
            _StoryElements=state["story_elements"],
        )
        response = self._call_llm(state, user_prompt)
        state["chapter_outlines"] = {"all": response}
        state["n_repeats"] = 0
        return state
   
def check_output_length(state: NovelState, minoutput: int = 100) -> bool:
    """
    Check if the output length is greater than the minimum output length.
    """
    prompt = INSUFFICIENT_LENGTH_PROMPT.format(_Length = minoutput)
    if len(state["messages"][-1]["content"]) < minoutput and state["n_repeats"] < state["max_repeats"]:
        state["messages"].append(
            {"role": "user", "content": prompt}
        )
        return False
    elif state["n_repeats"] >= state["max_repeats"]:
        print("Output length is sufficient, but maximum repeats reached.")
    state["n_repeats"] = 0
    return True




class CritiqueOutlineGeneralAgent(BasicNovelWriterAgent):
    def __init__(self, llm:OllamaLLM, 
                 system_prompt: str = CRITIQUE_SYSTEM,
                 prompt_template: str = CRITIC_OUTLINE_GENERAL_PROMPT,
                 personal_preference_prompt:str = PERSONAL_PREFERENCE_PROMPT,
                 ) -> None:
        super().__init__(llm, system_prompt, prompt_template, personal_preference_prompt)

    def __call__(self, state: NovelState) -> NovelState:
        if self.personal_preference_prompt:
            self.prompt_template = self.personal_preference_prompt + "\n" + self.prompt_template
        user_prompt = self.prompt_template.format(
            _Outline = state["chapter_outlines"]["all"],
            _Prompt = state["revised_user_context"],
        )

        response = self._call_llm(state, user_prompt)
        state["current_feedback"] = response
        return state
    
class CritiqueOutlineCompleteAgent(BasicNovelWriterAgent):
    def __init__(self, llm:OllamaLLM, 
                 system_prompt: str = CRITIQUE_SYSTEM,
                 prompt_template: str = CRITIC_OUTLINE_COMPLETE_PROMPT,
                 personal_preference_prompt:str = PERSONAL_PREFERENCE_PROMPT,
                 ) -> None:
        super().__init__(llm, system_prompt, prompt_template, personal_preference_prompt)

    def __call__(self, state: NovelState) -> NovelState:
        self._add_personal_preference()
        user_prompt = self.prompt_template.format(
            _Prompt = state["revised_user_context"],
            _Outline=state["chapter_outlines"]["all"],
        )
        response = self._call_llm(state, user_prompt, format_json=True)

        response_lines = response.split("\n")
        response_line = [line for line in response_lines if line.find("IsComplete") != -1][0].lower()
        if "true" in response_line:
            state["outline_complete"] = True
        elif "false" in response_line:
            state["outline_complete"] = False
        else:
            raise ValueError("No valid JSON structure found in the response: " + response)
        return state

def check_outline_review_number(state: NovelState) -> bool:
    """
    Check if the number of outline reviews is less than the maximum number of outline reviews.
    """
    if state["n_outline_reviews"] >= state["max_outline_reviews"]:
        state["n_outline_reviews"] = 0
        return True
    return False

class OutlineRevisionAgent(BasicNovelWriterAgent):
    def __init__(self, llm: OllamaLLM, 
                 system_prompt: str = WRITER_SYSTEM,
                 prompt_template: str = OUTLINE_REVISION_PROMPT,
                 personal_preference_prompt: str = PERSONAL_PREFERENCE_PROMPT,
                 ) -> None:
        super().__init__(llm, system_prompt, prompt_template, personal_preference_prompt)

    def __call__(self, state: NovelState) -> NovelState:
        self._add_personal_preference()
        user_prompt = self.prompt_template.format(
            _Prompt=state["revised_user_context"],
            _Outline=state["chapter_outlines"]["all"],
            _Feedback=state["current_feedback"],
        )
        response = self._call_llm(state, user_prompt)
        state["chapter_outlines"]["all"] = response
        state["n_outline_reviews"] += 1
        return state
    
class CheckOutlineComplete(BaseModel):
    """
    Check if the outline is complete.
    """
    is_complete: bool = Field(default=False, description="Check if the outline is complete.")
    message: str = Field(default="", description="Message to be displayed.")

def check_outline_complete(state: NovelState) -> NovelState:
    """
    Check if the outline is complete.
    """
    return state["outline_complete"]  # This should be set to True or False based on the outline review process



def generate_full_outline(state: NovelState) -> NovelState:
    """
    Generate the full outline for the novel.
    """
    # Check if the outline is complete
    state["chapter_outlines"]["all"] = """
    <BASECONTEXT>
    {_BaseContext}
    </BASECONTEXT>

    <STORYELEMENTS>
    {_StoryElements}
    </STORYELEMENTS>

    <OUTLINE>
    {_Outline}
    </OUTLINE>
    """.format(
        _BaseContext = state["revised_user_context"],
        _StoryElements = state["story_elements"],   
        _Outline = state["chapter_outlines"]["all"],
    )




class CheckChapterCount(BasicNovelWriterAgent):
    def __init__(self, llm:OllamaLLM, 
                 system_prompt: str = WRITER_SYSTEM,
                 prompt_template: str = CHAPTER_COUNT_PROMPT,
                 personal_preference_prompt:str = PERSONAL_PREFERENCE_PROMPT,
                 ) -> None:
        super().__init__(llm, system_prompt, prompt_template, personal_preference_prompt)

    def __call__(self, state: NovelState) -> NovelState:
        if self.personal_preference_prompt:
            self.prompt_template = self.personal_preference_prompt + "\n" + self.prompt_template
        user_prompt = self.prompt_template.format(
            _Summary = state["revised_user_context"] + "\n" + state["chapter_outlines"]["all"],
        )
        response = self.llm.invoke(
            user_prompt
        )
        response = remove_think(response)
        response = strip_any_unnecessary_chars_for_json(response)
        n_chapters = json.loads(response)["TotalChapters"]
        state["chapter_outlines"].update(
            {i: "" for i in range(n_chapters)}
        )
        state["chapters"] = {i: "" for i in range(n_chapters)}
        state["current_chapter_index"] = 0
        state["current_chapter"] = ""
        state["last_chapter"] = ""
        return state
    

class ChapterOutlineAgent(BasicNovelWriterAgent):
    def __init__(self, llm: OllamaLLM, 
                 system_prompt: str = WRITER_SYSTEM,
                 prompt_template: str = CHAPTER_OUTLINE_PROMPT,
                 personal_preference_prompt: str = PERSONAL_PREFERENCE_PROMPT,
                 ) -> None:
        super().__init__(llm, system_prompt, prompt_template, personal_preference_prompt)

    def __call__(self, state: NovelState) -> NovelState:
        self._add_personal_preference()
        for i in range(len(state["chapters"])):
            state["current_chapter_index"] = i
            user_prompt = self.prompt_template.format(
                _ChapterNum=i+1,
                _Outline=state["chapter_outlines"]["all"],
                _Prompt=state["revised_user_context"],
            )
            response = self._call_llm(state, user_prompt, logfile_suffix = f"chapter-{i+1}", with_history = False)
            state["chapter_outlines"][i] = response
        return state

def create_mega_outline(state: NovelState) -> NovelState:
    """
    Create the mega outline for the novel.
    """
    # Check if the outline is complete
    state["mega_outline"] = """
    <BASEOUTLINE>
    {_StoryElements}
    </BASEOUTLINE>

    <DETAILEDOUTLINE>
    {_Outline}
    </DETAILEDOUTLINE>
    """.format(
        _StoryElements = state["story_elements"],
        _Outline = "\n".join([state["chapter_outlines"][i] for i in range(len(state["chapters"]))]),
    )



def reset_chapter(state: NovelState) -> NovelState:
    """
    Reset the chapter to the first chapter.
    """
    state["current_chapter_index"] = 0
    return state

class LastChapterSummaryAgent(BasicNovelWriterAgent):
    def __init__(self, llm: OllamaLLM, 
                 system_prompt: str = WRITER_SYSTEM,
                 prompt_template: str = LAST_CHAPTER_SUMMARY_PROMPT,
                 personal_preference_prompt: str = PERSONAL_PREFERENCE_PROMPT,
                 ) -> None:
        super().__init__(llm, system_prompt, prompt_template, personal_preference_prompt)

    def __call__(self, state: NovelState) -> NovelState:
        if state["current_chapter_index"] <= 1:
            state["last_chapter"] = "No previous chapter, This is the first chapter."
            state["last_chapter_summary"] = "No previous chapter, This is the first chapter."
            return state
        user_prompt = self.prompt_template.format(
            _ChapterNum=state["current_chapter_index"] + 1,
            _LastChapterNum=state["current_chapter_index"] + 1 - 1,
            _TotalChapters=len(state["chapters"]) - 1,
            _LastChapter=state["chapters"][state["current_chapter_index"] - 1],
            _Outline=state["chapter_outlines"]["all"],
            _ThisOutline=state["chapter_outlines"][state["current_chapter_index"]],
        )
        response = self._call_llm(state, user_prompt, with_history = False)
        state["last_chapter"] = state["chapters"][state["current_chapter_index"] - 1]
        state["last_chapter_summary"] = response
        state["current_chapter"] = state["chapters"][state["current_chapter_index"]]
        return state
    
class NextChapterConnectAgent(BasicNovelWriterAgent):
    def __init__(self, llm: OllamaLLM, 
                 system_prompt: str = WRITER_SYSTEM,
                 prompt_template: str = NEXT_CHAPTER_CONNECT_PROMPT,
                 personal_preference_prompt: str = PERSONAL_PREFERENCE_PROMPT,
                 ) -> None:
        super().__init__(llm, system_prompt, prompt_template, personal_preference_prompt)

    def __call__(self, state: NovelState) -> NovelState:
        if state["current_chapter_index"] >= len(state["chapters"]) - 1:
            state["next_chapter_summary"] = "No next chapter, This is the last chapter."
            return state
        user_prompt = self.prompt_template.format(
            _ChapterNum = state["current_chapter_index"] + 1,
            _NextChapterNum = state["current_chapter_index"] + 2,
            _Outline = state["chapter_outlines"]["all"],
            _ThisOutline = state["chapter_outlines"][state["current_chapter_index"]],
            _NextOutline = state["chapter_outlines"][state["current_chapter_index"] + 1],
        )
        response = self._call_llm(state, user_prompt, with_history = False)
        state["next_chapter_summary"] = response
        return state

def switch_chapter(state: NovelState) -> NovelState:
    """
    Switch to the next chapter. Return True if the last chapter is reached.
    """
    state["n_chapter_reviews"] = 0
    state["current_feedback"] = ""
    a = state["current_chapter_index"]
    print(f"Current chapter index: {a}")
    print(f"Total chapters: {len(state['chapters'])}")
    if a < len(state["chapters"]):
        state["current_chapter_index"] += 1
    return state

    


class ChapterGenerationAgent(BasicNovelWriterAgent):
    def __init__(
        self,
        llm: OllamaLLM,
        system_prompt: str = WRITER_SYSTEM,
        prompt_template: str = CHAPTER_GENERATION_STAGE1_PROMPT,
        personal_preference_prompt: str = PERSONAL_PREFERENCE_PROMPT,
        min_response_length: int = 800,
        max_retry: int = 10,
        max_response_length: int = 5000,
    ) -> None:
        super().__init__(llm, system_prompt, prompt_template, personal_preference_prompt)
        self._min_response_length = min_response_length
        self._max_response_length = max_response_length
        self.min_response_length = min_response_length
        self.max_response_length = max_response_length
        self.last_response_length = 0
        self.max_retry = max_retry

    def _round(self, state: NovelState, prompt_template: str) -> None:
        if self.personal_preference_prompt and self.personal_preference_prompt not in prompt_template:
            prompt_template = self.personal_preference_prompt + "\n" + prompt_template

        if state["current_chapter_index"] == 0:
            last_chapter = "No previous chapter, This is the first chapter."
            last_chapter_summary = "No previous chapter, This is the first chapter."
        else:
            last_chapter = state["chapters"][state["current_chapter_index"] - 1]
            last_chapter_summary = state["last_chapter_summary"]
        if state["current_chapter_index"] >= len(state["chapters"]) - 1:
            next_chapter_summary = "No next chapter, This is the last chapter."
        else:
            next_chapter_summary = state["next_chapter_summary"]

        user_prompt = prompt_template.format(
            _ChapterNum=state["current_chapter_index"] + 1,
            _TotalChapters=len(state["chapters"]),
            _LastChapterSummary=last_chapter_summary,
            _LastChapter=last_chapter,
            _LastChapterNum=state["current_chapter_index"],
            _NextChapterSummary=next_chapter_summary,
            _NextChapterNum=state["current_chapter_index"] + 2,
            _Outline=state["chapter_outlines"][state["current_chapter_index"]],
            _Chapter=state["chapters"][state["current_chapter_index"]],
            _Background="\n".join([state["revised_user_context"], state["story_elements"]]),
        )
        response = self._call_llm(state, user_prompt, with_history = False, logfile_suffix= f"chapter-{state['current_chapter_index'] + 1}")
        if len(response) >= self.min_response_length and len(response) <= self.max_response_length:
            self.last_response_length = len(response)
            self.min_response_length = int(self.last_response_length * 0.8)
        elif len(response) < self.min_response_length:
            for i in range(self.max_retry):
                print(f"response length is less than the minimum response length {self.min_response_length}, " + f"Retrying {i+1} times.")
                user_prompt_1 = user_prompt + "\n" + INSUFFICIENT_LENGTH_PROMPT.format(_Length = self.min_response_length)
                response = self._call_llm(state, user_prompt_1, with_history = False, logfile_suffix= f"chapter-{state['current_chapter_index'] + 1}")
                if len(response) >= self.min_response_length:
                    self.last_response_length = len(response)
                    break
            else:
                print("Warning: response length is still less than the minimum response length.")
        elif len(response) > self.max_response_length:
            print(f"response length is greater than the maximum response length {self.max_response_length}, " + f"No longer extending the response length.")
            # for i in range(self.max_retry):
            #     print(f"response length is greater than the maximum response length {self.max_response_length}, " + f"Retrying {i+1} times.")
            #     user_prompt_2 = user_prompt + "\n" + OVER_LENGTH_PROMPT.format(_Length = self.max_response_length)
            #     response = self._call_llm(state, user_prompt_2, with_history = False, logfile_suffix= f"chapter-{state['current_chapter_index'] + 1}")
            #     if len(response) <= self.max_response_length:
            #         self.last_response_length = len(response)
            #         break
            # else:
            #     print("Warning: response length is still greater than the maximum response length.")

        if self.last_response_length > int(self.last_response_length * 1.25):
            self.min_response_length = min(int(self.last_response_length * 0.8), self.max_response_length * 0.8)
        state["chapters"][state["current_chapter_index"]] = response
        state["current_chapter"] = response

    def __call__(self, state: NovelState) -> NovelState:
        current_chapter = state["current_chapter_index"]
        if current_chapter == 0:
            state["last_chapter_summary"] = "No previous chapter, This is the first chapter."
        self._round(state, CHAPTER_GENERATION_STAGE1_PROMPT)
        self._round(state, CHAPTER_GENERATION_STAGE2_PROMPT)
        # self._round(state, CHAPTER_GENERATION_STAGE3_PROMPT)
        self._round(state, CHAPTER_GENERATION_STAGE4_PROMPT)
        self.min_response_length = self._min_response_length
        self.max_response_length = self._max_response_length

        return state



class CritiqueChapterAgent(BasicNovelWriterAgent):
    def __init__(
        self,
        llm: OllamaLLM,
        system_prompt: str = CRITIQUE_SYSTEM,
        prompt_template: str = CRITIC_CHAPTER_CONNECT_PROMPT,
        personal_preference_prompt: str = PERSONAL_PREFERENCE_PROMPT,
    ) -> None:
        super().__init__(llm, system_prompt, prompt_template, personal_preference_prompt)

    def __call__(self, state: NovelState) -> NovelState:
        return state  # disable this node can give better result. Interesting
        self._add_personal_preference()
        user_prompt = self.prompt_template.format(
            _Prompt=state["revised_user_context"] + "\n" + state["revised_user_context"],
            _ChapterNum=state["current_chapter_index"],
            _TotalChapters=len(state["chapters"]),
            _LastChapterSummary=state["last_chapter_summary"],
            _Outline=state["chapter_outlines"][state["current_chapter_index"]],
            _Chapter=state["chapters"][state["current_chapter_index"]],
        )
        response = self._call_llm(state, user_prompt)
        state["current_feedback"] = response
        print(f"{state['n_chapter_reviews']} reviews for chapter {state['current_chapter_index']} finished.")
        log_response(
            user_prompt,
            response,
            f"{self.__class__.__name__}-chapter-{state['current_chapter_index']}.md",
        )
        return state
    

class ChapterRevisionAgent(BasicNovelWriterAgent):
    def __init__(
        self,
        llm: OllamaLLM,
        system_prompt: str = WRITER_SYSTEM,
        prompt_template: str = CHAPTER_REVISION_PROMPT,
        personal_preference_prompt: str = PERSONAL_PREFERENCE_PROMPT,
    ) -> None:
        super().__init__(llm, system_prompt, prompt_template, personal_preference_prompt)

    def __call__(self, state: NovelState) -> NovelState:
        return state  # disable this node can give better result. Interesting
        self._add_personal_preference()
        user_prompt = self.prompt_template.format(
            _ChapterNum=state["current_chapter_index"],
            _TotalChapters=len(state["chapters"]),
            _LastChapterSummary=state["last_chapter_summary"],
            _Outline=state["chapter_outlines"]['all'],
            _Chapter=state["chapters"][state["current_chapter_index"]],
            _Feedback=state["current_feedback"],
        )
        response = self._call_llm(state, user_prompt)
        state["chapters"][state["current_chapter_index"]] = response
        state["n_chapter_reviews"] += 1
        state["current_chapter"] = response
        print(f"{state['n_chapter_reviews']} revision for chapter {state['current_chapter_index']}")
        log_response(
            user_prompt,
            response,
            f"{self.__class__.__name__}-chapter-{state['current_chapter_index']}.md"
        )
        return state


class ChapterScrubAgent(BasicNovelWriterAgent):
    def __init__(
        self,
        llm: OllamaLLM,
        system_prompt: str = WRITER_SYSTEM,
        prompt_template: str = CHAPTER_SCRUB_PROMPT,
        personal_preference_prompt: str = PERSONAL_PREFERENCE_PROMPT,
    ) -> None:
        super().__init__(llm, system_prompt, prompt_template, personal_preference_prompt)

    def __call__(self, state: NovelState) -> NovelState:
        n_chapters = len(state["chapters"])
        for i in range(n_chapters):
            user_prompt = self.prompt_template.format(
                _Chapter=state["chapters"][i],
            )
            response = self._call_llm(state, user_prompt)
            state["chapters"][i] = response
            state["current_chapter"] = response
        return state




def check_chapter_review_number(state: NovelState) -> bool:
    """
    Check if the number of chapter reviews is less than the maximum number of chapter reviews.
    """
    if state["n_chapter_reviews"] >= state["max_chapter_reviews"]:
        state["n_chapter_reviews"] = 0
        return True
    return False

def check_chapter_finished(state: NovelState) -> bool:
    """
    Check if the chapter is finished.
    """
    if state["current_chapter_index"] >= len(state["chapters"]):
        return True
    return False

def output_story(state: NovelState) -> NovelState:
    """
    Output the story.
    """
    storypath = state.get("output_dir")
    dirname = os.path.dirname(storypath)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(storypath, "w", encoding="utf-8") as f:
        for i in range(len(state["chapters"])):
            f.write(f"Chapter {i+1}:\n")
            f.write(state["chapters"][i])
            f.write("\n\n")
    return state


def build_graph(planner_llm:BaseLLM, writer_llm:BaseLLM, critic_llm:BaseLLM, reviser_llm:BaseLLM) -> StateGraph:
    planner = planner_llm
    writer  = writer_llm
    critic  = critic_llm
    reviser = reviser_llm

    graph_builder = StateGraph(state_schema=NovelState,)
    graph_builder.add_node("initialize_state", initialize_state)
    graph_builder.add_edge(START, "initialize_state")
    graph_builder.add_node("get_important_base_info", GetImportantBaseInfoAgent(llm=writer))
    graph_builder.add_edge("initialize_state", "get_important_base_info")
    graph_builder.add_node("generate_story_elements", GenerateStoryElementAgent(llm=writer))
    graph_builder.add_edge("get_important_base_info", "generate_story_elements")

    graph_builder.add_node("whether_write_outline", WhetherWriteOutlineAgent(llm=planner))
    # graph_builder.add_node("clean_outline", CleanOutlineAgent(llm=planner))
    # graph_builder.add_node("clean_user_prompt", CleanUserPromptAgent(llm=planner))

    graph_builder.add_node("clean_outline", clean_outline)
    graph_builder.add_node("clean_user_prompt", CleanUserPromptAgent(llm=planner))  # no-op, just to keep the structure consistent

    graph_builder.add_node("initial_outline",       InitialOutlineAgent(llm=writer))

    graph_builder.add_edge("generate_story_elements", "whether_write_outline")

    graph_builder.add_conditional_edges(
        "whether_write_outline",
        check_whether_write_outline,
        {
            True : "initial_outline",
            False: "clean_outline",
        },
    )
    graph_builder.add_edge("clean_outline", "clean_user_prompt")


    graph_builder.add_node("critique_outline_general",   CritiqueOutlineGeneralAgent(llm=critic, prompt_template=CRITIC_OUTLINE_GENERAL_PROMPT))
    graph_builder.add_node("critique_outline_complete", CritiqueOutlineCompleteAgent(llm=critic, prompt_template=CRITIC_OUTLINE_COMPLETE_PROMPT))
    graph_builder.add_conditional_edges(
        "initial_outline",
        check_output_length,
        {
            False: "initial_outline",
            True : "critique_outline_general",
        },
    )
    graph_builder.add_edge("clean_user_prompt", "critique_outline_general")
    graph_builder.add_edge("critique_outline_general", "critique_outline_complete")

    graph_builder.add_node("outline_revision", OutlineRevisionAgent(llm=reviser, prompt_template=OUTLINE_REVISION_PROMPT))
    
    graph_builder.add_conditional_edges(
        "critique_outline_complete",
        check_outline_complete,
        {
            False: "outline_revision",
            True : "generate_full_outline",
        },
    )
    
    graph_builder.add_node("generate_full_outline", generate_full_outline)
    graph_builder.add_conditional_edges(
        "outline_revision",
        check_outline_review_number,
        {
            False: "critique_outline_general",
            True : "generate_full_outline",
        },
    )
    graph_builder.add_node("check_chapter_count", CheckChapterCount(llm=planner))
    graph_builder.add_edge("generate_full_outline", "check_chapter_count")
    graph_builder.add_node("chapter_outline", ChapterOutlineAgent(llm=writer))
    graph_builder.add_edge("check_chapter_count", "chapter_outline")
    graph_builder.add_node("create_mega_outline", create_mega_outline)
    graph_builder.add_edge("chapter_outline", "create_mega_outline")


    graph_builder.add_node("reset_chapter", reset_chapter)
    graph_builder.add_edge("create_mega_outline", "reset_chapter")


    # chapter writing, complex route to improve quality
    graph_builder.add_node("last_chaptersummary", LastChapterSummaryAgent(llm=planner))
    graph_builder.add_node("next_chapter_connect", NextChapterConnectAgent(llm=planner))
    graph_builder.add_node("chapter_generation", ChapterGenerationAgent(llm=writer))
    graph_builder.add_node("critique_chapter", CritiqueChapterAgent(llm=critic, prompt_template=CRITIC_CHAPTER_CONNECT_PROMPT))
    graph_builder.add_node("chapter_revision", ChapterRevisionAgent(llm=reviser, prompt_template=CHAPTER_REVISION_PROMPT))
    graph_builder.add_node("switch_chapter", switch_chapter)

    graph_builder.add_edge("reset_chapter", "last_chaptersummary")
    graph_builder.add_edge("last_chaptersummary", "next_chapter_connect")
    graph_builder.add_edge("next_chapter_connect", "chapter_generation")
    graph_builder.add_edge("chapter_generation", "critique_chapter")
    graph_builder.add_edge("critique_chapter", "chapter_revision")
    graph_builder.add_conditional_edges(
        "chapter_revision",
        check_chapter_review_number,
        {
            False: "critique_chapter",
            True : "switch_chapter",
        },
    )
    graph_builder.add_conditional_edges(
        "switch_chapter",
        check_chapter_finished,
        {
            False: "last_chaptersummary",
            True : "output_story",
        },
    )
    # graph_builder.add_node("chapter_scrub", ChapterScrubAgent(llm=llm1))
    graph_builder.add_node("output_story", output_story)
    # graph_builder.add_edge("chapter_scrub", "output_story")
    graph_builder.add_edge("output_story", END)
    # ph_builder.add_edge("chapter_generation", "switch_chapter")
    return graph_builder    

