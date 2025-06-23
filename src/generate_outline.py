from langchain_ollama import OllamaLLM


from logging import getLogger
import time
import dotenv
dotenv.load_dotenv()

from datetime import datetime
import os


from .prompts import *
from .utils import *
from .states import *
from .base import BasicNovelWriterAgent


class GetImportantBaseInfoAgent(BasicNovelWriterAgent):
    def __init__(self, llm: OllamaLLM, 
                 system_prompt: str = WRITER_SYSTEM,
                 prompt_template: str = GET_IMPORTANT_BASE_INFO_PROMPT,
                 personal_preference_prompt: str = PERSONAL_PREFERENCE_PROMPT,
                 ) -> None:
        super().__init__(llm, system_prompt, prompt_template, personal_preference_prompt)

    def __call__(self, state: NovelState) -> NovelState:
        if not state["user_context"]:
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
    
# split the characters from the story elements
class SplitCharactersAgent(BasicNovelWriterAgent):
    def __init__(self, llm: OllamaLLM, 
                 system_prompt: str = GENERAL_SYSTEM,
                 prompt_template: str = EXTRACT_CHARACTER_PROMPT,
                 personal_preference_prompt: str = PERSONAL_PREFERENCE_PROMPT,
                 ) -> None:
        super().__init__(llm, system_prompt, prompt_template, personal_preference_prompt)

    def __call__(self, state: NovelState) -> NovelState:
        user_prompt = EXTRACT_CHARACTER_PROMPT.format(
            _StoryElements=state["story_elements"],
        )
        response1 = self._call_llm(state, user_prompt)
        
        characters_strs = split_by_sharp(response1, split_by="####")
        characters_strs = [char.strip() for char in characters_strs if char.strip()]
        for i, char_str in enumerate(characters_strs):
            is_main = ("主要角色" in char_str.lower() or "main character" in char_str.lower())
            character = Character(
                index = i, 
                description=char_str, 
                template=CHARACTER_TEMPLATE,
                is_main_character=is_main,
            )
            state["characters"].append(character)

        user_prompt = EXTRACT_ELEMENTS_PROMPT.format(
            _StoryElements=state["story_elements"],
        )
        response2 = self._call_llm(state, user_prompt)
        state["story_settings"] = response2 

        state["story_elements"] = response2 + state.get_characters() 
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
        )
        response = self._call_llm(state, user_prompt, with_history=False, format_json=True)
        state["do_outline_generation"] = not get_boolean_result_anyway(response, "IsOutline")
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
        )
        response = self._call_llm(state, user_prompt)
        state["outline"] = response
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
            _Outline=state["outline"]
        )
        response = self._call_llm(state, user_prompt)
        state["revised_user_context"] = response
        return state


def clean_outline(state: NovelState) -> NovelState:
    """
    Clean the outline from the user context.
    """
    # if the user has an outline, then clean the outline from the user context
    state["outline"] = state["revised_user_context"]
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
        state["outline"] = response
        state["n_repeats"] = 0
        return state
   
def check_output_length(state: NovelState, minoutput: int = 100) -> bool:
    """
    Check if the output length is greater than the minimum output length.
    """
    if len(state["outline"]) >= minoutput:
        state["n_repeats"] = 0
        return True
    
    prompt = INSUFFICIENT_LENGTH_PROMPT.format(_Length = minoutput)
    state["n_repeats"] += 1

    if state["n_repeats"] >= state["max_repeats"]:
        print(f"Output length is insufficient, but reached maximum repeats ({state['max_repeats']}).")
        state["n_repeats"] = 0
        return True
    state["messages"].append(
        {"role": "user", "content": prompt}
    )
    return False


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
            _Outline = state["outline"],
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
            _Outline=state["outline"],
        )
        response = self._call_llm(state, user_prompt, format_json=True)

        is_complete = get_boolean_result_anyway(response, "IsComplete")
        state["outline_complete"] = is_complete
        return state

def is_max_outline_review_reached(state: NovelState) -> bool:
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
            _Outline=state["outline"],
            _Feedback=state["current_feedback"],
        )
        response = self._call_llm(state, user_prompt)
        state["outline"] = response
        state["n_outline_reviews"] += 1
        return state

def is_outline_complete(state: NovelState) -> NovelState:
    """
    Check if the outline is complete.
    """
    return state["outline_complete"]  # This should be set to True or False based on the outline review process



def generate_full_outline(state: NovelState) -> NovelState:
    """
    Generate the full outline for the novel.
    """
    # Check if the outline is complete
    state["outline"] = """
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
        _Outline = state["outline"],
    )
