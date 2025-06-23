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

class AdjustCharacterAgent(BasicNovelWriterAgent):
    def __init__(self, llm: OllamaLLM, 
                 system_prompt: str = GENERAL_SYSTEM,
                 prompt_template: str = CHARACTER_APPEAR_PROMPT, 
                 personal_preference_prompt: str = PERSONAL_PREFERENCE_PROMPT,
                 ) -> None:
        super().__init__(llm, system_prompt, prompt_template, personal_preference_prompt)

    def __call__(self, state: NovelState) -> NovelState:
        prompt_template1 = CHARACTER_APPEAR_PROMPT

        if state["current_chapter_index"] == len(state["chapters"]) - 1:
            return state  # No need to adjust characters in the last chapter

        # switch all character's appears to False
        for char in state["characters"]:
            char: Character
            char["appears"] = False

        # assign characters to chapters based on their appearances
        chapter = state["chapters"][state["current_chapter_index"]]
        chapter: ChapterState
        for character in state["characters"]:
            character: Character
            user_prompt = prompt_template1.format(
                _ChapterContent=chapter["outline"],
                _Character=character["description"],
            )
            response = self._call_llm(state, user_prompt, with_history=False, format_json=True)
            char_appears = get_boolean_result_anyway(response, "IsAppeared")
            if char_appears:
                character["appears"] = True
                chapter["characters"].append(character)

        return state
    
class AddCharacterAgent(BasicNovelWriterAgent):
    def __init__(self, llm: OllamaLLM, 
                 system_prompt: str = GENERAL_SYSTEM,
                 prompt_template: str = ADD_CHARACTER_PROMPT, 
                 personal_preference_prompt: str = PERSONAL_PREFERENCE_PROMPT,
                 ) -> None:
        super().__init__(llm, system_prompt, prompt_template, personal_preference_prompt)

    def __call__(self, state: NovelState) -> NovelState:
        if state["current_chapter_index"] == len(state["chapters"]) - 1:
            return state    # No need to add characters in the last chapter
        
        # add more characters to each chapter based on the chapter outline
        chapter = state["chapters"][state["current_chapter_index"]]
        chapter: ChapterState
        prompt_template2 = ADD_CHARACTER_PROMPT
        user_prompt = prompt_template2.format(
            _Background=state["story_settings"],
            _ChapterNum= state["current_chapter_index"] + 1,
            _ChapterOutline=chapter["outline"],
            _ExistingCharacters=chapter.get_characters()
        )
        response = self._call_llm(state, user_prompt, with_history=False)
        if "不需要添加新角色" in response or len(response.strip()) < 15:
            return state
            
        # parse the response and add new characters
        new_characters = split_by_sharp(response, split_by="####")
        if len(new_characters) == 0:
            print("No new characters found in the response.")
            return state
        
        prompt_template3 = CHARACTER_EXISTS_PROMPT
        existing_char_list = state.get_character_name_list()
        
        c_index = len(state["characters"])
        for char_str in new_characters:
            if not char_str.strip():
                continue
            prompt = prompt_template3.format(
                _Character=char_str,
                _ExistingCharacters=existing_char_list,
            )
            response = self._call_llm(state, prompt, with_history=False, format_json=True)
            is_char = get_boolean_result_anyway(response, "IsCharacter")
            if not is_char:
                print(f"Not a character description: {char_str.strip()}")
                continue
            char_exists = get_boolean_result_anyway(response, "IsExists")
            if not char_exists:
                character = Character()
                character = character.from_string(
                    description = char_str.strip(), 
                    index = c_index,
                )
                character["appears"] = True
                chapter["characters"].append(character)
                state["characters"].append(character)
                c_index += 1
        return state


class CharacterUpdateAgent(BasicNovelWriterAgent):
    def __init__(
        self,
        llm: OllamaLLM,
        system_prompt: str = WRITER_SYSTEM,
        prompt_template: str = CHARACTER_UPDATE_PROMPT,
        personal_preference_prompt: str = PERSONAL_PREFERENCE_PROMPT,
    ) -> None:
        super().__init__(llm, system_prompt, prompt_template, personal_preference_prompt)

    def __call__(self, state: NovelState) -> NovelState:
        self._add_personal_preference()
        chapter = state["chapters"][state["current_chapter_index"]]
        chapter: ChapterState

        character_list = chapter.get_characters()
        story_settings = state["story_settings"]
        user_contextss = state["revised_user_context"]
        background = "\n".join([user_contextss, story_settings, character_list])

        for character in chapter["characters"]:
            character: Character
            user_prompt = self.prompt_template.format(
                _Background=background,
                _ChapterNum=state["current_chapter_index"] + 1,
                _ChapterContent=chapter["content"],
                _Character = character["description"],
            )
            response = self._call_llm(state, user_prompt)
            character["historical_description"].append(character["description"])
            character["description"] = response
        # the same character object appears in multiple chapters and state. 
        # because it is the same object, the characters in the state will also be updated.
        return state

