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


CHARACTER_TEMPLATE = """
#### 角色 1
- **姓名**:
- **外貌描述**:
- **性格**:
- **背景**:
- **人际关系**:
- **特殊能力（如有）**:
- **故事中的角色**:
- **历史经历**:
"""



class Character(BaseModel):
    index: int = Field(default=0, description="Index of the character in the story")
    is_main_character: bool = Field(default=False, description="Whether the character is a main character in the story")
    historical_description: list[str] = Field(default_factory=list, description="List of historical descriptions of the character")
    description: str = Field(default="", description="Description of the character, can be empty if not specified")
    template: str = Field(default=CHARACTER_TEMPLATE, description="Template for the character, can be empty if not specified")
    appears: bool = Field(default=False, description="Whether the character appears in the story")
    # do not use json schema or some other fancy things here. 
    # store all information as a simple markdown formatted string 
    # this allows LLM to easily understand and update the character information 

    def __str__(self):
        return self.description
    
    def __getitem__(self, key):
        """
        Get the attribute of the character by key
        """
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        """
        Set the attribute of the character by key
        """
        return setattr(self, key, value)
    
    def get_name(self):
        """
        Get the name of the character from the description
        """
        lines = self.description.strip().split("\n")
        for line in lines:
            if "姓名" in line:
                return "".join(line.split("：")[1:])
    
    def from_string(self, description: str, index: int = 0):
        """
        Update the character's description from a markdown formatted string
        """
        lines = description.strip().split("\n")
        line0 = lines[0].strip()
        if "主要" in line0:
            self.is_main_character = True

        self.index = index
        if self.is_main_character:
            self.description = f"#### 主要角色 {self.index + 1}\n"
        else:
            self.description = f"#### 角色 {self.index + 1}\n"
        self.description += "\n".join(lines[1:]).strip()
        return self
        

class ChapterState(BaseModel):
    index: int = Field(default=0, description="Index of the chapter in the part")
    title: str = Field(default="", description="Title of the chapter, can be empty if not specified")
    outline: str = Field(default="", description="Outline of the chapter, can be empty if not specified")
    content: str = Field(default="", description="Content of the chapter, can be empty if not specified")
    theme: str = Field(default="", description="Theme of the chapter, can be empty if not specified")
    characters: list[Character] = Field(default_factory=list, description="List of characters involved in the chapter")
    settings: list[str] = Field(default_factory=list, description="List of settings or locations in the chapter")
    last_cohesion_requirement: str = Field(default="", description="Requirements for cohesively continuing the previous chapter")
    next_cohesion_requirement: str = Field(default="", description="Requirements for cohesively continuing the next chapter")
    next_cohision_suggestion: str = Field(default="", description="Suggestion for the next chapter to cohesively continue the story")
    chapter_summary: str = Field(default="", description="Summary of the chapter")
    max_length: int = Field(default=5000, description="Maximum length of the chapter content")
    min_length: int = Field(default=1000, description="Minimum length of the chapter content")

    current_feedback: str = Field(default="", description="Current feedback from LLM about the chapter")
    outline_complete: bool = Field(default=False, description="Whether the outline of the chapter is complete")
    outline_revision_count: int = Field(default=0, description="Number of revisions made to the chapter outline")
    outline_max_revisions: int = Field(default=0, description="Maximum number of revisions allowed for the chapter outline")
    content_revision_count: int = Field(default=0, description="Number of revisions made to the chapter")
    content_max_revisions: int = Field(default=3, description="Maximum number of revisions allowed for the chapter")

    class Config:
        arbitrary_types_allowed = True

    def __getitem__(self, key):
        """
        Get the attribute of the character by key
        """
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        """
        Set the attribute of the character by key
        """
        return setattr(self, key, value)

    def get_characters(self):
        """
        return a markdown formatted string of all characters in the story
        """
        if not self["characters"]:
            return "No characters defined in the story."
        
        # 首先拉出所有主要角色
        main_characters = [char for char in self["characters"] if isinstance(char, Character) and char.is_main_character]
        # 然后拉出所有非主要角色
        non_main_characters = [char for char in self["characters"] if isinstance(char, Character) and not char.is_main_character]

        printed_strings = ["\n", ]
        for i, main_char in enumerate(main_characters):
            printed_strings.append(f"### 主要角色 \n")
            printed_strings.append(f"#### 主要角色 {i + 1}\n")
            s = main_char.description.strip().split("\n")
            s = [line for line in s if line.strip()]
            s = [line for line in s if not line.startswith("####")]
            printed_strings.extend(s)
            printed_strings.append("\n")
        for i, non_main_char in enumerate(non_main_characters):
            printed_strings.append(f"### 配角 \n")
            printed_strings.append(f"#### 角色 {i + 1}\n")
            s = non_main_char.description.strip().split("\n")
            s = [line for line in s if line.strip()]
            s = [line for line in s if not line.startswith("####")]
            printed_strings.extend(s)
            printed_strings.append("\n")
        return "\n".join(printed_strings)

class PartState(BaseModel):
    index: int = Field(default=0, description="Index of the part in the book")
    title: str = Field(default="", description="Title of the part")
    outline: str = Field(default="", description="Outline of the part")
    theme: str = Field(default="", description="Theme of the part")
    characters: list[Character] = Field(default_factory=list, description="List of characters involved in the part")
    settings: list[str] = Field(default_factory=list, description="List of settings or locations in the part")

    last_cohesion_requirement: str = Field(default="", description="Requirements for cohesively continue the previous part")
    next_cohesion_requirement: str = Field(default="", description="Requirements for cohesively continue the next part")
    next_cohesion_suggestion: str = Field(default="", description="Suggestion for the next part to cohesively continue the story")
    part_summary: str = Field(default="", description="Summary of the part")

    outline_revision_count: int = Field(default=0, description="Number of revisions made to the part outline")
    outline_max_revisions: int = Field(default=0, description="Maximum number of revisions allowed for the part outline")
    is_generate: bool = Field(default=True, description="Whether the outline of the part need to be generated")
    is_outline_complete: bool = Field(default=False, description="Whether the outline of the part is complete")

    chapters: list[ChapterState] = Field(default_factory=list, description="List of chapters in the part")
    current_chapter_index: int = Field(default=0, description="Index of the current chapter in the part")

    class Config:
        arbitrary_types_allowed = True  # support for custom types like ChapterState

    def __getitem__(self, key):
        """
        Get the attribute of the character by key
        """
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        """
        Set the attribute of the character by key
        """
        return setattr(self, key, value)

class NovelState(BaseModel):
    messages: list = Field(default_factory=list, description="List of messages exchanged in the conversation")

    # global information about the story
    user_context: str = Field(default="", description="User's background information or preferences")
    revised_user_context: str = Field(default="", description="Revised user context after LLM processing")
    story_settings: str = Field(default="", description="Settings of the story, such as time period, location, and world-building details")
    story_elements: str = Field(default="", description="this is story settings + characters. This field is dynamic")

    # info at the Book level, include the outline of each book
    outline: str = Field(default="", description="Overall outline of the story")
    title: str = Field(default="", description="Title of the story")
    theme: str = Field(default="", description="Theme of the story")
    characters: list[Character] = Field(default_factory=list, description="List of characters in the story")
    do_outline_generation: bool = Field(default=True, description="Whether to generate the outline of the story")

    # store critique feedback from LLM
    current_feedback: str = Field(default="", description="Current feedback from LLM about the story")
    outline_complete: bool = Field(default=False, description="Whether the outline of the story is complete")
    n_outline_reviews: int = Field(default=0, description="Number of reviews made to the story outline")
    max_outline_reviews: int = Field(default=3, description="Maximum number of reviews allowed for the story outline")

    # repeat
    n_repeats: int = Field(default=0, description="Number of times the story has been repeated")
    max_repeats: int = Field(default=3, description="Maximum number of times the story can be repeated")

    # part level information (Not implemented yet)
    # parts: list = Field(default_factory=list, description="List of parts in the book, each part contains its own chapters and outlines")
    # current_part_index: int = Field(default=0, description="Index of the current part in the book")

    # chapter level information
    chapters: list[ChapterState] = Field(default_factory=list, description="List of chapters in the book, each chapter contains its own outline and content")
    current_chapter_index: int = Field(default=0, description="Index of the current chapter in the book")

    # techincal information
    working_dir: str = Field(default="./stories/", description="Directory where the story files are stored")
    output_dir: str = Field(default="./stories/story.txt", description="Output file path for the story")
    logging_dir: str = Field(default="./logs/", description="Directory where the logs are stored")

    def __getitem__(self, key):
        """
        Get the attribute of the character by key
        """
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        """
        Set the attribute of the character by key
        """
        return setattr(self, key, value)
    
    def get_character_name_list(self):
        """
        return a formatted string of all character names in the story
        """
        if not self["characters"]:
            return "No characters defined in the story."
        character_names = []
        for char in self["characters"]:
            if isinstance(char, Character):
                name = char.get_name()
                if name:
                    character_names.append(name)
        if not character_names:
            return "No character names found in the story."
        character_names = "\n".join(character_names)
        character_names = f"### 角色列表\n{character_names}\n"
        
        return character_names

    def get_characters(self):
        """
        return a markdown formatted string of all characters in the story
        """
        if not self["characters"]:
            return "No characters defined in the story."
        
        # 首先拉出所有主要角色
        main_characters = [char for char in self["characters"] if isinstance(char, Character) and char.is_main_character]
        # 然后拉出所有非主要角色
        non_main_characters = [char for char in self["characters"] if isinstance(char, Character) and not char.is_main_character]

        printed_strings = ["\n", ]
        for i, main_char in enumerate(main_characters):
            printed_strings.append(f"### 主要角色 \n")
            printed_strings.append(f"#### 主要角色 {i + 1}\n")
            s = main_char.description.strip().split("\n")
            s = [line for line in s if line.strip()]
            s = [line for line in s if not line.startswith("####")]
            printed_strings.extend(s)
            printed_strings.append("\n")
        for i, non_main_char in enumerate(non_main_characters):
            printed_strings.append(f"#### 角色 {i + 1}\n")
            s = non_main_char.description.strip().split("\n")
            s = [line for line in s if line.strip()]
            s = [line for line in s if not line.startswith("####")]
            printed_strings.extend(s)
            printed_strings.append("\n")
        return "\n".join(printed_strings)
    
    def __getitem__(self, key):
        return getattr(self, key)

if __name__ == "__main__":
    # Example usage

    desc = """
#### 角色 2
- **姓名**：绵月依姬
- **外貌描述**：紫色马尾，红色长裙，手持武士刀。
- **性格**：严肃认真，冷静理智。
- **背景**：月之都的公主之一，永琳的学生。
- **人际关系**：与绵月丰姬是姐妹，称呼永琳为“师匠”。
- **特殊能力**：召唤所有神明（如天照大神、火雷神等）。
- **故事中的角色**：月之都的军事领袖，负责执行计划。
    """

    character = Character(
        index=1,
        is_main_character=True,
        description=desc,
        template=CHARACTER_TEMPLATE,
    )
    print(character)