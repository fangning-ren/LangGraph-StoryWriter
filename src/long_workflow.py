from langgraph.prebuilt import create_react_agent
from langchain_ollama import OllamaLLM
from langchain_together import Together, ChatTogether
from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseLLM
from langgraph.types import RetryPolicy

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

from datetime import datetime
import os


from .prompts import *
from .utils import *
from .states import *
from .base import BasicNovelWriterAgent
from .generate_chapter import *
from .generate_outline import *
from .generate_character import *




def build_graph(planner_llm:BaseLLM, writer_llm:BaseLLM, critic_llm:BaseLLM, reviser_llm:BaseLLM) -> StateGraph:
    planner = planner_llm
    writer  = writer_llm
    critic  = critic_llm
    reviser = reviser_llm

    retry_policy = RetryPolicy(
        max_attempts=5,
    )

    graph_builder = StateGraph(state_schema=NovelState,)
    graph_builder.add_node("get_important_base_info", GetImportantBaseInfoAgent(llm=writer), retry=retry_policy)
    graph_builder.add_edge(START, "get_important_base_info")
    graph_builder.add_node("generate_story_elements", GenerateStoryElementAgent(llm=writer), retry=retry_policy)
    graph_builder.add_edge("get_important_base_info", "generate_story_elements")
    graph_builder.add_node("split_characters", SplitCharactersAgent(llm=planner), retry=retry_policy)
    graph_builder.add_edge("generate_story_elements", "split_characters")

    graph_builder.add_node("whether_write_outline", WhetherWriteOutlineAgent(llm=planner), retry=retry_policy)
    graph_builder.add_edge("split_characters", "whether_write_outline")
    # graph_builder.add_node("clean_outline", CleanOutlineAgent(llm=planner))
    graph_builder.add_node("clean_outline", clean_outline)
    graph_builder.add_node("clean_user_prompt", CleanUserPromptAgent(llm=planner), retry=retry_policy)  # no-op, just to keep the structure consistent

    graph_builder.add_node("initial_outline",       InitialOutlineAgent(llm=writer), retry=retry_policy)

    graph_builder.add_conditional_edges(
        "whether_write_outline",
        check_whether_write_outline,
        {
            True : "initial_outline",
            False: "clean_outline",
        },
    )
    graph_builder.add_edge("clean_outline", "clean_user_prompt")


    graph_builder.add_conditional_edges(
        "initial_outline",
        check_output_length,
        {
            False: "initial_outline",
            True : "dummy_outline_node",  # Dummy node to keep the structure consistent
        },
    )

    graph_builder.add_node("dummy_outline_node", lambda state: state, retry=retry_policy)  # Dummy node to keep the structure consistent
    graph_builder.add_edge("clean_user_prompt", "dummy_outline_node")
    graph_builder.add_conditional_edges(
        "dummy_outline_node",
        is_max_outline_review_reached,
        {
            False: "critique_outline_complete",
            True : "generate_full_outline",
        },
    )

    graph_builder.add_node("critique_outline_general",   CritiqueOutlineGeneralAgent(llm=critic, prompt_template=CRITIC_OUTLINE_GENERAL_PROMPT), retry=retry_policy)
    graph_builder.add_node("critique_outline_complete", CritiqueOutlineCompleteAgent(llm=critic, prompt_template=CRITIC_OUTLINE_COMPLETE_PROMPT), retry=retry_policy)
    graph_builder.add_node("outline_revision", OutlineRevisionAgent(llm=reviser, prompt_template=OUTLINE_REVISION_PROMPT), retry=retry_policy)
    graph_builder.add_conditional_edges(
        "critique_outline_complete",
        is_outline_complete,
        {
            False: "critique_outline_general",
            True : "generate_full_outline",
        },
    )
    graph_builder.add_edge("critique_outline_general", "outline_revision")
    graph_builder.add_edge("outline_revision", "dummy_outline_node")
    
    graph_builder.add_node("generate_full_outline", generate_full_outline)
    
    graph_builder.add_node("check_chapter_count", CheckChapterCount(llm=planner), retry=retry_policy)
    graph_builder.add_edge("generate_full_outline", "check_chapter_count")

    graph_builder.add_node("chapter_outline", ChapterOutlineAgent(llm=writer), retry=retry_policy)


    
    graph_builder.add_edge("check_chapter_count", "chapter_outline")
    graph_builder.add_node("reset_chapter", reset_chapter)
    graph_builder.add_edge("chapter_outline", "reset_chapter")


    # chapter writing, complex route to improve quality
    graph_builder.add_node("adjust_character", AdjustCharacterAgent(llm=planner), retry=retry_policy)
    graph_builder.add_node("add_character", AddCharacterAgent(llm=writer), retry=retry_policy)
    graph_builder.add_node("last_chaptersummary", LastChapterSummaryAgent(llm=planner), retry=retry_policy)
    graph_builder.add_node("next_chapter_connect", NextChapterConnectAgent(llm=planner), retry=retry_policy)
    graph_builder.add_node("chapter_generation", ChapterGenerationAgent(llm=writer), retry=retry_policy)

    graph_builder.add_edge("reset_chapter", "adjust_character")
    graph_builder.add_edge("adjust_character", "add_character")
    graph_builder.add_edge("add_character", "last_chaptersummary")
    graph_builder.add_edge("last_chaptersummary", "next_chapter_connect")
    graph_builder.add_edge("next_chapter_connect", "chapter_generation")

    # chapter critique and revision
    graph_builder.add_node("dummy_chapter_node", lambda state: state, retry=retry_policy)  # Dummy node to keep the structure consistent
    graph_builder.add_node("critique_chapter_general", CritiqueChapterGeneralAgent(llm=critic, prompt_template=CRITIC_CHAPTER_GENERAL_PROMPT), retry=retry_policy)
    graph_builder.add_node("critique_chapter_complete", CritiqueChapterCompleteAgent(llm=critic, prompt_template=CRITIC_CHAPTER_COMPLETE_PROMPT), retry=retry_policy)
    graph_builder.add_node("chapter_revision", ChapterRevisionAgent(llm=reviser, prompt_template=CHAPTER_REVISION_PROMPT), retry=retry_policy)
    
    graph_builder.add_edge("chapter_generation", "dummy_chapter_node")
    graph_builder.add_conditional_edges(
        "dummy_chapter_node",
        is_max_chapter_review_reached,
        {
            False: "critique_chapter_complete",
            True : "character_update",
        },
    )
    graph_builder.add_conditional_edges(
        "critique_chapter_complete",
        is_chapter_complete,
        {
            False: "critique_chapter_general",
            True : "character_update",
        },
    )
    graph_builder.add_edge("critique_chapter_general", "chapter_revision")
    graph_builder.add_edge("chapter_revision", "dummy_chapter_node")


    # post-chapter character update and chapter switch
    graph_builder.add_node("character_update", CharacterUpdateAgent(llm=writer, prompt_template=CHARACTER_UPDATE_PROMPT), retry=retry_policy)
    graph_builder.add_node("switch_chapter", switch_chapter)
    graph_builder.add_edge("character_update", "switch_chapter")
    graph_builder.add_conditional_edges(
        "switch_chapter",
        is_last_chapter,
        {
            False: "adjust_character",
            True : "output_story",
        },
    )

    # graph_builder.add_node("chapter_scrub", ChapterScrubAgent(llm=llm1))
    graph_builder.add_node("output_story", output_story)
    # graph_builder.add_edge("chapter_scrub", "output_story")
    graph_builder.add_edge("output_story", END)
    # ph_builder.add_edge("chapter_generation", "switch_chapter")
    return graph_builder    

