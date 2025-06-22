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


from src.prompts import *
from src.utils import *
from src.long_workflow import build_graph




if __name__ == "__main__":
    
    


    # llm1 = OllamaLLM(model="deepseek-r1:32b", temperature=0.7, max_tokens=12800)
    # llm2 = OllamaLLM(model="qwen3:32b", temperature=0.7, max_tokens=12800)
    # llm3 = OllamaLLM(model="deepseek-r1:32b", temperature=0.7, max_tokens=12800)
    # llm4 = Together(model="deepseek-ai/DeepSeek-V3", max_tokens=12800, temperature=0.7)

    # benchmark, use larger model
    planner = Together (model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", temperature=0.2, max_tokens=12800)
    writer  = Together (model="deepseek-ai/DeepSeek-V3", temperature=0.7, max_tokens=16384)
    reviser = Together (model="deepseek-ai/DeepSeek-V3", temperature=0.2, max_tokens=16384)
    critic  = Together (model="deepseek-ai/DeepSeek-V3", temperature=0.2, max_tokens=16384)

    # debugging, use smaller model
    # planner = OllamaLLM(model="deepseek-r1:32b", temperature=0.2, max_tokens=8192)
    # writer  = OllamaLLM(model="deepseek-r1:32b", temperature=0.7, max_tokens=8192) 
    # critic  = OllamaLLM(model="deepseek-r1:32b", temperature=0.2, max_tokens=8192)
    # reviser = OllamaLLM(model="deepseek-r1:32b", temperature=0.2, max_tokens=8192)

    graph = build_graph(planner, writer, critic, reviser)
    graph = graph.compile()
    print(graph.get_graph().draw_mermaid())

    import sys
    storyname = sys.argv[1] if len(sys.argv) > 1 else "1942"

    STORYTITLE = storyname + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    storypath = "./stories/" + STORYTITLE + ".txt"

    with open(f"./prompts/{storyname}.txt", "r", encoding="utf-8") as f:
        user_info = f.read()
    user_info = user_info.replace("{_Date}", "2025-05-20")

    max_recursion_depth = 4096 # set to high for long story generation
    # 如果比这个递归深度还大的话，那估计是出现无限递归了
    a = graph.invoke({"user_context": user_info, "output_dir": storypath}, config = {"recursion_limit": max_recursion_depth})
    # dump a as json
    # import json
    # with open(storypath.replace(".txt", ".json"), "w", encoding="utf-8") as f:
    #     json.dump(a, f, ensure_ascii=False, indent=4)
