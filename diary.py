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
import tqdm

from logging import getLogger
import time
import dotenv
dotenv.load_dotenv()


from src.prompts import *
from src.utils import *

PERSONAL_PREFERENCE_PROMPT = "" #"多加抓住女角色脚踝，或脚铐锁住女主脚踝的情节，详细描写每次抓脚的情景和角色感受。"


llm = OllamaLLM(model="qwen3:32b", temperature=0.7, max_tokens=8192)


class DiaryState(TypedDict):
    """
    State for the diary writing process.
    """
    messages: list[dict]
    content: str
    split_line_indices: list[int]
    content_list: list[str]
    extracted_content_list: list[str]
    current_content: str
    last_content: str
    diaries: list[str]
    current_diary: str
    last_diary: str

    character: str
    character_description: str
    date: str
    diarypath: str
    


def initialize_state(state: DiaryState) -> DiaryState:
    """
    Initialize the state for the diary writing process.
    """
    # state["messages"] = []
    state["content"] = ""
    state["split_line_indices"] = []
    state["content_list"] = []
    state["extracted_content_list"] = []
    state["current_content"] = ""
    state["last_content"] = ""
    state["diaries"] = []
    state["current_diary"] = ""
    state["last_diary"] = ""

    # check if the user already provided the path or content in the messages
    if len(state["messages"]) > 0:
        content = state["messages"][0]["content"]
    if os.path.exists(content):
        with open(content, "r", encoding="utf-8") as f:
            state["content"] = f.read()
    else:
        state["content"] = content

    state["date"] = current_date
    state["diarypath"] = diarypath
    state["character"] = character
    state["character_description"] = character_description
    return state



class BasicDiaryWriterAgent(ABC):
    def __init__(self, llm:OllamaLLM, 
                 system_prompt: str = WRITER_SYSTEM,
                 prompt_template: str = "",
                 personal_preference_prompt:str = PERSONAL_PREFERENCE_PROMPT,
                 ) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
        self.personal_preference_prompt = personal_preference_prompt
        self.logger = NovelWritingLogger()

    @abstractmethod
    def __call__(self, state: DiaryState) -> DiaryState:
        raise NotImplementedError("Subclasses must implement __call__ method")

    def _add_personal_preference(self):
        # check if the personal preference prompt is already in the prompt template
        if self.personal_preference_prompt and self.personal_preference_prompt not in self.prompt_template:
            self.prompt_template = self.personal_preference_prompt + "\n" + self.prompt_template
    
    def _call_llm(self, state: DiaryState, user_prompt: str, with_history: bool = True, filter_think: bool = True, format_json: bool = False, logfile_suffix = None) -> str:
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
        logname = "Diary-" + logname
        self.logger.log(logname, message, user_message=user_prompt, assistant_response=response)
        return response


class ContentExtractionAgent(BasicDiaryWriterAgent):
    def __init__(self, llm:OllamaLLM, 
                 system_prompt: str = GENERAL_SYSTEM,
                 prompt_template: str = CONTENT_EXTRACTION_PROMPT, 
                 personal_preference_prompt:str = PERSONAL_PREFERENCE_PROMPT,
                 ) -> None:
        super().__init__(llm, system_prompt, prompt_template, personal_preference_prompt="")
        self._add_personal_preference()

    def __call__(self, state: DiaryState) -> DiaryState:
        """
        Extract the content from the diary.
        """
        for i, chunk in enumerate(state["content_list"]):
            user_prompt = self.prompt_template.format(
                _Content=chunk,
                _Character=state["character"],
            )
            response = self._call_llm(state, user_prompt, with_history = False, logfile_suffix=f"scene-{i}")
            if len(response) < 10:
                continue
            state["extracted_content_list"].append(response)
        return state


class SplitSceneAgent(BasicDiaryWriterAgent):
    def __init__(self, llm:OllamaLLM, 
                 system_prompt: str = GENERAL_SYSTEM,
                 prompt_template: str = SPLIT_SCENE_PROMPT,
                 personal_preference_prompt:str = PERSONAL_PREFERENCE_PROMPT,
                 ) -> None:
        super().__init__(llm, system_prompt, prompt_template, personal_preference_prompt="")
        self._add_personal_preference()

        self.window_size = 16   # 16 lines

    def _call_llm(self, state: DiaryState, user_prompt: str, chunk: str, **kwargs) -> str:
        """
        override the _call_llm method to handle the window size
        """
        last_line = chunk.split("\n")[-1]
        prompt = self.prompt_template.format(
            _Content=chunk,
            _LastLine=last_line,
        )
        response = self.llm.invoke(
            [{"role": "system", "content": self.system_prompt},
             {"role": "user", "content": prompt},],
        )
        response = remove_think(response)
        response = strip_any_unnecessary_chars_for_json(response)
        return response

    def __call__(self, state: DiaryState) -> DiaryState:
        """
        Split the scene in t
        """
        content_lines = remove_empty_lines(state["content"], re_join=False)
        # 小模型太傻了，根本分不清楚场景切换
        # 大模型要运行的次数太多，运行一次就要收费。每一句话都要运行一次，是要死人的。
        # 所以，用经验法则来分割场景，就是按照“Chapter 1”这种格式来分割场景
        # Identify all lines containing "chapter" (case-insensitive)
        split_indices = [i for i, line in enumerate(content_lines) if "chapter" in line.lower()]
        # Always include the start and end
        split_indices = [0] + split_indices + [len(content_lines)]
        # Remove duplicates and sort
        split_indices = sorted(set(split_indices))
        state["split_line_indices"] = split_indices
        for i in range(len(state["split_line_indices"])-1):
            start = state["split_line_indices"][i]
            end = state["split_line_indices"][i+1]
            # check if the chunk is empty
            if start == end:
                continue
            # extract the content from the chunk
            chunk = content_lines[start:end]
            chunk = "\n".join(chunk)
            # call the LLM to extract the content
            state["content_list"].append(chunk)
        log_message = "\n#######\n".join(state["content_list"])
        self.logger.log("Diary-SplitScene", f"Splited into {len(state['content_list'])} scenes:", user_message="Split scene", assistant_response=log_message)
        return state

        # 以下的代码根本不work，不再运行它们。
        # using a sliding window to split the content into chunks, the step size is 1
        # and the window size is self.window_size
        for i in tqdm.tqdm(range(0, len(content_lines) - self.window_size + 1)):
            chunk = content_lines[i:i + self.window_size]
            chunk = "\n".join(chunk)
            # check if the chunk is empty
            if not chunk.strip():
                continue
            # check if the chunk is too short
            if len(chunk) < 10:
                continue
            # call the LLM to check if the scene is split
            response = self._call_llm(state, state["current_content"], chunk)
            # if there is a line with "reason", remove this line from the response
            print(response)
            if "reason" in response:
                response = response.split("\n")
                response = [line for line in response if "reason" not in line]
                response = "\n".join(response)
            response = json.loads(response)
            if response["is_split"]:
                # if the scene is split, add the index to the split_line_indices
                state["split_line_indices"].append(i+self.window_size-1)
        state["split_line_indices"].append(len(content_lines))
        for i in range(len(state["split_line_indices"])-1):
            start = state["split_line_indices"][i]
            end = state["split_line_indices"][i+1]
            # check if the chunk is empty
            if start == end:
                continue
            # extract the content from the chunk
            chunk = content_lines[start:end]
            chunk = "\n".join(chunk)
            # call the LLM to extract the content
            state["content_list"].append(chunk)
        log_message = "\n#######\n".join(state["content_list"])
        self.logger.log("Diary-SplitScene", f"Splited into {len(state['content_list'])} scenes:", user_message="Split scene", assistant_response=log_message)
        return state

class DiaryWritingAgent(BasicDiaryWriterAgent):
    def __init__(self, llm:OllamaLLM, 
                 system_prompt: str = WRITER_SYSTEM,
                 prompt_template: str = DIARY_PROMPT,
                 personal_preference_prompt:str = PERSONAL_PREFERENCE_PROMPT,
                 ) -> None:
        super().__init__(llm, system_prompt, prompt_template, personal_preference_prompt="")
        self._add_personal_preference()

    def __call__(self, state: DiaryState) -> DiaryState:
        """
        Write the diary.
        """
        for i, chunk in enumerate(state["extracted_content_list"]):
            if not chunk.strip() or len(chunk) < 10:
                continue
            previous_diary = NO_PREVOUS_DIARY_PROMPT if len(state["diaries"]) == 0 else "\n".join(state["diaries"][-3:])
            user_prompt = self.prompt_template.format(
                _Character=state["character"],
                _Date=state["date"],
                _Content=chunk,
                _CharacterDescription=state["character_description"],
                _PreviousDiary=previous_diary,
            )
            min_response_len = int(len(chunk) * 0.1)
            max_response_len = max(int(len(chunk) * 0.8), 1000)
            # call the LLM to write the diary
            response = self._call_llm(state, user_prompt, with_history=False, filter_think=True, format_json=False, logfile_suffix=f"diary-{i}")
            max_retry = 10
            while (len(response) < min_response_len or len(response) > max_response_len) and max_retry > 0:
                if len(response) > max_response_len:
                    user_prompt_1 = user_prompt + "\n" + "您刚才回复了“" + response + "”，但是这个回复超过了" + str(max_response_len) + "个字符。请您重新回复，确保回复的内容不超过" + str(max_response_len) + "个字符。请不要照抄原文，尤其注意所用语言与原文必须一致。"
                elif len(response) < min_response_len:
                    user_prompt_1 = user_prompt + "\n" + "您刚才回复了“" + response + "”，但是这个回复少于" + str(min_response_len) + "个字符。请您重新回复，确保回复的内容不少于" + str(min_response_len) + "个字符。"
                response = self._call_llm(state, user_prompt_1, with_history=False, filter_think=True, format_json=False, logfile_suffix=f"diary-{i}")
                max_retry -= 1
            if max_retry == 0 and len(response) < min_response_len:
                print(f"Warning: response is too short: {response}")
            elif max_retry == 0 and len(response) > max_response_len:
                print(f"Warning: response is too long: {response}")
            state["diaries"].append(response)
            state["current_diary"] = response

        return state


class EndDiaryAgent(BasicDiaryWriterAgent):
    def __init__(self, llm:OllamaLLM, 
                 system_prompt: str = WRITER_SYSTEM,
                 prompt_template: str = END_DIARY_PROMPT,
                 personal_preference_prompt:str = PERSONAL_PREFERENCE_PROMPT,
                 ) -> None:
        super().__init__(llm, system_prompt, prompt_template, personal_preference_prompt="")
        self._add_personal_preference()

    def __call__(self, state: DiaryState) -> DiaryState:
        """
        End the diary.
        """
        user_prompt = self.prompt_template.format(
            _Character=state["character"],
            _Date=state["date"],
            _Content="\n".join(state["diaries"]),
            _CharacterDescription=state["character_description"],
        )
        # call the LLM to end the diary
        response = self._call_llm(state, user_prompt, with_history=False, filter_think=True, format_json=False)
        state["diaries"].append(response)
        return state

def output_diary(state: DiaryState) -> DiaryState:
    """
    Output the story.
    """
    with open(diarypath, "w", encoding="utf-8") as f:
        f.write("\n".join(state["diaries"]))
    return state

def build_graph(planner_llm:BaseLLM, writer_llm:BaseLLM, critic_llm:BaseLLM, reviser_llm:BaseLLM) -> StateGraph:
    planner = planner_llm
    writer  = writer_llm
    critic  = critic_llm
    reviser = reviser_llm

    graph_builder = StateGraph(state_schema=DiaryState)
    graph_builder.add_node("initialize_state", initialize_state)
    graph_builder.add_node("split_scene", SplitSceneAgent(llm = planner))
    graph_builder.add_node("content_extraction", ContentExtractionAgent(llm = planner))
    graph_builder.add_node("diary_writing", DiaryWritingAgent(llm = writer))
    graph_builder.add_node("end_diary", EndDiaryAgent(llm = writer))
    graph_builder.add_node("output_diary", output_diary)
    # add the edges
    graph_builder.add_edge(START, "initialize_state")
    graph_builder.add_edge("initialize_state", "split_scene")
    graph_builder.add_edge("split_scene", "content_extraction")
    graph_builder.add_edge("content_extraction", "diary_writing")
    graph_builder.add_edge("diary_writing", "end_diary")
    graph_builder.add_edge("end_diary", "output_diary")
    graph_builder.add_edge("output_diary", END)
    
    return graph_builder    

if __name__ == "__main__":
    model = OllamaLLM(model="deepseek-r1:32b", temperature=0.2, max_tokens=128*1000)

    prompt_template = SUMMARY_ALL_PROMPT
    with open(r"C:\Users\fangn\Desktop\just_for_fun\AIStoryWriter-langgraph\stories\gensokyo-2025-05-15.txt", "r", encoding="utf-8") as f:
        content = f.read()
    user_prompt = prompt_template.format(
        _Content=content,
        _Character="古明地恋",
    )
    response = model.invoke(
        [{"role": "system", "content": GENERAL_SYSTEM},
            {"role": "user", "content": user_prompt},],
    )
    response = remove_think(response)
    with open(r"C:\Users\fangn\Desktop\just_for_fun\AIStoryWriter-langgraph\stories\gensokyo-2025-05-15-summary.txt", "w", encoding="utf-8") as f:
        f.write(response)
    quit()




if __name__ == "__main__":
    # benchmark, use larger model
    planner = OllamaLLM(model="deepseek-r1:32b",         temperature=0.2, max_tokens=12800)
    writer  = Together (model="deepseek-ai/DeepSeek-V3", temperature=0.7, max_tokens=16384)
    reviser = Together (model="deepseek-ai/DeepSeek-V3", temperature=0.2, max_tokens=16384)
    critic  = Together (model="deepseek-ai/DeepSeek-V3", temperature=0.2, max_tokens=12800)

    # debugging, use smaller model
    planner = OllamaLLM(model="glm4", temperature=0.2, max_tokens=8192)
    writer  = Together (model="deepseek-ai/DeepSeek-V3", temperature=0.7, max_tokens=16384)
    critic  = OllamaLLM(model="glm4", temperature=0.2, max_tokens=8192)
    reviser = OllamaLLM(model="glm4", temperature=0.2, max_tokens=8192)

    graph = build_graph(planner, writer, critic, reviser)
    graph = graph.compile()
    print(graph.get_graph().draw_mermaid())


    DIARYTITLE = "diary-2025-05-15"
    diarypath = "./stories/" + DIARYTITLE + ".txt"
    character = "古明地恋（Komeiji Koishi）（昵称为“恋”，“恋恋”）"
    with open("./prompts/古明地恋.txt", "r", encoding="utf-8") as f:
        character_description = f.read()
    current_date = "2025年5月15日"

    with open("./prompts/prompt_514.txt", "r", encoding="utf-8") as f:
        user_info = f.read()

    with open("./stories/gensokyo-2025-05-15.txt", "r", encoding="utf-8") as f:
        content = f.read()

    a = graph.invoke({"messages": [{"role": "user", "content": content}]}, config = {"recursion_limit": 200})
    # dump a as json
    import json
    with open(diarypath.replace(".txt", ".json"), "w", encoding="utf-8") as f:
        json.dump(a, f, ensure_ascii=False, indent=4)
