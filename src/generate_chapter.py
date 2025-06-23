from langchain_ollama import OllamaLLM
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
            _Summary = state["outline"],
        )
        response = self.llm.invoke(
            user_prompt
        )
        response = remove_think(response)
        response = strip_any_unnecessary_chars_for_json(response)
        try:
            n_chapters = json.loads(response)["TotalChapters"]
        except json.JSONDecodeError:
            n_chapters = get_number_result_anyway(response, "TotalChapters")

        state["chapters"] = [ChapterState(index=i) for i in range(n_chapters)]
        state["current_chapter_index"] = 0
        return state
    

class ChapterOutlineFilterAgent(BasicNovelWriterAgent):
    def __init__(self, llm: OllamaLLM, 
                 system_prompt: str = WRITER_SYSTEM,
                 prompt_template: str = CHAPTER_OUTLINE_FILTER_PROMPT,
                 personal_preference_prompt: str = PERSONAL_PREFERENCE_PROMPT,
                 ) -> None:
        super().__init__(llm, system_prompt, prompt_template, personal_preference_prompt)

    def __call__(self, state: NovelState) -> NovelState:
        cidx = state["current_chapter_index"]
        user_prompt = self.prompt_template.format(
            _ChapterNum = cidx + 1,
            _Outline = state["outline"],
            _Prompt = state["revised_user_context"],
            _ChapterOutline = state["chapters"][cidx]["outline"],
        )
        response = self._call_llm(state, user_prompt, with_history = False)
        chapter = state["chapters"][cidx]
        chapter: ChapterState
        chapter["outline"] = response
        return state


class ChapterOutlineAgent(BasicNovelWriterAgent):
    def __init__(self, llm: OllamaLLM, 
                 system_prompt: str = WRITER_SYSTEM,
                 prompt_template: str = CHAPTER_OUTLINE_PROMPT,
                 personal_preference_prompt: str = PERSONAL_PREFERENCE_PROMPT,
                 min_response_length: int = 200,
                 max_retry: int = 10,
                 ) -> None:
        super().__init__(llm, system_prompt, prompt_template, personal_preference_prompt)
        self.min_response_length = min_response_length
        self.max_retry = max_retry

    def __call__(self, state: NovelState) -> NovelState:
        self._add_personal_preference()
        cidx = state["current_chapter_index"]
        chapter = state["chapters"][cidx]
        chapter: ChapterState 
        user_prompt = self.prompt_template.format(
            _ChapterNum = cidx + 1,
            _Outline = state["outline"],
            _Prompt = state["revised_user_context"],
        )
        response = self._call_llm(state, user_prompt, logfile_suffix = f"chapter-{cidx+1}", with_history = False)
        for i in range(self.max_retry):
            if len(response) >= self.min_response_length:
                break
            print(f"Retrying chapter outline generation for chapter {cidx+1}, response too short: {len(response)} < {self.min_response_length}")
            user_prompt_1 = user_prompt + INSUFFICIENT_LENGTH_PROMPT.format(
                _Length = self.min_response_length,
            )
            response = self._call_llm(state, user_prompt_1, logfile_suffix = f"chapter-{cidx+1}-retry-{i+1}", with_history = False)
        else:
            print(f"max retry reached for chapter {cidx+1}, response too short: {len(response)} < {self.min_response_length}")
        chapter["outline"] = response
        return state
    
def switch_chapter_outline(state: NovelState) -> NovelState:
    """
    Switch to the next chapter outline.
    """
    state["current_chapter_index"] += 1
    return state

def is_last_chapter_outline(state: NovelState) -> bool:
    """
    Check if the current chapter is the last chapter.
    """
    cidx = state["current_chapter_index"]
    return cidx >= len(state["chapters"])

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
            chapter = state["chapters"][0]
            chapter: ChapterState
            chapter["last_cohesion_requirement"] = "This is the first chapter, no previous chapter to summarize."
            return state
        cidx = state["current_chapter_index"]
        user_prompt = self.prompt_template.format(
            _ChapterNum     = cidx + 1,
            _LastChapterNum = cidx + 1 - 1,
            _TotalChapters  = len(state["chapters"]),
            _LastChapter    = state["chapters"][cidx - 1]["content"],
            _Outline        = state["outline"],
            _ThisOutline    = state["chapters"][cidx]["outline"],
        )
        response = self._call_llm(state, user_prompt, with_history = False)
        current_chapter = state["chapters"][cidx]
        current_chapter: ChapterState
        current_chapter["last_cohesion_requirement"] = response
        return state
    
class NextChapterConnectAgent(BasicNovelWriterAgent):
    def __init__(self, llm: OllamaLLM, 
                 system_prompt: str = WRITER_SYSTEM,
                 prompt_template: str = NEXT_CHAPTER_CONNECT_PROMPT,
                 personal_preference_prompt: str = PERSONAL_PREFERENCE_PROMPT,
                 ) -> None:
        super().__init__(llm, system_prompt, prompt_template, personal_preference_prompt)

    def __call__(self, state: NovelState) -> NovelState:
        cidx = state["current_chapter_index"]
        if cidx >= len(state["chapters"]) - 1:
            state["chapters"][cidx]["next_cohesion_requirement"] = "This is the last chapter, no next chapter to connect."
            return state
        user_prompt = self.prompt_template.format(
            _ChapterNum = cidx + 1,
            _NextChapterNum = cidx + 2,
            _Outline = state["outline"],
            _ThisOutline = state["chapters"][cidx]["outline"],
            _NextOutline = state["chapters"][cidx + 1]["outline"],
        )
        response = self._call_llm(state, user_prompt, with_history = False)
        current_chapter = state["chapters"][cidx]
        current_chapter: ChapterState
        current_chapter["next_cohesion_requirement"] = response
        return state

def switch_chapter(state: NovelState) -> NovelState:
    """
    Switch to the next chapter. 
    """
    cidx = state["current_chapter_index"]
    chapter = state["chapters"][cidx]

    chapter: ChapterState
    chapter["outline_revision_count"] = 0
    chapter["outline_complete"] = True 
    chapter["content_revision_count"] = 0
    chapter["current_feedback"] = "" 

    print(f"Current chapter: Chapter {cidx + 1}")
    print(f"Total chapters: {len(state['chapters'])}")
    if cidx < len(state["chapters"]):
        state["current_chapter_index"] += 1

    # working_dir = state["working_dir"]
    # if not os.path.exists(working_dir):
    #     os.makedirs(working_dir)
    # with open(os.path.join(working_dir, f"chapter-{cidx + 1}_characters.md"), "w", encoding="utf-8") as f:
    #     for char in state.characters:
    #         char: Character
    #         f.write(char['description']+"\n\n")
    #     for char in chapter.characters:
    #         char: Character
    #         f.write(char['description']+"\n\n")

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
        cidx = state["current_chapter_index"]
        chapter = state["chapters"][cidx]
        chapter: ChapterState
        if len(chapter["content"]) > self.max_response_length:
            print(f"Chapter {cidx + 1} content is too long, skipping generation.")
            return

        last_chapter_content = state["chapters"][cidx - 1]["content"] if cidx > 0 else "This is the first chapter, no previous chapter to summarize."

        character_list = chapter.get_characters()
        story_settings = state["story_settings"]
        user_contextss = state["revised_user_context"]
        book_outline = state["outline"]
        # background = "\n".join([user_contextss, story_settings, character_list])
        background = "\n".join([story_settings, character_list, book_outline])

        user_prompt = prompt_template.format(
            _ChapterNum=state["current_chapter_index"] + 1,
            _TotalChapters      = len(state["chapters"]),
            _LastChapterSummary = chapter["last_cohesion_requirement"],
            _LastChapter        = last_chapter_content,
            _LastChapterNum     = cidx + 1 - 1,
            _NextChapterSummary = chapter["next_cohesion_requirement"],
            _NextChapterNum     = cidx + 1 + 1,
            _Outline            = chapter["outline"],
            _Background         = background,
            _Chapter            = chapter["content"],
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
        chapter["content"] = response

    def __call__(self, state: NovelState) -> NovelState:
        self._round(state, CHAPTER_GENERATION_STAGE1_PROMPT)
        self._round(state, CHAPTER_GENERATION_STAGE2_PROMPT)
        # self._round(state, CHAPTER_GENERATION_STAGE3_PROMPT)
        self._round(state, CHAPTER_GENERATION_STAGE4_PROMPT)
        self.min_response_length = self._min_response_length
        self.max_response_length = self._max_response_length

        return state

class CritiqueChapterCompleteAgent(BasicNovelWriterAgent):
    def __init__(
        self,
        llm: OllamaLLM,
        system_prompt: str = CRITIQUE_SYSTEM,
        prompt_template: str =  CRITIC_CHAPTER_COMPLETE_PROMPT,
        personal_preference_prompt: str = PERSONAL_PREFERENCE_PROMPT,
    ) -> None:
        super().__init__(llm, system_prompt, prompt_template, personal_preference_prompt)

    def __call__(self, state: NovelState) -> NovelState:
        self._add_personal_preference()
        user_prompt = self.prompt_template.format(
            _Prompt=state["revised_user_context"],
            _Outline=state["outline"],
            _ChapterOutline=state["chapters"][state["current_chapter_index"]]["outline"],
            _Chapter=state["chapters"][state["current_chapter_index"]]["content"],
            _ChapterNum=state["current_chapter_index"] + 1,
            _LastChapterNum=state["current_chapter_index"] + 1 - 1,
            _NextChapterNum=state["current_chapter_index"] + 1 + 1
        )
        response = self._call_llm(state, user_prompt)
        is_complete = get_boolean_result_anyway(response, "IsComplete")
        state["chapters"][state["current_chapter_index"]]["outline_complete"] = is_complete
        return state


class CritiqueChapterGeneralAgent(BasicNovelWriterAgent):
    def __init__(
        self,
        llm: OllamaLLM,
        system_prompt: str = CRITIQUE_SYSTEM,
        prompt_template: str =  CRITIC_CHAPTER_GENERAL_PROMPT,
        personal_preference_prompt: str = PERSONAL_PREFERENCE_PROMPT,
    ) -> None:
        super().__init__(llm, system_prompt, prompt_template, personal_preference_prompt)

    def __call__(self, state: NovelState) -> NovelState:
        self._add_personal_preference()
        user_prompt = self.prompt_template.format(
            _Prompt=state["revised_user_context"],
            _Outline=state["outline"],
            _ChapterOutline=state["chapters"][state["current_chapter_index"]]["outline"],
            _Chapter=state["chapters"][state["current_chapter_index"]]["content"],
            _ChapterNum=state["current_chapter_index"] + 1,
            _LastChapterNum=state["current_chapter_index"] + 1 - 1,
            _NextChapterNum=state["current_chapter_index"] + 1 + 1
        )
        response = self._call_llm(state, user_prompt)
        state["chapters"][state["current_chapter_index"]]["current_feedback"] = response
        return state

class CritiqueChapterScoreAgent(BasicNovelWriterAgent):
    def __init__(
        self,
        llm: OllamaLLM,
        system_prompt: str = CRITIQUE_SYSTEM,
        prompt_template: str =  CRITIC_CHAPTER_SCORE_PROMPT,
        personal_preference_prompt: str = PERSONAL_PREFERENCE_PROMPT,
    ) -> None:
        super().__init__(llm, system_prompt, prompt_template, personal_preference_prompt)

    def __call__(self, state: NovelState) -> NovelState:
        self._add_personal_preference()
        user_prompt = self.prompt_template.format(
            _Prompt=state["revised_user_context"],
            _Outline=state["outline"],
            _ChapterOutline=state["chapters"][state["current_chapter_index"]]["outline"],
            _Chapter=state["chapters"][state["current_chapter_index"]]["content"],
            _ChapterNum=state["current_chapter_index"] + 1,
            _LastChapterNum=state["current_chapter_index"] + 1 - 1,
            _NextChapterNum=state["current_chapter_index"] + 1 + 1
        )
        response = self._call_llm(state, user_prompt)
        score = get_chapter_score(response)
        state["chapters"][state["current_chapter_index"]]["chapter_score"] = score
        state["chapters"][state["current_chapter_index"]]["outline_complete"] = (score >= 6)
        return state


class ChapterRevisionAgent(BasicNovelWriterAgent):
    def __init__(
        self,
        llm: OllamaLLM,
        system_prompt: str = WRITER_SYSTEM,
        prompt_template: str = CHAPTER_REVISION_PROMPT,
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

    def __call__(self, state: NovelState) -> NovelState:
        self._add_personal_preference()
        cidx = state["current_chapter_index"]
        chapter = state["chapters"][cidx]
        user_prompt = self.prompt_template.format(
            _Prompt=state["revised_user_context"],
            _Outline=state["outline"],
            _ChapterOutline=chapter["outline"],
            _Chapter=chapter["content"],
            _ChapterNum=cidx + 1,
            _LastChapterNum=cidx + 1 - 1,
            _NextChapterNum=cidx + 1 + 1,
            _LastChapterSummary=chapter["last_cohesion_requirement"],
            _NextChapterSummary=chapter["next_cohesion_requirement"],
            _Feedback=chapter["current_feedback"],
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
        
        if self.last_response_length > int(self.last_response_length * 1.25):
            self.min_response_length = min(int(self.last_response_length * 0.8), self.max_response_length * 0.8)
        chapter["content"] = response
        chapter["content_revision_count"] += 1
        
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



def is_max_chapter_review_reached(state: NovelState) -> bool:
    """
    Check if the number of chapter reviews is less than the maximum number of chapter reviews.
    """
    chapter = state["chapters"][state["current_chapter_index"]]
    chapter: ChapterState
    if chapter["content_revision_count"] >= chapter["content_max_revisions"]:
        return True
    return False

def is_chapter_complete(state: NovelState) -> bool:
    """
    Check if the chapter is complete.
    """
    chapter = state["chapters"][state["current_chapter_index"]]
    chapter: ChapterState
    if chapter["outline_complete"]:
        return True
    return False

def is_last_chapter(state: NovelState) -> bool:
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
    storypath = state["output_dir"]
    dirname = os.path.dirname(storypath)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(storypath, "w", encoding="utf-8") as f:
        for i in range(len(state["chapters"])):
            f.write(f"Chapter {i+1}:\n")
            f.write(state["chapters"][i]["content"])
            f.write("\n\n")
    return state
