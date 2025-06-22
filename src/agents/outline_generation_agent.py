from langgraph.prebuilt import create_react_agent
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

from ..prompts import *


llm = OllamaLLM(model="qwen3:32b", temperature=0.7, max_tokens=8192)



def get_weather(city: str) -> str:  
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model=llm,
    tools=[],
    prompt = WRITER_SYSTEM,
)

response = agent.invoke({"messages": [{"role": "user", "content": "what is the weather in sf"}]})


agent = create_react_agent(
    model=llm,
    tools=[],
    prompt = WRITER_SYSTEM,
)

base_info_prompt = PromptTemplate.from_template(GET_IMPORTANT_BASE_INFO_PROMPT, template_format="f-string")
with open(r"C:\Users\fangn\Desktop\just_for_fun\AIStoryWriter-langgraph\prompt_4.txt", "r", encoding="utf-8") as f:
    user_info = f.read()

prompt = base_info_prompt.format(_Prompt=user_info)
response = agent.invoke(
    {"messages": [{"role": "user", "content": prompt}]},
)