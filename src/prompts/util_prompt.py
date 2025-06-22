GENERAL_SYSTEM = "你是一个智能助手，擅长处理各种任务和问题。请根据用户的需求提供准确和有用的信息。"

WRITER_SYSTEM = "你是一个小说作家，擅长创作引人入胜的故事情节和角色发展。正在创作一个精彩的新故事，并希望创作出高质量的作品。你需要依靠接下来提供的信息完成创作。"

CRITIQUE_SYSTEM = "你是一个小说评论家，擅长分析和评估小说的情节、角色和写作风格。你正在阅读一本新书，并希望提供深入的评论和反馈。你需要依靠接下来提供的章节或大纲完成批判性评论，并给出具体的，有建设性的建议。"


CHAPTER_COUNT_PROMPT = """
请帮我从以下大纲中获取章节数。即判断这个小说会有多少章。请以JSON格式返回章节数。

<OUTLINE>
{_Summary}
</OUTLINE>

请提供 JSON 格式的回复，其中包含上述大纲中的章节总数。

回复内容为 {{"TotalChapters": <total chapter count>}}
请不要包含任何其他文本，仅包含 JSON，这很重要。因为您的回复将由计算机解析。
"""

CHAPTER_EXTRACTION_PROMPT = """
请帮我提取此大纲中仅属于第 {_ChapterNum} 章节的部分.

<OUTLINE>
{_Outline}
</OUTLINE>

除了第{_ChapterNum}章所属的大纲内容外，不要在回复中包含任何其他内容 .
"""

INSUFFICIENT_LENGTH_PROMPT = """
您上一次回复的内容太少了。请确保您提供的内容足够详细和全面。请至少回复 {_Length} 个字符。
"""

OVER_LENGTH_PROMPT = """
您上一次回复的内容太多了。请尝试删除重复的内容或不必要的细节。请确保您的回复在 {_Length} 个字符以内。
"""


CHAPTER_SCRUB_PROMPT = """
你是一个有帮助的AI助手，擅长整理和编辑文本。你现在正在分块处理一个小说的内容。

这个小说的有一部分是由AI生成的，可能会包含一些不必要的内容，比如提示词模板的一部分、解释说明、指令、markdown形式的修改等。

如果你在以下文段中遇到这些并非小说的内容，将其移除。其他内容请**原封不动**地保留。

如果以下文本不含有这些内容，请将其**原封不动**地保留。

所需要处理的文本如下：
<CHAPTER>
{_Chapter}
</CHAPTER>

请勿包含<CHAPTER>和</CHAPTER>标签。请不要再添加任何解释或额外的内容。只能返回处理后的文本内容。

"""



JSON_PARSE_ERROR = "Please revise your JSON. It encountered the following error during parsing: {_Error}. Remember that your entire response is plugged directly into a JSON parser, so don't write **anything** except pure json."

PERSONAL_PREFERENCE_PROMPT = "" #"多加抓住女角色脚踝，或脚铐锁住女主脚踝的情节，详细描写每次抓脚的情景和角色感受。"

