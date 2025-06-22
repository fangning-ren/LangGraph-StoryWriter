from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, MessageGraph, MessagesState, START, END
from langgraph.graph.message import add_messages


from langgraph.types import RetryPolicy, default_retry_on 
from typing import Optional, Sequence, Any, Union, Callable






def short_generation_retry(
    max_retries: int = 3,
    delay: Union[int, float] = 1,
    retry_on: Optional[Union[Sequence[RetryPolicy], Callable[[Exception], bool]]] = None,
) -> Sequence[RetryPolicy]:
    """Create a retry policy for short generation tasks."""
    return [
        RetryPolicy(
            max_retries=max_retries,
            delay=delay,
            retry_on=retry_on or default_retry_on,
        )
    ]