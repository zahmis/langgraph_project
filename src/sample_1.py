import os
from typing import Annotated

from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

os.environ["LANGCHAIN_TRACING_V2"] = "false"

class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")


def chatbot(state: State):
    print("受け取った状態:", state)
    response = llm.invoke(state["messages"])
    print("LLMからの応答:", response)
    return {"messages": [response]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")
graph = graph_builder.compile()


messages = [
    {"role": "user", "content": "こんにちは！"}
]

print("\n最初のメッセージ:", messages)
print("\nグラフを実行します...")

response = graph.invoke({
    "messages": messages
})

messages.append(response["messages"][-1])  # AIの応答を履歴に追加
messages.append({"role": "user", "content": "おっす！"})



response = graph.invoke({
    "messages": messages
})
