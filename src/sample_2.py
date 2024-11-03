import getpass
import os
from typing import Annotated

from IPython.display import Image, display
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

os.environ["LANGCHAIN_TRACING_V2"] = "false"

# 環境変数が設定されていない場合にユーザーに入力を求める関数
def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")
# 必要なAPI keyの設定
_set_env("ANTHROPIC_API_KEY")
_set_env("TAVILY_API_KEY")

# 会話の状態を定義するクラス
class State(TypedDict):
    messages: Annotated[list, add_messages]

# グラフビルダーの初期化
graph_builder = StateGraph(State)

# 検索ツールの設定
tool = TavilySearchResults(max_results=2)
tools = [tool]

# LLMの初期化とツールの接続
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
# LLMに検索ツールを接続
llm_with_tools = llm.bind_tools(tools)

# チャットボット関数の定義
def chatbot(state: State):
    # 現在の会話履歴から応答を生成
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# グラフにノードを追加
graph_builder.add_node("chatbot", chatbot) # チャットボットノードを追加
tool_node = ToolNode(tools=[tool]) 
graph_builder.add_node("tools", tool_node) # ツールノードを追加

# ノード間の接続を設定
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition, # ツールを呼び出す条件分岐
)
graph_builder.add_edge("tools", "chatbot") #ツール実行後はチャットボットノードに遷移
graph_builder.add_edge(START, "chatbot") # 開始ノードからチャットボットノードに遷移

# メモリ設定とグラフのコンパイル
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)


# グラフの描画（Mermaid形式）
try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    pass

# チャットボットの実行設定
config = {"configurable": {"thread_id": "1"}} # スレッドIDを設定
user_input = "Hi there!, My name is wai." # ユーザーの入力

# グラフの実行
events = graph.stream(
    {"messages": [("user", user_input)]}, #初期メッセージ
    config,
    stream_mode="values",
)

# 応答の表示
for event in events:
    event["messages"][-1].pretty_print()
