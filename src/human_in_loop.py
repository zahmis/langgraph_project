import getpass
import os
from typing import Annotated

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "false"

# 環境変数が設定されていない場合にユーザーに入力を求める関数
def _set_env(var: str):
    if var not in os.environ or not os.environ[var]:
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

# ツール実行前の人間の介入ポイント
def human_intervention(state: State):
    """ツール実行前の人間の介入ポイント"""
    print("\n=== ツール実行前の確認 ===")
    last_message = state['messages'][-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        tool_call = last_message.tool_calls[0]
        try:
            # tool_callの構造をデバッグ出力
            print("Tool Call Structure:", tool_call)
            
            # 異なる方法でクエリにアクセス
            if isinstance(tool_call, dict):
                query = tool_call.get('args', {}).get('query', '')
            else:
                query = tool_call.function.arguments.get('query', '')
                
            print(f"予定されている検索クエリ: {query}")        
            response = input("このまま実行しますか? (y/n): ")
            if response.lower() != 'y':
                modified_query = input("修正したクエリを入力してください: ")
                # クエリを修正して状態を更新
                if isinstance(tool_call, dict):
                    tool_call['args']['query'] = modified_query
                else:
                    tool_call.function.arguments['query'] = modified_query
        except Exception as e:
            print(f"Error processing tool call: {e}")
            print(f"Tool call structure: {tool_call}")
    
    return state

# グラフにノードを追加
graph_builder.add_node("chatbot", chatbot) # チャットボットノードを追加
graph_builder.add_node("human_intervention", human_intervention)
graph_builder.add_node("tools", ToolNode(tools=[tool]) ) # ツールノードを追加

# ノード間の接続を設定
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition, # ツールを呼び出す条件分岐
    {
        "tools": "human_intervention",  # ツールが必要な場合は人間の介入へ
        END: END                        # ツールが不要な場合は終了
    }
)
graph_builder.add_edge("human_intervention", "tools")
graph_builder.add_edge("tools", "chatbot") #ツール実行後はチャットボットノードに遷移

graph_builder.set_entry_point("chatbot")

# メモリ設定とグラフのコンパイル
memory = MemorySaver()
graph = graph_builder.compile(
    checkpointer=memory,
    interrupt_before=["tools"],
    )



user_input = "I'm learning LangGraph. Could you do some research on it for me?"
config = {"configurable": {"thread_id": "1"}}


# The config is the **second positional argument** to stream() or invoke()!
events = graph.stream(
    {"messages": [("user", user_input)]}, config, stream_mode="values"
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
        snapshot = graph.get_state(config)
        print(snapshot.next)
        print(snapshot.values['messages'][-1])


