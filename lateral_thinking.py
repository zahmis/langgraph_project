import os
import warnings
from typing import Annotated

from anthropic import Anthropic
from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

warnings.filterwarnings("ignore")

# LangChainのトレーシングを無効化
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_PROJECT"] = ""  # プロジェクト名を空に
os.environ["LANGCHAIN_ENDPOINT"] = ""  # エンドポイントを空に
os.environ["LANGCHAIN_API_KEY"] = ""  # APIキーを空に


TOTAL_COST= 0.0

class State(TypedDict):
    messages: Annotated[list, add_messages]
    quiz: str
    quality_score: float

graph_builder = StateGraph(State)
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")

def calculate_cost(input_tokens, output_tokens):
    # Claude 3.5 Sonnetの料金（2024年3月時点）
    input_cost_per_1k = 0.003  # $0.003 per 1K input tokens
    output_cost_per_1k = 0.015 # $0.015 per 1K output tokens
    
    cost = (input_tokens * input_cost_per_1k / 1000) + (output_tokens * output_cost_per_1k / 1000)
    return cost

def count_tokens(text: str) -> int:
    client = Anthropic()
    return client.count_tokens(text)

def calculate_usage(state: State, prompt: str, response: str):
    global TOTAL_COST
    input_tokens = count_tokens(prompt)
    output_tokens = count_tokens(response)
    cost = calculate_cost(input_tokens, output_tokens)
    
    TOTAL_COST += cost
    
    print("\n=== トークン使用量 ===")
    print(f"入力トークン: {input_tokens}")
    print(f"出力トークン: {output_tokens}")
    print(f"推定料金: ${cost:.4f}")
    
    return input_tokens, output_tokens, cost

def quiz_generator(state: State):
    prompt = """
    水平思考クイズを1つ作成してください。以下の要素を含めてください：
    - 謎の状況説明
    - 解答
    - 解説
    
    クイズは論理的で解けるものにしてください。
    """
    response = llm.invoke([{
        "role": "user",
        "content": prompt
    }])
    calculate_usage(state, prompt, response.content)
    print("\n=== 生成されたクイズ ===")
    print(response.content)
    return {"quiz": response.content}

def quality_checker(state: State):
    check_prompt = f"""
    以下の水平思考クイズの品質を評価してください。
    各項目を0-10点で評価し、合計点を100点満点に換算してください。

    評価対象のクイズ：
    {state["quiz"]}

    評価項目：
    1. 論理性：解答に至る過程が論理的か
    2. オリジナリティ：既存のクイズと似ていないか
    3. 解きやすさ：ヒントから解答にたどり着けるか
    4. 面白さ：解答を聞いたときの驚きや納得感

    【重要】各項目の点数と合計点を明確に記載してください。
    回答形式：
    論理性: X/10
    オリジナリティ: X/10
    解きやすさ: X/10
    面白さ: X/10
    合計: XX/100
    """
    response = llm.invoke([{
        "role": "user",
        "content": check_prompt
    }])
    print(101,response)
    calculate_usage(state, check_prompt, response.content)

    try:
        # 応答内容から合計スコアを抽出
        content = response.content
        for line in content.split('\n'):
            if '合計:' in line or '合計：' in line:
                # 数字を抽出 (例: "合計: 70/100" から "70" を取得)
                score = float(line.split('/')[0].split(':')[1].strip())
                print(f"\n=== 品質チェックスコア: {score}点 ===")
                if score < 80:
                    print(f"目標スコア(80点)まであと: {80 - score:.1f}点")
                return {"quality_score": score}
        
        # 合計スコアが見つからない場合
        raise ValueError("合計スコアが見つかりませんでした")
        
    except (ValueError, IndexError) as e:
        print(f"\n=== スコア解析エラー: {str(e)} ===")
        print("=== デフォルトスコア70を使用 ===")
        return {"quality_score": 70.0}

def quiz_improver(state: State):
    if state["quality_score"] < 80:
        print("\n=== クイズの改善を開始 ===")
        improve_prompt = f"""
        以下の水平思考クイズを改善してください：
        {state["quiz"]}

        現在の品質スコア: {state["quality_score"]}
        特に論理性と解きやすさを重視して改善してください。
        """
        response = llm.invoke([{
            "role": "user",
            "content": improve_prompt
        }])

        calculate_usage(state, improve_prompt, response.content)

        print("\n=== 改善されたクイズ ===")
        print(response.content)
        return {"quiz": response.content}
    print("\n=== 十分な品質が達成されました ===")
    return {"quiz": state["quiz"]}

# グラフの構築
graph_builder.add_node("generator", quiz_generator)
graph_builder.add_node("checker", quality_checker)
graph_builder.add_node("improver", quiz_improver)

graph_builder.set_entry_point("generator") 

graph_builder.add_edge("generator", "checker")
graph_builder.add_edge("improver", "checker")

def should_continue(state: State):
    # スコアが80未満の場合はimproverへ、80以上の場合はendへ
    return "improver" if state["quality_score"] < 80 else "end"

graph_builder.add_conditional_edges(
    "checker",
    should_continue,
    {
        "improver": "improver",
        "end": END,
    }
)

graph = graph_builder.compile()

print("\n=== 水平思考クイズ生成開始 ===")
result = graph.invoke({
    "messages": [],
    "quiz": "",
    "quality_score": 0
})
print("\n=== 最終的なクイズ ===")
print(result["quiz"])
print("\n=== 使用料金: ${TOTAL_COST: .4f} ===")