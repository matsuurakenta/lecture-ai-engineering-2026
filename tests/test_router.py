"""分類・既存ツール呼び出し・使用量合算を、外部APIなしで確認する。"""

import json

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage

from app.router import run_routed


@pytest.mark.parametrize("method,tool_name", [
    ("vector", "search_materials"),
    ("tree", "browse_tree"),
    ("graph", "search_graph"),
    ("hybrid", "search_materials"),
])
def test_route_calls_existing_tool_and_counts_both_stages(monkeypatch, method, tool_name):
    hit = {"page_id": "lec02-main-p052", "lecture": 2, "page": 52,
           "content": "元の重みは固定し、小さな行列だけ学習する"}
    searches = []

    def search(query):
        searches.append(query)
        return [hit]

    monkeypatch.setattr("app.retrieval.search", search)
    monkeypatch.setattr("app.retrieval.search_graph", lambda query: {"seeds": [hit], "expanded": []})
    requests = []

    def respond(messages, info):
        requests.append(info)
        if info.output_tools:
            part = ToolCallPart(info.output_tools[0].name,
                                {"method": method, "reason": "質問に合う範囲を検索"})
        elif len(requests) == 2:
            names = {tool.name for tool in info.function_tools}
            assert tool_name in names
            args = {} if tool_name == "browse_tree" else {"query": "LoRA"}
            part = ToolCallPart(tool_name, args)
        else:
            returns = [p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]
            assert any(p.tool_name == tool_name for p in returns)
            data = json.loads(returns[-1].content)
            if tool_name == "search_materials":
                assert data["results"] == [hit]
            elif tool_name == "search_graph":
                assert data["seeds"] == [hit]
            else:
                assert data["children"]  # 既存Treeの講義一覧を取得
            part = TextPart("回答")
        return ModelResponse(parts=[part], usage=RequestUsage(input_tokens=10, output_tokens=5))

    result = run_routed("LoRAとは？", FunctionModel(respond))
    assert result.decision.method == method
    assert result.answer_result.output == "回答"
    assert result.answer_result.usage.requests == 3  # 分類1回＋取得・回答2回
    assert result.answer_result.usage.input_tokens == 30
    assert result.answer_result.usage.output_tokens == 15
    if tool_name == "search_materials":
        assert searches == ["LoRA"]


def test_vector_can_expand_with_existing_graph(monkeypatch):
    monkeypatch.setattr("app.retrieval.search", lambda query: [])
    hit = {"page_id": "lec03-main-p040", "content": "全部は残さない"}
    monkeypatch.setattr("app.retrieval.search_graph", lambda query: {"seeds": [], "expanded": [hit]})
    calls = []

    def respond(messages, info):
        if info.output_tools:
            part = ToolCallPart(info.output_tools[0].name, {"method": "vector", "reason": "特定の事実"})
        elif not calls:
            calls.append("search_materials")
            part = ToolCallPart("search_materials", {"query": "日々のメモ"})
        elif len(calls) == 1:
            calls.append("search_graph")
            part = ToolCallPart("search_graph", {"query": "長期メモリ"})
        else:
            returns = [p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]
            assert json.loads(returns[-1].content)["expanded"] == [hit]
            part = TextPart("全部は残しません（第3回・PDF p.40）。")
        return ModelResponse(parts=[part])

    result = run_routed("日々のメモはすべて残りますか？", FunctionModel(respond))
    assert calls == ["search_materials", "search_graph"]
    assert "全部は残しません" in result.answer_result.output


def test_request_limit_includes_routing():
    def respond(messages, info):
        assert info.output_tools  # 分類後の回答リクエストは上限で止まる
        return ModelResponse(parts=[ToolCallPart(
            info.output_tools[0].name, {"method": "tree", "reason": "講義の概要"},
        )])

    with pytest.raises(UsageLimitExceeded):
        run_routed("第3回の概要は？", FunctionModel(respond), max_requests=1)


def test_reject_empty_question():
    with pytest.raises(ValueError, match="質問が空"):
        run_routed("  ", None)
