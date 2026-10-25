"""第1回: 素のLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --interactive  # 会話履歴を保って続けて質問する

通常モードでは会話履歴を持たない「素の」呼び出し。
対話モードでは、同じチャットセッション内で会話履歴を保つ。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
import time

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(question: str) -> str:
    client = genai.Client(api_key=get_api_key())
    started_at = time.perf_counter()
    response = client.models.generate_content(model=MODEL, contents=question)
    print_usage(response, time.perf_counter() - started_at)
    return response.text or ""


def print_usage(response, elapsed_seconds: float) -> None:
    usage = getattr(response, "usage_metadata", None)
    if usage is not None:
        prompt_tokens = getattr(usage, "prompt_token_count", 0)
        completion_tokens = getattr(usage, "candidates_token_count", 0)
        total_tokens = getattr(usage, "total_token_count", 0)
        print(f"[トークン数] 入力: {prompt_tokens}, 出力: {completion_tokens}, 合計: {total_tokens}")
    print(f"[回答時間] {elapsed_seconds:.2f}秒")


def interactive_chat(first_question: str | None = None) -> None:
    client = genai.Client(api_key=get_api_key())
    chat = client.chats.create(model=MODEL)

    print("対話モードです。終了するには /exit を入力するか、Ctrl+D を押してください。")
    question = first_question
    while True:
        if question is None:
            try:
                question = input("質問> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n会話を終了します。")
                return

        if question.lower() in {"/exit", "exit", "quit"}:
            print("会話を終了します。")
            return
        if not question:
            question = None
            continue

        print(f"[質問] {question}\n")
        started_at = time.perf_counter()
        response = chat.send_message(question)
        print_usage(response, time.perf_counter() - started_at)
        print(response.text or "")
        print()
        question = None


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", help="モデルに送る質問")
    parser.add_argument(
        "-i", "--interactive", action="store_true", help="会話履歴を保って続けて質問する"
    )
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    try:
        if args.interactive:
            interactive_chat(args.question)
        else:
            question = args.question or DEFAULT_QUESTION
            print(f"[質問] {question}\n")
            print(ask(question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
