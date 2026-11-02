# 用語集の LoRA 学習・評価データ

`glossary_sft.jsonl` の58用語を対象に、同じ用語の定義を尋ねる質問を10通りずつ用意し、用語ごとに **train / valid / test = 80% / 10% / 10%** に分割したデータです。学習量を確保しながら、全用語について検証と最終評価を行える割合にしています。

| ファイル | 件数 | 用途・形式 |
|---|---:|---|
| `train.jsonl` | 464（80%） | 学習用。`term_id` と `messages`（user / assistant） |
| `valid.jsonl` | 58（10%） | 検証用。train と同じ形式。損失の確認や設定の選択に使う |
| `test.jsonl` | 58（10%） | 最終評価用。`term_id` と `question` のみ |
| `gold.json` | 58 | test の質問文をキーとする正解。`term_id`・`answer`・`must_include` |

分割対象は合計580件です。`gold.json` は test の正解を別に保持するファイルなので、分割件数には加算しません。

## 作成方針

- 元データの116問（各用語2問）は、質問と回答をそのまま train に収録しています。各用語に学習用の言い換え6問、検証用1問、テスト用1問を追加しました。
- 全58用語が、各分割に必ず8問・1問・1問ずつ含まれます。用語単位では分けず、学習した用語について未見の聞き方に答えられるかを評価します。
- 検証・テスト用の質問は、用語に合わせた聞き方や別名で作成しています。各分割の内外に質問の重複がないことを、Unicode NFKC 正規化・大小文字の統一・空白除去後にも確認しています。
- 分割先は固定し、各ファイルの行順だけを seed 42 の乱数でシャッフルしています。
- 回答は元データの定義文をそのまま使い、`data/glossary.json` の定義と一致することも確認しています。test の正解は `gold.json` のみに保存しています。
- valid は設定選択に使い、test と gold は設定を決めた後の最終評価に使います。test の結果に合わせて設定やプロンプトを調整すると、最終評価としての独立性が失われます。

## Colab での読み込み

4つのデータファイルを同じディレクトリにアップロードして読み込みます。以下はカレントディレクトリに置いた場合です。リポジトリのルートで実行する場合は `DATA = Path("lectures/02/advance")` に変更してください。

```python
import json
from pathlib import Path
from datasets import Dataset

DATA = Path(".")

def read_jsonl(name):
    return [json.loads(line) for line in (DATA / name).read_text(encoding="utf-8").splitlines() if line.strip()]

train = read_jsonl("train.jsonl")
valid = read_jsonl("valid.jsonl")
test = read_jsonl("test.jsonl")
gold = json.loads((DATA / "gold.json").read_text(encoding="utf-8"))

# 元ノートブックの to_chat による変換は不要です。
train_ds = Dataset.from_list([{"messages": row["messages"]} for row in train])
valid_ds = Dataset.from_list([{"messages": row["messages"]} for row in valid])
```

`train_ds` を `SFTTrainer` の `train_dataset`、`valid_ds` を `eval_dataset` に渡します。

## 評価方法

このデータの学習対象は、元の `glossary_sft.jsonl` と同じ**通常の文章による用語説明**です。元ノートブックの FAQ 本文を含む入力や、`answer` / `needs_review` の JSON 出力を求める指示とは形式が異なります。`build_messages` と評価処理を以下に置き換え、学習時と同じ user の質問だけをモデルに渡してください。gold や定義文は入力に含めません。

以下は、元ノートブックの `generate(model, example)` が定義された後に使う例です。元の `evaluate` の代わりに `evaluate_glossary` を呼びます。

```python
import unicodedata

def build_messages(example):
    return [{"role": "user", "content": example["question"]}]

def normalize(text):
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())

def evaluate_glossary(model):
    rows = []
    for example in test:
        raw = generate(model, example)
        expected = gold[example["question"]]
        ok = all(normalize(word) in normalize(raw) for word in expected["must_include"])
        rows.append({"question": example["question"], "raw": raw,
                     "reference": expected["answer"], "content_ok": ok})
    return {"n": len(rows), "content_rate": sum(row["content_ok"] for row in rows) / len(rows)}, rows

# LoRA 学習前に実行
before_summary, before_rows = evaluate_glossary(model)
# LoRA 学習後に実行（tuned は学習済みモデル）
# after_summary, after_rows = evaluate_glossary(tuned)
```

`must_include` は内容を簡易確認するための必須語句です。意味が合っていても別の表現なら不合格になり、語句が含まれていても説明が誤っている可能性があります。`answer` の正解文と生成結果を読み比べて確認してください。JSON 形式遵守率は、この文章回答の評価には含めません。

今回は全問が用語集にある既知の用語です。用語集にない質問への確認要求や、未知の用語への回答能力は評価対象に含まれません。
