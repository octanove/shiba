# SHIBAとは

SHIBA は、日本語 Wikipedia コーパスを用いて事前学習した CANINE[[1]](#1) モデルの PyTorch 再実装です。
CANINE をご存知なければ、非常に高効率な文字レベル BERT モデルだと考えてください。もちろん、SHIBA という名前は日本の canine (犬)である柴犬に由来しています。

![CANINE/SHIBA Architecture](canine_architecture.png "CANINE/SHIBA Architecture")

SHIBA の最大のメリットは、CANINE と同様、以下の2つです：

1. 語彙の制限がなく、あらゆるユニコード文字を処理できること。事前学習中にモデルが観察したことのない文字、単語、言語でもファインチューニングできます。
2. 効率よく多くの文字を処理できること。文字レベル BERT に比べると、同等の計算量で4倍（2048文字) の文字を埋め込むことができます。

また、2つの下流タスクにおける性能も良好です。

# 性能 

1つ目の下流タスクは、モデルが一度に処理できる量のテキストを使った[livedoorニュースコーパス](https://www.rondhuit.com/download.html)の分類です。

| モデル | 精度 |
|---|---|
| SHIBA | 95.5% |
| [bert-base-japanese](https://huggingface.co/cl-tohoku/bert-base-japanese) | 95.1% |
| [bert-base-japanese-char](https://huggingface.co/cl-tohoku/bert-base-japanese-char) |  92.9% |

2件目の下流タスクは、[UD Japanese GSD corpus](https://universaldependencies.org/treebanks/ja_gsd/index.html)における単語分割です。

| モデル | F1 スコア  |
|---|---|
| MeCab | 99.7%  |
| SHIBA | 97.9% |

UD 上の単語分割において MeCab を超えるのは難しそうですが、MeCab と違って辞書が必要ないので、非標準的なテキストに対する単語分割では、SHIBA が役に立つことが期待できます。

## 使い方

モデルだけの使用なら、以下のようにインストールできます：

```bash
pip install shiba-model
```

日本語 Wikipedia で事前学習したチェックポイントは下記のように使えます。
`get_pretrained_state_dict()` は自動的にチェックポイントをダウンロードします。
自分でダウンロードしたい方は[ここ](https://storage.googleapis.com/shiba.octanove.com/published_checkpoints/shiba_check45k.pt)からダウンロードできます。

```python
from shiba import Shiba, CodepointTokenizer, get_pretrained_state_dict
shiba_model = Shiba()
shiba_model.load_state_dict(get_pretrained_state_dict())
shiba_model.eval() # disable dropout
tokenizer = CodepointTokenizer()

inputs = tokenizer.encode_batch(['自然言語処理', '柴ドリル'])
outputs = shiba_model(**inputs)
```

他のトランスフォーマーのエンコーダと同様に、分類や文字レベルタスクに合わせてファインチューニングできます。
タスク固有レイヤーを付け足すのは簡単なはずですが、本リポジトリには、分類と系列ラベリングに使えるモデル `ShibaForClassification` と `ShibaForSequenceLabeling` も含まれています。

```python
from shiba import ShibaForClassification
cls_model = ShibaForClassification(vocab_size=3)
cls_model.load_encoder_checkpoint()
```

`load_encoder_checkpoint()` は事前学習されたエンコーダのみをロードする関数ですが、`cls_model.shiba_model.load_state_dict(get_pretrained_state_dict())`とほぼ同じです。

また、比較的学習しやすいタスクで、効率的な文字レベルのモデルを学習したいだけであれば、SHIBA をゼロから学習することもできます。

# 詳細

近いうちに、SHIBA に関する技術ブログを公開するつもりです。
以下に、重要な詳細を記載します。

## CANINE との違い

SHIBA の構造は CANINEとほぼ同じですが、注意すべき違いがいくつかあります

* SHIBA は、 CANINE に使われている blockwise local attention ではなく、[windowed local attention](https://github.com/lucidrains/local-attention) を使っています。
* SHIBA に token type の埋め込みはありません。 
* 文字埋め込みのダウンサンプリングの細かいところが SHIBA と CANINE で少し異なります。主な違いとしては、CANINE と違って SHIBA は最大長の文字列の最終文字を切り詰めません。

## モデルのコード

モデルのコードは [model.py](shiba/model.py) に、トークナイザーは [codepoint_tokenizer.py](shiba/codepoint_tokenizer.py) にあります。
わかりやすさと変更のしやすさを意識して書いたコードなので、モデルの細かい仕組みを理解したい場合は、コードを自分で読んだりいじったりしていただくのが一番早いかもしれません。

## 学習方法

日本語 Wikipedia コーパスを学習データとして使い、東北大学の[日本語 BERT](https://github.com/cl-tohoku/bert-japanese)とほぼ同様な全処理をしました。
訓練インスタンスの生成は RoBERTa[[2]](#2)と同様に、インスタンスにつきできるだけ多くの文を詰め込みました。
マスキングには、ランダムスパンマスキングという動的にランダムなスパンをマスクする手法を使いました。
`[M]`がマスク文字を表す Unicode コードポイントだとすると、マスキングの具体例は下記のようになります

> 柴犬は最強の犬種である
> 
> 柴犬は[M][M]の犬種である

マスクされたスパンを置き換える際には、同じデータで学習されたBPE語彙からランダムで同じ長さのものが選択されます。学習を再現したい方は、[TRAINING.md](TRAINING.md)をご参考ください。

マスキング手法を含め、ハイパーパラメータはデータのサブセットにおける性能に基づいて決めました。
また、トランスフォーマーエンコーダの学習を扱う RoBERTa[[2]](#2) と Academic Budget BERT[[3]](#3) で使われているハイパーパラメータも参考にしました。

## 学習コード

SHIBA の学習用の実装もこのリポジトリに含まれています。
モデルのコードに比べると、学習コードは依存関係が多くあまり洗練されていませんが、同じようなモデルを学習したい場合には役に立つかもしれません。ランダムスパンマスキングとランダムBPEマスキングの実装は [masking.py](training/masking.py) で見られます。

## チェックポイント

デフォルトのモデルは下流タスクで最も高い性能を発揮したものですが、他に言語モデルのチェックポイントなども提供しています。


| Type              | Step | Note            |
|-------------------|------|-----------------|
| [Encoder Only](https://storage.googleapis.com/shiba.octanove.com/published_checkpoints/shiba_check45k.pt)      | 45k  | (default model) |
| [Encoder Only](https://storage.googleapis.com/shiba.octanove.com/published_checkpoints/shiba_check60k.pt)     | 60k  |                 |
| [LM Head + Encoder](https://storage.googleapis.com/shiba.octanove.com/published_checkpoints/lm_check45k.pt) | 45k  |                 |
| [LM Head + Encoder](https://storage.googleapis.com/shiba.octanove.com/published_checkpoints/lm_check60k.pt) | 60k  |                 |

# ライセンス

このレポジトリーの内容とコードは Apache 2.0 ライセンスで提供されています。事前学習されたモデルのチェックポイントは、CC BY-SA 4.0  ライセンスで提供されています。

# 引用

本リポジトリは、以下のように引用してください。

```bibtex
@misc{shiba,
  author = {Joshua Tanner and Masato Hagiwara},
  title = {SHIBA: Japanese CANINE model},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/octanove/shiba}},
}
```

また、CANINE の論文は、以下のように引用してください。

```bibtex
@misc{clark2021canine,
      title={CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation}, 
      author={Jonathan H. Clark and Dan Garrette and Iulia Turc and John Wieting},
      year={2021},
      eprint={2103.06874},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

# 文献
<a id="1">[1]</a> Jonathan H. Clark and Dan Garrette and Iulia Turc and John Wieting (2021). [CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation](https://arxiv.org/abs/2103.06874). CoRR, abs/2103.06874.

<a id="2">[2]</a> Yinhan Liu and Myle Ott and Naman Goyal and Jingfei Du and Mandar Joshi and Danqi Chen and Omer Levy and Mike Lewis and Luke Zettlemoyer and Veselin Stoyanov (2019). [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692). CoRR, abs/1907.11692.

<a id="3">[3]</a>
Peter Izsak and Moshe Berchansky and Omer Levy (2021). [How to Train BERT with an Academic Budget](https://arxiv.org/abs/2104.07705). CoRR, abs/2104.07705.












