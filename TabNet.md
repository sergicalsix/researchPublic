#題名
[paper](https://db-event.jpn.org/deim2017/papers/236.pdf)

<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
<script type="text/x-mathjax-config">
 MathJax.Hub.Config({
 tex2jax: {
 inlineMath: [['$', '$'] ],
 displayMath: [ ['$$','$$'], ["\\[","\\]"] ]
 }
 });
</script>
## Abstruct
表の種類を分類するための深層ニューラルネットワークアーキテクチャTabNet を提案する．
![](TabNet1.png)

## 1.Intro
近年のビッグデータ研究の進展に伴っ
て，表からの情報抽出に関する研究が改めて注目を集めてお
り，大規模データセットである WDC Web Table Corpus 2015
のリリース [25] や，表の検索 [38]，表セルの検索 [37]，表の拡
張 [26]，表に基づく質問応答 [49]，表知識獲得 [11] など様々な
研究が行われている

RNN と CNN の結合により表の意味構造を理解可能な
新アーキテクチャTabNet を提案した（

## 2.問題定義

## 3.アーキテクチャ
![](TabNet2.png)
### 3.1 概要
V: 語彙サイズ

大きい表データはクロッピングする。emb層（埋め込み層）でE次元のベクトルに変換する。
次にLSTMとAttensionで意味を獲得したH次元のベクトルに変換

６クラス分類
### 3.2 埋め込み
小文字統一、NFKC正規化
### 3.3 RNN
### 3.4 CNN
2層の畳み込み< F個の3*3フィルタ(ストライド１)

##4. 評価方法
### 4.1 データセット
### 4.2 指標
重み付きマクロF1値(各クラスについてF1値を計算し、データ数によって重み付け)
### 4.3 ベースライン
- 表分類に特化
以下の特徴からRF

• Cafarella08 [5]
Cafarella らが設計した表全体に対する 7 つの特徴．行数，列数，セル内文字列長の分散など．


• Crestan11 [9]
Crestan らが設計した 107 個の特徴．表全体に対する特徴に加えて，表の最初の 2 行・2 列および最終の行・列について，行・列単位の素性が含まれる．構造的特徴に加えて，<th>タグの割合など HTML レベルの特徴や，コロンが含まれるセルの割合などセル内文字列の特徴が利用される．


• Eberius15 [12]
Eberius らによる Crestan11 を拡張して得られた 127 個の特徴．Dresden Web Table Corpus の構築に利用された．

- NN
HAN:2層のAttension

• Bidirectional HAN
上記した HAN では表を行の系列として扱ったが，もう一つの階層構造として，列の系列として表を見ることができる．そこで，行方向と列方向の 2 つの HAN を構築し，これらの出力を連
結して分類層に与えて表種類分類を行ったものを Bidirectional HAN として利用した．

### 4.5 結果
RQ(リサーチクエスチョン)

- RQ1 精度

- RQ2 DNN は表の訓練データ数が増えると共に
分類精度を向上できるか？

これまで主に取り組まれてきた手動設計した特徴量を Random Forests などの機械学習アルゴリズムにより学習する方法
では，表の訓練データ数が増えても大きな精度向上に結びつく
ことは無かった．そこで，訓練 Web サイト数を 1，100，200，
300（表数はそれぞれ 29,050，48,909，57,665，60,678 個）と
変化させた場合の分類精度について評価した．図 5 に示す通り，
Eberius15 については訓練データ数を増加しても分類精度の向上
が頭打ちとなるが，DNN を使用する TabNet と Bidirectional
HAN は訓練データ数の増加により分類精度が改善された．これらの結果は，手動設計した特徴量に比べ DNN が表の意味構
造を本質的に捉えやすく，データ数が増えることにより様々な
表の構造およびトピックを学習できることを示唆する．
![](TabNet3.png)

- RQ3 TabNet の分類エラーは妥当な結果となっているか？

- RQ4 . TabNet は表の意味構造を捉える特徴を抽出できるか？

図 7 は CNN の最初と最終の畳み込みフィルタの活性化マップの平均を可視化したものを示す．興味深いことに，属性行・
列を持つ関係表とエンティティ表では属性行・列に沿った線状の活性が見られ，それらを持たない行列表では同様の活性が見られなかった．その一方で，行列表ではマップ上で広く分布した活性が見られた．これは，同一属性を持つ値セルのグループ（sibling ブロック）を捉えた特徴である．このような，サイズや形状が様々に異なるセルのブロックについては，HAN などの階層 RNN では捉えにくい特徴であり，TabNet が用いるCNN が特徴抽出に有効に働いていると考える

![](TabNet4.png)

## 5. 議論


## 6.終わりに
TabNet は表の構造を反映して設計された
アーキテクチャであり，RNN が各セルのトークン系列をエンコードし，CNN がセルの意味構造（属性を記載する行の存在など）を特徴として抽出することで，表種類の分類を行う．


は，TabNet で分類を行って構築した高品質な表コーパスを用いて，知識ベース構築や知識検索を改善することが目標である．また，提案アーキテクチャの応用可能性について，TabNet を他のタスクに適用して評価したい．


