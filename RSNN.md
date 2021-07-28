# Recurrent Spiking Neural Network Learning Based on a Competitive Maximization of Neuronal Activity

[paper](https://www.frontiersin.org/articles/10.3389/fninf.2018.00079/full)

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
スパイキングニューラルネットワーク（SNN）は、特定のニューロチップハードウェアのリアルタイムソリューションにおいて、高い計算能力とエネルギー効率を発揮すると考えられています。しかし、バックプロパゲーション法に匹敵する効率性を持ち、教師なしで学習可能なリカレント接続を持つ複雑なSNNの学習アルゴリズムは不足しています。ここでは、生物学的なニューラルネットワークの各ニューロンは、他のニューロンとの競争の中で自分の活動を最大化する傾向があると仮定し、この原理を新しいSNN学習アルゴリズムの基礎としています。

このようにして、フィードフォワード接続、相互接続、層内抑制接続を学習したスパイキングネットワークを、MNISTデータベースの数字認識に導入しました。このSNNは、同じアルゴリズムで重みの初期化を短時間行うだけで、教師なしで学習できることが実証されました

また、ニューロンは、異なる数字クラスとその関連性に対応した階層構造のファミリーにグループ化されることが示されている。この特性は、深層ニューラルネットワークの層数を減らしたり、生体の神経系における様々な機能構造の形成をモデル化するのに役立つと期待されています。

今回提案したアルゴリズムの学習特性を、疎分散表現法と比較したところ、符号化の仕方が似ているだけでなく、前者の利点がいくつか見られた。提案したアルゴリズムの基本原理は、より複雑で多様な課題解決型SNNの構築に実用的に適用できると考えられる。我々はこの新しいアプローチを「Family-Engaged Execution and Learning of Induced Neuron Groups」、すなわちFEELINGと呼んでいる。


## 1.Intro

