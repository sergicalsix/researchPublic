#題名
[paper](https://www.researchgate.net/publication/335940697)

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
スパイキングニューラルネットワーク（SNN）は、脳の神経細胞の動きを模倣する有望なモデルのカテゴリーであり、脳にインスパイアされたコンピューティングで注目を集め、ニューロモーフィックデバイスに広く展開されています。しかし、SNNの実用性については、長い間、懐疑的な議論が続けられてきました。SNNは、スパイク駆動による低消費電力という利点を除けば、特にアプリケーションの精度という点ではANNよりも劣っています。近年、研究者たちはこの問題を解決するために、バックプロパゲーションなどのANNの学習方法を利用して、高精度のSNNモデルを学習しようとしています。この分野の急速な進歩は、ネットワークのサイズが大きくなるにつれ、驚くべき結果を継続的に生み出しており、その成長の道のりは深層学習の発展と似ているように思えます。これらの方法はSNNにANNの精度に近づく能力を与えているが、ANN指向の作業負荷や単純な評価指標の使用により、SNNの自然な優位性とANNを凌駕する方法が潜在的に失われている。
本論文では、視覚認識タスクをケーススタディとして取り上げ、「どのような作業負荷がSNNにとって理想的で、どのようにSNNを評価することが理にかなっているのか」という疑問に答えます。異なるタイプのデータセット（ANN指向およびSNN指向）、多様な処理モデル、信号変換方法、および学習アルゴリズムを用いて、一連の対比テストを設計する。これらのモデルを評価するために、アプリケーションの精度やメモリ・計算機のコストに関する包括的な指標を提案し、広範な実験を行った。その結果、ANN指向のワークロードでは、SNNはANNに勝てないが、SNN指向のワークロードでは、SNNは十分に性能を発揮できることがわかった。さらに、SNNでは、アプリケーションの精度と実行コストの間にトレードオフが存在し、そのトレードオフは、シミュレーションの時間窓と発火しきい値に影響されることを示した。これらの豊富な分析結果に基づいて、各シナリオに最適なモデルを推奨します。我々の知る限りでは、系統的な比較を用いて、ANNからSNNへの単純な作業負荷の移植が賢明ではないことを明示的に明らかにした最初の研究であり、多くの研究がそうしており、包括的な評価が実際に重要である。最後に、より広範なタスク、データセット、評価基準を持つSNNのベンチマークフレームワークを構築することが急務であることを強調する。

## 1. Intro
研究者たちは、多層パーセプトロン（MLP）または畳み込みニューラルネットワーク（CNN）ベースの画像認識（He, Zhang, Ren, & Sun, 2016）、音声認識（Abdel-Hamid et al, 2014）、言語処理（Hu, Lu, Li, & Chen, 2014; Young, Hazarika, Poria, & Cambria, 2018）、物体検出（Redmon & Farhadi, 2017）、日射量推定（Jahani & Mohammadi, 2018）、医療診断（Esteva et al, 2017）、ゲーム実況（Silver et al., 2016）など、リカレントニューラルネットワーク（RNN）ベースの音声認識（Lam et al., 2019）、言語処理（Ghaeini et al., 2018）、状態制御、場合によってはCNNとRNNの組み合わせ（Caglayan & Burak Can, 2018; Zhang, Bai, & Zhu, 2019; Zoph, Vasudevan, Shlens, & Le, 2018）などがあります。

SNNは、時空間的に連続したダイナミクスと、イベント駆動型の発火活動（0-nothingまたは1-spike event）で動作します。非同期のスパイクメカニズムにより、SNNは、オプティカルフロー推定（Haessig, Cassidy, Alvarez, Benosman, & Orchard, 2017）、スパイクパターン認識（Wu et al., 2019）、SLAM（Vi- dal, Rebecq, Horstschaefer, & Scaramuzza, 2018）などのイベントベースのシナリオで優位性を示している。そのほかにも、例えば、確率的推論（Maass, 2014）、NPハード問題をヒューリスティックに解く（Jonke, Habenschuss, & Maass, 2016）または最適化問題を素早く解く（Davies et al. さらに、SNNは、脳にインスパイアされた計算を行うためのニューロモルフィックデバイスに広く展開されています（Davies et al.2018; Furber, Galluppi, Temple, & Plana, 2014; Merolla et al.2014; Shi et al.2015）。


SNNの適用精度を向上させるために、従来のスパイクタイミング依存可塑性（STDP）のスーパーバイスされていない学習規則を、横方向の抑制と適応閾値（Diehl & Cook, 2015）または再区のメカニズム（Mozafari, Ganjtabesh, Nowzari-Dalini, Thorpe, & Masquelier, 2018）を追加するなどして改善することができる