#Going Deeper in Spiking Neural Networks: VGG and Residual Architectures
[paper](https://www.frontiersin.org/articles/10.3389/fnins.2019.00095/full)

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
SNNs low-power event-driven neuromorphic hardware

機械学習におけるSNNの応用は、単純な問題に対する非常に浅いニューラルネットワークアーキテクチャに限られていました。本論文では、深層アーキテクチャを持つSNNを生成するための新しいアルゴリズム技術を提案し

## 1.Introduction
SNN (Farabet et al., 2012)

（ANN）は、脳の生物学的ニューロン（このようなコンピューティングフレームワークにインスパイアされている）が2値のスパイクベースの情報を処理するという事実を無視しています.

SNNは本質的に生物学的な妥当性が高いことに加えて、イベントドリブンなハードウェア操作が可能であるという利点があります。

スパイキングニューロンは、2値のスパイク信号が入力されたときにのみ、入力情報を処理する。まばらに分散したスパイク信号が入力された場合、スパイクまたはイベントベースのハードウェアのオーバーヘッド（消費電力）は大幅に削減されます（Chen et al.

SNNに関する研究の大半は、MNIST（LeCun et al., 1998）のような比較的単純な数字認識データセットにおける非常に単純で浅いネットワークアーキテクチャに限られており、CIFAR-10（Krizhevsky and Hinton, 2009）やImageNet（Russakovsky et al., 2015）のようなより複雑な標準的なビジョンデータセットでの性能を報告している研究はわずかしかない。彼らの限られた性能の主な理由は、SNNが時間的な情報処理能力を持つため、ANNの動作から大きくシフトしていることに起因する。このため、SNNの学習メカニズムを見直す必要があった。

##2.Related Work
SNNの学習には大きく分けて、教師ありと教師なしの2種類があります.

STDP（Spike-Timing Dependent Plasticity）のような教師なし学習メカニズムは、低消費電力のオンチップ局所学習を実現する上で魅力的ですが、MNISTデータセットのような単純な数字認識プラットフォームにおいても、その性能は依然として教師ありネットワークよりも優れています.
（Diehl and Cook, 2015） 

教師付きSNN学習アルゴリズムの特定のカテゴリでは、バックプロパゲーションなどの標準的な学習スキームを使用してANNを学習し（ANNの標準的な学習技術の優れた性能を活用するため）、その後、イベント駆動型SNNに変換してネットワークを操作しようとしている

Diehlら（2015）で著者が提案したANN-SNN変換スキームに基づいている。しかし、先行研究では、変換プロセス中にのみANNの動作を考慮しているが、我々は、変換ステップ中に実際のSNNの動作を考慮することが、分類精度の損失を最小限にするために重要であることを示す。そのため、我々は、変換段階で実際のSNN操作がループ内にあることを保証する新しい重み正規化技術を提案する。本研究は、ネットワークをスパイキング領域に変換することで、神経活性化のスパース性を利用し、電力効率の高いハードウェア実装を実現しようとするものであり、シナプス結合のスパース性を探ることを目的とした取り組みと補完関係にあることに留意されたい(Han et al., 2015a)。

##3.Main Contributions
(i) 後のセクションで説明するように、SNNにほぼロスレスで変換できるANNをトレーニングするには、様々なアーキテクチャ上の制約がある。したがって、提案された技術が、より複雑なタスクのために、より大きく、より深いアーキテクチャに拡張できるかどうかは不明である。我々は、CIFAR-10やImageNetのような複雑なデータセットにおいて、16層から34層に拡張された深いSNNが競争力のある精度を提供できるという概念実証実験を行った。

(ii) 新しいANN-SNN変換技術を提案し、最先端の技術を統計的に凌駕します。CIFAR-10データセットでの分類エラーは8.45%であり、これは現在までに報告されているSNNネットワークの中で最高のパフォーマンスを示しています。また、ImageNet 2012の全検証セットにおいて、初めてSNNの性能を報告しました。VGG-16アーキテクチャでは、30.04%のトップ1エラーレートと10.99%のトップ5エラーレートを達成しました。

(iii) より深いSNNを実現するための潜在的な経路として、Residual Network (ResNet)アーキテクチャを調査します。ResNetsのANN-SNN変換を確実にするために必要な洞察力と設計上の制約を提示します。CIFAR-10データセットでは12.54%の分類エラー、ImageNet検証セットでは34.53%のトップ1エラー率と13.67%のトップ5エラー率を報告しています。これは、残差のあるネットワーク・アーキテクチャーを持つSNNを探求しようとした最初の作品である。

(iv) SNNネットワークのスパース性は、ネットワークの深さが増すにつれて著しく増加することを実証した。このことは、計算オーバーヘッドを削減するために、イベント駆動型の操作のためにANNをSNNに変換することを検討する動機付けとなります。

##4.Preliminaries
###4.1. Input and Output Representation
ANNの入力は静的なものですが、SNNは時間の関数としての動的な二値スパイク入力に基づいて動作します。
SNNでは、ニューラルノードの入力と出力がアナログ値であるANNとは異なり、ニューラルノードも2値のスパイク入力信号を送受信する。

十分に大きな時間窓の中でネットワークへの入力として送信されるスパイクの平均数が、元のANN入力の大きさ（ここではピクセル強度）にほぼ比例するようなレートエンコードされたネットワーク動作を考えます。

ポアソンイベント生成プロセスは，ネットワークへの入力スパイクトレインを生成するために使用されます．SNNの動作の各タイムステップでは，乱数が生成され，その値が対応する入力の大きさと比較される．生成された乱数が、対応するピクセルの強度の値よりも小さい場合、スパイクイベントが発生します。このプロセスは、SNNにおける入力スパイクの平均数が、対応するANN入力の大きさに比例することを保証し、一般的に、静止画像のデータセットに基づく認識タスクのSNNをシミュレートするために使用される（Diehl et al). 



図1は、CIFAR-10データセットの特定の画像について、SNNに送信された入力スパイクの特定の時間的スナップショットを示しています。ここでは、ピクセル単位の平均値減算画像を対象としているため、入力層では、入力の大きさに比例し、入力の符号に等しい符号を持つスパイクを受信しています。しかし、それ以降の層では、ネットワーク内のスパイキング・ニューロンによって生成されるため、すべてのスパイクは正の符号となります。

このようなネットワークでは、特定の層が前の層から入力されたスパイクに対して即座に動作し、前の層のニューロンからの情報が蓄積されるまで複数の時間ステップを待つ必要がない「疑似同時動作」が行われます。ポアソンで生成されたスパイク列がネットワークに入力されると、ネットワークの出力にスパイクが生成されます。


推論は、ネットワークの出力層にあるニューロンの、与えられた時間窓における累積スパイク数に基づいて行われます。


![](goingSNN1.png)
左: ピクセルの平均値を引いたANNの入力
真ん中: アナログ入力画像から生成されたポアソン型スパイクトレインの特定のインスタンスを表しています。
右：1,000タイムステップにわたってSNNに提供された累積イベントが描かれています。これは、入力画像がSNNの動作のために時間をかけてレートエンコードされていることを示しています。

###4.2 ANN and SNN Neural Operation
ANN-to-SNN変換に用いられる活性化関数はrelu

Integrate-Fire (IF) Spiking Neuron without any leak and refractory period (Cao et al., 2015; Diehl et al., 2015)

膜電位が特定の閾値vthを超えると、入力スパイクを統合して出力スパイクを生成します。出力スパイクが発生すると、膜電位はゼロにリセットされます。新しい画像やパターンに対応するスパイク列が提示されると、すべてのニューロンがリセットされる.

vmem(t+1)=vmem(t)+∑iwi.𝕏i(t) (2)

シナプス荷重が負の場合、IFニューロンは発火電位vthを超えることができないため、出力スパイク活動はゼロとなり、ReLUの機能を反映している。

発火閾値が比較的高いと、ニューロンが出力スパイクを生成するまでの遅延が大きくなります。

深層アーキテクチャの場合、このような遅延はすぐに蓄積

低い閾値を設定すると、SNNはスパイキングニューロンの膜電位（式2の∑iwi.𝕏i(t)という項）に蓄積されるスパイク入力の異なる大きさを区別する能力を失う。

ANN-SNN変換プロセス中の分類精度の低下を最小限に抑えるためには、ニューロンの閾値とシナプスの重みの比を適切に選択することが不可欠である（Diehl et al.、2015）。

###4.3 Architectural Constraints
4.3.1 Bias in Neural Units

ANN-SNN変換スキームに使用されるニューラルユニットは、バイアス項なしで学習されます(Diehl et al., 2015)

Batch Normalization techniqueは、ネットワークの各層への入力にバイアスをかけて、各層がゼロ平均を持つ入力を提供するようにします。その代わりに，正則化手法としてdropout（Srivastava et al.，2014）を使用します。

4.3.2 プーリング操作
SNNでは、ニューロンの活性化がアナログ値ではなく2値であるため、最大プーリングを行うと次の層の情報が大きく失われてしまいます。その結果、本作ではプーリングのメカニズムとして空間平均化を考慮している（Diehl et al.2015）。

## 5. Deep Convolutional SNN Architectures: VGG
データベースの正規化」手法は、MNISTデータセットにおける3層の完全連結および畳み込みアーキテクチャについて評価されている（Diehl et al.2015）。なお、本文中では、この処理を「ウェイト・ノーマライゼーション」や「閾値バランシング」と互換的に呼んでいる。

ニューロンの発火閾値vthに対するシナプスの重みの比率を最適化することである。
したがって、ニューラル層に先行するすべてのシナプス重みは、最大ニューロン活性化に等しい正規化係数wnormでスケーリングされ、閾値は1に等しく設定される（"重み正規化"）か、または、シナプス重みを変更せずに閾値vthを対応する層の最大ニューロン活性化に等しく設定する（"閾値バランス"）。どちらの操作も数学的には全く同じです。

### 5.1  Proposed Algorithm: Spike-Norm

、上記のアルゴリズムは、私たちに疑問を投げかけます。ANNの活性化はSNNの活性化を代表するものなのか？このニューロンは、0.5と1という2つの入力を受け取る。このニューロンは2つの入力、すなわち0.5と1を受け取る。このシナリオでは、シナプスの重みを1つと考える。ReLUの最大活性化は1.5であるから、ニューロンの閾値は1.5に等しく設定されるだろう。しかし、このネットワークをSNNモードに変換すると、両方の入力がバイナリのスパイク信号を伝播することになる。ANNの入力（1）は、時間ステップごとに送信されるスパイクに変換され、もう一方の入力は、十分に大きな時間窓の持続時間の50％のスパイクを送信することになります。したがって、時間ステップごとにニューロンが受け取るスパイク入力の実際の合計は、サンプル数が多い場合には2となり、スパイク閾値（1.5）よりも高くなります。このような証拠の統合が行われないと、明らかに情報の損失が生じます。

このアルゴリズムでは、ネットワークの重みを各層ごとに順次正規化していく。訓練された特定のANNが与えられた場合，最初のステップは，十分に大きな時間窓の訓練セット上でネットワークの入力ポアソンスパイクトレインを生成することである．このポアソンスパイクトレインにより、ネットワークの第1ニューラル層が受け取ることになる重み付きスパイク入力の最大和（式2の∑iwi.𝕏i(t)という用語、以下、この文章ではSNNの最大活性化と呼ぶ）を記録することができる。ニューロンの時間的な遅延を最小限に抑えると同時に、ニューロンの発火閾値が低すぎないようにするため、第1層が受け取る最大のスパイクベースの入力に応じて、第1層を重み付け正規化する。第1層の閾値が設定された後、第1層の出力に代表的なスパイクトレインが与えられ、これにより次の層の入力スパイクストリームを生成することができます。このプロセスは、ネットワークのすべての層で連続的に行われます。我々の提案と先行研究(Diehl et al., 2015)との主な違いは、提案された重み正規化スキームが変換プロセス中の実際のSNN操作を考慮しているという事実である。

## 6. Extension to Residual Architectures

## 8. Conclusions and Future Work
SNNがANNと同等の計算能力を持つという事実を証明するものです。これにより、低消費電力のニューロモーフィック・ハードウェアで実現可能な大規模な視覚認識タスクにSNNを使用する道が開かれる可能性があります。しかし、SNNの性能を向上させるためには、まだ未解決の分野があります。深層NNの現在の成功に大きく貢献しているのは、バッチ正規化によるものです（Ioffe and Szegedy, 2015）。バイアスの少ないニューラルユニットを使用することで、バッチ・ノーマライゼーションを行わずにネットワークをトレーニングすることができますが、バイアス項を持つスパイキング・ニューロンを実装するためのアルゴリズム技術を検討する必要があります。さらに、ANNを学習してSNNに変換する際には、精度を落とすことなく行うことが望ましい。提案された変換技術は、変換損失を大幅に削減しようとしているが、このギャップをさらに減らすためには、ReLU-IF Spiking Neurons以外の他のニューラル機能のバリエーションを検討することが可能である。さらに、SNNの性能をより深いアーキテクチャに拡張するために、ResNetアーキテクチャのANN-SNN変換における精度損失を最小化するためのさらなる最適化を検討する必要がある。
