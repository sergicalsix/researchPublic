#Equivalence of Backpropagation and Contrastive Hebbian

[paper](https://watermark.silverchair.com/089976603762552988.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAAsMwggK_BgkqhkiG9w0BBwagggKwMIICrAIBADCCAqUGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMXRUjrsXn0o_MchCzAgEQgIICdoys1-bItI1R7zAsToVddrFxv4cT-85mgZKHMvzPEKtEeljwyDHxNxeH7GqltrhgJu9zD1lo2ECDXe6J4mEwYWwsWcz5mwfPgOT3qXCnfzJWMr_6CBX1CuCiBhiZsuQtfYkZMtCBLXX5AohY52WMB9mJtL4IrtB67MR5Yk5L5SbZehacfieQ6zYY51uxniSBpS_bYC5tmBj-wkWc55WFHdZtvdf6kI48lZ2lHKTycpeucqvKpSxGNS87ocemRdjI2HIla5w3983P92ghGCSjlSbVTRtA6dunGfQVSVlxrQoh8zqcyj4Kh1l79O3jQn4jSoaC4Y41wsCygNa0zeeoaVBl5adfCFxZUEtZ_WEztpII3I3gZPxBqq2ndYXC0G43ayboSU4ITvvKuLQk7QiWXR63yJ-eltY2k9-v6WeOVS_nzKCzuwd2wfZmP00hK95mBAzDgD_k-E_gxpSt6vGkD1ibql9YtaooZVV7CBsbL2P3Ir_7gZO6iAawQqlmC1iVQHC0slh-DbCBlytBRu0Bb1_VpBi70cu4kxU9Ug_B2m18YY6brSD3R8N2MTPxTWhg3EOsRWP5SvYH6E-VccAvgizV9a8PsrhOQ_pcqrfZ08gtVkkx-nAzRNw7wJ6hXyiJtiQDJLZAoQuCfZECtsp4or0gBHrV9Gjeg-wvl5jA-s7etBuqH3QF1BLzxY0VH0b4wDYhzC9cMdexhPLYuOYtomyEC_JVFh62fBQ7SzZpm-3PrQs9yVyExlYm08-WuQun-ahR7a9Pf7RR_Kr50uyA874i0GSyR7KiyEepepJyQBkqSu7DeUjSUdAQc_i5koUHvl48Y2nWTQ)

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
隠れたニューロンを持つネットワークを学習する方法として，バックプロパゲーションとコントラスト・ヘブン・ラーニングがある。
バックプロパゲーションは、出力ニューロンのエラー信号を計算し、それを隠れニューロンに拡散させます。対照的なヘブの学習では、出力ニューロンを望ましい値に固定し、その効果をフィードバック接続によってネットワーク全体に拡散させる。この2つの学習形式の関係を調べるために、両者が同一の特殊なケースとして、線形出力ユニットを持つ多層パーセプトロンに弱いフィードバック接続を加えたものを考えます。この場合、出力ニューロンをクランプすることで生じるネットワーク状態の変化は、スカラーの前因子を除いて、バックプロパゲーションによって広がるエラー信号と同じであることがわかります。これは、バックプロパゲーションの機能を、生体ネットワークへの実装に適したヘブ型の学習アルゴリズムで代替できることを示唆している。


## 2.Method
![](CHL/)
CHLを定式化するために、第k-1層から第k層へのフィードフォワード接続に加えて、隣接する層の間にもフィードバック接続がある修正ネットワークを考えます（図1B参照）。フィードバック接続は、フィードフォワード接続と対称的であると仮定しますが、正の係数γが掛けられている点が異なります。言い換えれば，行列γ W Tkには，層kから層k - 1に戻るフィードバック接続が含まれている．

