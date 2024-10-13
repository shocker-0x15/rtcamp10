# レイトレ合宿10 レンダラー & デノイザー？
[レイトレ合宿10](https://sites.google.com/view/rtcamp10)用に書いたレンダラーのコード置き場。
合宿用に書いたコードなので人に見せるようなコードではないですが、概要：\
VolumetricかつSpectralなLight Vertex Cache Bidirectional Path Tracing (LVC-BPT) [1]の実装。レイトレ合宿の提出用で締切間近でアドホックな調整も加えているのでもはや(ボリューム)レンダリング方程式に対してconsistentになっていないかもしれない。\
https://speakerdeck.com/shocker_0x15/renderer-introduction-ray-tracing-camp-10

![output](output.png)\
Stanford bunny from [McGuire CG Archive](https://casual-effects.com/data/)

[1] Progressive Light Transport Simulation on the GPU: Survey and Improvements