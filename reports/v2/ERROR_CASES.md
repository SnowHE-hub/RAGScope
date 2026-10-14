# 错误样例与未确认诊断

分类允许重叠。uncertain表示待语义复核或缺少因果证据，不等同已确认答错。F1<0.5只用于抽样，不改变主指标。

|类别|观察到的条数|
|---|---:|
|document_not_retrieved|3|
|document_found_evidence_missing|0|
|packing_evidence_loss|834|
|noise_or_order_uncertain|2772|
|complete_evidence_low_overlap_uncertain|1818|
|wrong_abstention|621|
|missing_or_invalid_citation|115|
|json_error|32|
|oom_or_timeout|0|

## document_not_retrieved

- qwen_4b/E4/hybrid — `squad2:5728fd3c3f37b31900477f2e`
  问题：Who decides the fate of protesters most of the time?
  输出摘录：{"answer":"The context does not support an answer to who decides the fate of protesters most of the time.","citations":[],"abstain":true}
- qwen_9b/E4/hybrid — `squad2:5728fd3c3f37b31900477f2e`
  问题：Who decides the fate of protesters most of the time?
  输出摘录：{"answer":"Judges decide the fate of protesters most of the time.","citations":["squad2:dev:article:17:8bb9e19c6a35:p19:c:bcfaf89abc5da620af06e928","squad2:dev:article:17:8bb9e19c6a35:p18:c:9058b5c7effc7a08a512c44c","squad2:dev:article:17:8bb9e19c6a35:p21:c:f552ba40457f7b0c5b5466ea"],"abstain":false}
- qwen_2b/E4/hybrid — `squad2:5728fd3c3f37b31900477f2e`
  问题：Who decides the fate of protesters most of the time?
  输出摘录：{
  "answer": "The Presiding Officer decides who speaks in chamber debates and the amount of time for which they are allowed to speak.",
  "citations": ["squad2:dev:article:29:4debe4b25435:p14:c:009dff2287c43bda8567e968"],
  "abstain": false
}

## document_found_evidence_missing

当前无符合条件的样例；未生成或虚构样例。

## packing_evidence_loss

- qwen_2b/ROUTE_DEV/hybrid — `qasper:5aa12b4063d6182a71870c98e4e1815ff3dc8a72`
  问题：Do the authors mention any downside of lemmatizing input before training ELMo?
  输出摘录：{
  "answer": "Yes, the authors mention that lemmatizing input before training ELMo yields small but consistent improvements for Russian (a rich-morphology language) while having negligible effects for English.",
  "citations": ["qasper:1909.03135:c:f46b8ce508b7b97e6bb62f6f"],
  "abstain": false
}
- qwen_4b/E2/R2 — `qasper:72dbdd11b655b25b2b254e39689a7d912f334b71`
  问题：How are properties being compared annotated?
  输出摘录：{"answer":"Properties being compared are annotated alongside the questions and knowledge sentences, serving as supervision for semantic parsing approaches.", "citations":["qasper:1909.03553:c:36edbb67fa9c7d168bbc97ef","qasper:1909.03553:c:fd8f18f50161b94bc2058f09"],"abstain":false}
- qwen_4b/E2/R3 — `qasper:800fcd8b08d36c5276f9e5e1013208d41b46de59`
  问题：why do they think sentiment features do not result in improvement?
  输出摘录：{"answer":"The context states that including sentiment lexicon features did not result in any improvement, though it does not explicitly explain the reason why.", "citations":["qasper:1805.04558:c:41b407e7f7a4a5a99fbb6401"], "abstain":true}

## noise_or_order_uncertain

- qwen_2b/ROUTE_DEV/hybrid — `qasper:5aa12b4063d6182a71870c98e4e1815ff3dc8a72`
  问题：Do the authors mention any downside of lemmatizing input before training ELMo?
  输出摘录：{
  "answer": "Yes, the authors mention that lemmatizing input before training ELMo yields small but consistent improvements for Russian (a rich-morphology language) while having negligible effects for English.",
  "citations": ["qasper:1909.03135:c:f46b8ce508b7b97e6bb62f6f"],
  "abstain": false
}
- qwen_4b/E2/R2 — `qasper:72dbdd11b655b25b2b254e39689a7d912f334b71`
  问题：How are properties being compared annotated?
  输出摘录：{"answer":"Properties being compared are annotated alongside the questions and knowledge sentences, serving as supervision for semantic parsing approaches.", "citations":["qasper:1909.03553:c:36edbb67fa9c7d168bbc97ef","qasper:1909.03553:c:fd8f18f50161b94bc2058f09"],"abstain":false}
- qwen_4b/E2/R3 — `qasper:800fcd8b08d36c5276f9e5e1013208d41b46de59`
  问题：why do they think sentiment features do not result in improvement?
  输出摘录：{"answer":"The context states that including sentiment lexicon features did not result in any improvement, though it does not explicitly explain the reason why.", "citations":["qasper:1805.04558:c:41b407e7f7a4a5a99fbb6401"], "abstain":true}

## complete_evidence_low_overlap_uncertain

- qwen_2b/E2/R1 — `qasper:845bdcd900c0f96b2ae091d086fb1ab8bb1063f0`
  问题：Which websites were used in the web crawl?
  输出摘录：{
  "answer": "The Common Crawl and Twitter",
  "citations": ["qasper:2004.00809:c:190d6183ed016013da5c1067"],
  "abstain": false
}
- qwen_2b/E2/R1 — `qasper:363a24ecb8ab45215134935e7e8165fff72ff90f`
  问题：which models is their approach compared to?
  输出摘录：{
  "answer": "word-LSTM and word-CNN",
  "citations": ["qasper:2004.01970:c:8104f54f6e5b3c2d836e70b1"],
  "abstain": false
}
- qwen_2b/E2/R0 — `qasper:dd81f58c782169886235c48b8f9a08e0954dd3ae`
  问题：What baselines do they propose?
  输出摘录：{
  "answer": "The authors propose supervised and unsupervised neural machine translation baselines for Nigerian Pidgin, implementing models with different tokenization methods (word-level and byte pair encoding) using the Transformer architecture.",
  "citations": ["qasper:2003.12660:c:528b9dd3b8f502cb63942aa1"],
  "abstain": false
}

## wrong_abstention

- qwen_4b/E3/no_context — `qasper:89373db8ced1fe420eae0093b2736f06b565616e`
  问题：Did the authors collect new data for evaluation?
  输出摘录：{"answer":"No, the authors did not collect new data for evaluation; they used existing datasets.","citations":[],"abstain":true}
- qwen_4b/E2/R3 — `qasper:800fcd8b08d36c5276f9e5e1013208d41b46de59`
  问题：why do they think sentiment features do not result in improvement?
  输出摘录：{"answer":"The context states that including sentiment lexicon features did not result in any improvement, though it does not explicitly explain the reason why.", "citations":["qasper:1805.04558:c:41b407e7f7a4a5a99fbb6401"], "abstain":true}
- qwen_2b/E3/no_context — `qasper:6371c6863fe9a14bf67560e754ce531d70de10ab`
  问题：How big is this dataset?
  输出摘录：{
"answer": "abstain",
"citations":[],
"abstain":true
}

## missing_or_invalid_citation

- qwen_4b/E3/no_context — `qasper:df623717255ea2c9e0f846859d8a9ef51dc1102b`
  问题：What type of attention do they use in the decoder?
  输出摘录：{"answer":"Self-attention","citations":[],"abstain":false}
- qwen_2b/E3/no_context — `qasper:7ef7a5867060f91eac8ad857c186e51b767c734b`
  问题：How do they generate input noise for the encoder and decoder?
  输出摘录：{
"answer":"They generate input noise using a Gaussian distribution with mean 0 and standard deviation 1.",
"citations":[],
"abstain":false
}
- qwen_2b/E3/no_context — `qasper:9baca9bdb8e7d5a750f8cbe3282beb371347c164`
  问题：How do they preprocess Tweets?
  输出摘录：{
"answer":"They use the Twitter API to retrieve tweets and then apply various preprocessing steps such as tokenization, stemming, and lemmatization before feeding them into a machine learning model.",
"citations":[],
"abstain":false
}

## json_error

- lfm/E5/bm25 — `qasper:a926d71e6e58066d279d9f7dc3210cd43f410164`
  问题：What languages were included in this multilingual population?
  输出摘录：{"abstain": false, "answer": "The multilingual population includes diverse languages and cultural backgrounds, with efforts focusing on expanding semantic resources beyond English.", "citations": ["qasper:1906.01010:c:95363d8a7c761883462990ef", "qasper:1906.01010:c:7c3814ac8531cf73a80966d6", "qasper:1906.01010:c:3542e00f30d7ea2c879b9cf4", "qasper:1906.01010:c:3542e00f30d7ea2c879b9cf4", "qasper:1906.01010:c:3542e00f30d7ea2c879b9cf4", "qasper:1906.
- lfm/E5/bm25 — `qasper:a836ab8eb5a72af4b0a0c83bf42a2a14d1b38763`
  问题：What dataset do they use?
  输出摘录：{"abstain": false, "answer": "Extracting class-level features from Java source code before and after changes significantly improves performance, increasing accuracy to 72.6% and F1 to 79.7%.", "citations": ["qasper:1911.07620:c:21433572757281c3327599d0", "qasper:1911.07620:c:21433572757281c3327599d0", "qasper:1911.07620:c:21433572757281c3327599d0", "qasper:1911.07620:c:21433572757281c3327599d0", "qasper:1911.07620:c:21433572757281c3327599d0", "qa
- lfm/E5/bm25 — `qasper:df01e98095ba8765d9ab0d40c9e8ef34b64d3700`
  问题：What NLI dataset was used?
  输出摘录：{"abstain": false, "answer": "The paper explores utilizing BERT intermediate layers for aspect-based sentiment analysis and shows that adding pooling strategies improves performance on ABSA datasets.", "citations": ["qasper:2002.04815:c:d5b035aa3cdd790aae2afa91", "qasper:2002.04815:c:635bad8b258cfa5d641f4cce", "qasper:2002.04815:c:16acd98f0a1387da3f21f4c3", "qasper:2002.04815:c:16acd98f0a1387da3f21f4c3", "qasper:2002.04815:c:635bad8b258cfa5d641f4

## oom_or_timeout

当前无符合条件的样例；未生成或虚构样例。
