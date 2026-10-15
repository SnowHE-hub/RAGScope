# 真实前后案例

原答案、引用和原文均来自已保存的开发确认配对；按文字差值选例用于说明，不改变完整实验分母，不证明事实质量变化。人工NOT_REVIEWED。

## 文字改善

### qasper:fa527becb8e2551f4fd2ae840dbd4a68971349e0

What architecture does the encoder have?

文字差值：+0.600000。
这是参考表达匹配变化；上下文组织同时变化，不能直接认定答案更正确。

B0_policy_neighbor_bridge：

The encoder has an LSTM architecture.

实际引用：['qasper:1809.01541:c:71bcc763f2553354e8632b53']

原文 `qasper:1809.01541:c:71bcc763f2553354e8632b53`，源坐标1950–4362（节选，完整片段见审核包）：

The baseline provided by the shared task organisers is a seq2seq model with attention (similar to the winning system for reinflection in CoNLL–SIGMORPHON 2016, BIBREF1 ), which receives information about context through an embedding of the two words immediately adjacent to the target form. We use this baseline implementation as a starting point and achieve the best overall accuracy of 49.87 on Task 2 by introducing three augmentations to the provided baseline system: (1) We use an LSTM to encode the entire available context; (2) We employ a multi-task learning approach with the auxiliary objec

C1_policy_core_first_bridge：

LSTM

实际引用：['qasper:1809.01541:c:71bcc763f2553354e8632b53']

原文 `qasper:1809.01541:c:71bcc763f2553354e8632b53`，源坐标1950–4362（节选，完整片段见审核包）：

The baseline provided by the shared task organisers is a seq2seq model with attention (similar to the winning system for reinflection in CoNLL–SIGMORPHON 2016, BIBREF1 ), which receives information about context through an embedding of the two words immediately adjacent to the target form. We use this baseline implementation as a starting point and achieve the best overall accuracy of 49.87 on Task 2 by introducing three augmentations to the provided baseline system: (1) We use an LSTM to encode the entire available context; (2) We employ a multi-task learning approach with the auxiliary objec

完整原文与A/B回答：`D:\RAGScope_Runtime\runs\v3_20261015_014506\HUMAN_REVIEW_PACKAGE.jsonl`，REVIEW_0099

### qasper:74a17eb3bf1d4f36e2db1459a342c529b9785f6e

what were the evaluation metrics?

文字差值：+0.575758。
这是参考表达匹配变化；上下文组织同时变化，不能直接认定答案更正确。

B0_policy_neighbor_bridge：

The evaluation metrics used were the case-insensitive 4-gram BLEU score as an automatic metric, along with subjective evaluations based on faithfulness of translation results.

实际引用：['qasper:1805.09960:c:7a134061953716aed1ee444b', 'qasper:1805.09960:c:e55cce03d7e06f76b3cd5ffb']

原文 `qasper:1805.09960:c:7a134061953716aed1ee444b`，源坐标14888–17027（节选，完整片段见审核包）：

We use the Zoph_RNN toolkit to implement all our described methods. In all experiments, the encoder and decoder include two stacked LSTM layers. The word embedding dimension and the size of hidden layers are both set to 1,000. The minibatch size is set to 128. We limit the vocabulary to 30K most frequent words for both the source and target languages. Other words are replaced by a special symbol “UNK”. At test time, we employ beam search and beam size is set to 12. We use case-insensitive 4-gram BLEU score as the automatic metric BIBREF21 for translation quality evaluation.

Phrase Translation

C1_policy_core_first_bridge：

BLEU score

实际引用：['qasper:1805.09960:c:7a134061953716aed1ee444b']

原文 `qasper:1805.09960:c:7a134061953716aed1ee444b`，源坐标14888–17027（节选，完整片段见审核包）：

We use the Zoph_RNN toolkit to implement all our described methods. In all experiments, the encoder and decoder include two stacked LSTM layers. The word embedding dimension and the size of hidden layers are both set to 1,000. The minibatch size is set to 128. We limit the vocabulary to 30K most frequent words for both the source and target languages. Other words are replaced by a special symbol “UNK”. At test time, we employ beam search and beam size is set to 12. We use case-insensitive 4-gram BLEU score as the automatic metric BIBREF21 for translation quality evaluation.

Phrase Translation

完整原文与A/B回答：`D:\RAGScope_Runtime\runs\v3_20261015_014506\HUMAN_REVIEW_PACKAGE.jsonl`，REVIEW_0051

### qasper:4c18081ae3b676cc7831403d11bc070c10120f8e

What clustering algorithms were used?

文字差值：+0.500000。
这是参考表达匹配变化；上下文组织同时变化，不能直接认定答案更正确。

B0_policy_neighbor_bridge：

SVM with ADWS kernel

实际引用：['qasper:1712.00991:c:83621abc98de8fd7910d8302']

原文 `qasper:1712.00991:c:83621abc98de8fd7910d8302`，源坐标11834–14354（节选，完整片段见审核包）：

After identifying sentences in each class, we can now answer question (1) in Section SECREF1 . From 12742 sentences predicted to have label STRENGTH, we extract nouns that indicate the actual strength, and cluster them using a simple clustering algorithm which uses the cosine similarity between word embeddings of these nouns. We repeat this for the 9160 sentences with predicted label WEAKNESS or SUGGESTION as a single class. Tables TABREF15 and TABREF16 show a few representative clusters in strengths and in weaknesses, respectively. We also explored clustering 12742 STRENGTH sentences directly

C1_policy_core_first_bridge：

CLUTO

实际引用：['qasper:1712.00991:c:83621abc98de8fd7910d8302']

原文 `qasper:1712.00991:c:83621abc98de8fd7910d8302`，源坐标11834–14354（节选，完整片段见审核包）：

After identifying sentences in each class, we can now answer question (1) in Section SECREF1 . From 12742 sentences predicted to have label STRENGTH, we extract nouns that indicate the actual strength, and cluster them using a simple clustering algorithm which uses the cosine similarity between word embeddings of these nouns. We repeat this for the 9160 sentences with predicted label WEAKNESS or SUGGESTION as a single class. Tables TABREF15 and TABREF16 show a few representative clusters in strengths and in weaknesses, respectively. We also explored clustering 12742 STRENGTH sentences directly

完整原文与A/B回答：`D:\RAGScope_Runtime\runs\v3_20261015_014506\HUMAN_REVIEW_PACKAGE.jsonl`，REVIEW_0033

## 文字退步

### qasper:db62d5d83ec187063b57425affe73fef8733dd28

Which model do they use to convert between masculine-inflected and feminine-inflected sentences?

文字差值：-0.400000。
这是参考表达匹配变化；上下文组织同时变化，不能直接认定答案更正确。

B0_policy_neighbor_bridge：

A Markov random field with an optional neural parameterization

实际引用：['qasper:1906.04571:c:09a0b7612eca7b0239337d5a']

原文 `qasper:1906.04571:c:09a0b7612eca7b0239337d5a`，源坐标20905–22603（节选，完整片段见审核包）：

We presented a new approach for converting between masculine-inflected and feminine-inflected noun phrases in morphologically rich languages. To do this, we introduced a Markov random field with an optional neural parameterization that infers the manner in which a sentence must change to preserve morpho-syntactic agreement when altering the grammatical gender of particular nouns. To the best of our knowledge, this task has not been studied previously. As a result, there is no existing annotated corpus of paired sentences that can be used as “ground truth.” Despite this limitation, we evaluated

C1_policy_core_first_bridge：

a Markov random field

实际引用：['qasper:1906.04571:c:09a0b7612eca7b0239337d5a']

原文 `qasper:1906.04571:c:09a0b7612eca7b0239337d5a`，源坐标20905–22603（节选，完整片段见审核包）：

We presented a new approach for converting between masculine-inflected and feminine-inflected noun phrases in morphologically rich languages. To do this, we introduced a Markov random field with an optional neural parameterization that infers the manner in which a sentence must change to preserve morpho-syntactic agreement when altering the grammatical gender of particular nouns. To the best of our knowledge, this task has not been studied previously. As a result, there is no existing annotated corpus of paired sentences that can be used as “ground truth.” Despite this limitation, we evaluated

完整原文与A/B回答：`D:\RAGScope_Runtime\runs\v3_20261015_014506\HUMAN_REVIEW_PACKAGE.jsonl`，REVIEW_0090

### qasper:a7d72f308444616a0befc8db7ad388b3216e2143

What datasets are used?

文字差值：-0.375385。
这是参考表达匹配变化；上下文组织同时变化，不能直接认定答案更正确。

B0_policy_neighbor_bridge：

The datasets used are the ACE05 multilingual dataset and an in-house dataset.

实际引用：['qasper:1911.00069:c:d03c826a12262ae1cca00f3d']

原文 `qasper:1911.00069:c:d03c826a12262ae1cca00f3d`，源坐标16995–18234（节选，完整片段见审核包）：

$G_1=\lbrace \mathbf {h}_{1},..,\mathbf {h}_{b_1-1}\rbrace $ includes vectors that are left to the first entity $m_1$.

$G_2=\lbrace \mathbf {h}_{b_1},..,\mathbf {h}_{e_1}\rbrace $ includes vectors that are in the first entity $m_1$.

$G_3=\lbrace \mathbf {h}_{e_1+1},..,\mathbf {h}_{b_2-1}\rbrace $ includes vectors that are between the two entities.

$G_4=\lbrace \mathbf {h}_{b_2},..,\mathbf {h}_{e_2}\rbrace $ includes vectors that are in the second entity $m_2$.

$G_5=\lbrace \mathbf {h}_{e_2+1},..,\mathbf {h}_{n}\rbrace $ includes vectors that are right to the second entity $m_2$.

We perfor

C1_policy_core_first_bridge：

The datasets used are the ACE (Automatic Content Extraction) 2005 multilingual dataset and an in-house dataset containing manually annotated RE data for 6 languages.

实际引用：['qasper:1911.00069:c:553889ab5079661dcc975f27']

原文 `qasper:1911.00069:c:553889ab5079661dcc975f27`，源坐标18234–20394（节选，完整片段见审核包）：

Given the word embeddings of a sequence of words in a target language $t$, $(\mathbf {y}_1,...,\mathbf {y}_n)$, we project them into the English embedding space by applying the linear mapping $\mathbf {M}_{t\rightarrow s}$ learned in Section SECREF13: $(\mathbf {M}_{t\rightarrow s}\mathbf {y}_1, \mathbf {M}_{t\rightarrow s}\mathbf {y}_2,...,\mathbf {M}_{t\rightarrow s}\mathbf {y}_n)$. The neural network English RE model is then applied on the projected word embeddings and the entity label embeddings (which are language independent) to perform relationship classification.

Note that our models

完整原文与A/B回答：`D:\RAGScope_Runtime\runs\v3_20261015_014506\HUMAN_REVIEW_PACKAGE.jsonl`，REVIEW_0070

### qasper:430ad71a0fd715a038f3c0fe8d7510e9730fba23

which attention based nmt method did they compare with?

文字差值：-0.333333。
这是参考表达匹配变化；上下文组织同时变化，不能直接认定答案更正确。

B0_policy_neighbor_bridge：

attention-based

实际引用：['qasper:1805.09960:c:e55cce03d7e06f76b3cd5ffb']

原文 `qasper:1805.09960:c:e55cce03d7e06f76b3cd5ffb`，源坐标17027–18963（节选，完整片段见审核包）：

The first question we are interested in is whether or not phrase translation table can improve the translation quality of NMT. Compared to the baseline, our method markedly improves the translation quality on both CH-EN translation and EN-JA translation. In CH-EN translation, the average improvement is up to 2.23 BLEU points (34.94 vs. 32.71). In EN-JA translation, the improvement can reach 1.96 BLEU points (27.95 vs. 25.99). It indicates that incorporating a phrase table into NMT can substantially improve NMT's translation quality.

In Fig. 3, we show an illustrative example of CH-EN translat

C1_policy_core_first_bridge：

Arthur

实际引用：['qasper:1805.09960:c:7a134061953716aed1ee444b']

原文 `qasper:1805.09960:c:7a134061953716aed1ee444b`，源坐标14888–17027（节选，完整片段见审核包）：

We use the Zoph_RNN toolkit to implement all our described methods. In all experiments, the encoder and decoder include two stacked LSTM layers. The word embedding dimension and the size of hidden layers are both set to 1,000. The minibatch size is set to 128. We limit the vocabulary to 30K most frequent words for both the source and target languages. Other words are replaced by a special symbol “UNK”. At test time, we employ beam search and beam size is set to 12. We use case-insensitive 4-gram BLEU score as the automatic metric BIBREF21 for translation quality evaluation.

Phrase Translation

完整原文与A/B回答：`D:\RAGScope_Runtime\runs\v3_20261015_014506\HUMAN_REVIEW_PACKAGE.jsonl`，REVIEW_0029

## 比较与总结样本缺口

独立来源开发诊断只有2条总结和1条比较，没有达到每类至少3条真实配对案例的要求。
该交付项为BLOCKED_VALID_SOURCE_TASK_COVERAGE；原整篇教师6请求、17失败分支及兼容后9失败任务保留，不生成虚构补例。
已存在的全部3条前后结果及证据位置见TASK_KEYPOINT_DIAGNOSIS.md和原6条答案journal。它们未证明任务工作流泛化收益。
