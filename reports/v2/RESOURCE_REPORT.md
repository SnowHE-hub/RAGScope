# RAGScope v2 资源局部报告

## 结论与测量范围

完整主矩阵为5,520条、39个组，32条失败保留在原分母；下表呈现各组真实生成延迟和资源采样。生成计时从冻结context开始，不含检索。模型参数量不等于本机显存占用，CPU/GPU迁移后的重排成本也不能混作同设备加速比。非流式TTFT没有测量，保持N/A；device-total、服务树RSS与Torch allocator是不同量，不能直接相减解释offload。既有总览图见[质量与延迟](figures/quality_latency.png)。

后续复现实验应沿原模型身份和资源条件读取逐条样本；本报告没有新增基准、重新拟合或改变硬件配置。

生成时间从冻结context开始，不包含检索；单位按原字段保留。

|实验|模型|方法|实际N|计划N|失败|资源测量N|P50 ms|P95 ms|输入tokens P50|decode tokens/s P50|device-total峰值 bytes|服务树RSS峰值 bytes|TTFT|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|E2|qwen_2b|R0|200|200|0|200|1090.2930499978538|1704.9466450023826|2132.5|125.8088166223557|1519386624|1111556096|N/A|
|E2|qwen_2b|R1|200|200|0|200|1098.078250000981|1661.4154050002956|2152.0|122.27712437194484|1519386624|1117581312|N/A|
|E2|qwen_2b|R2|200|200|0|200|1079.1407500018977|1711.9616549978664|2136.5|122.10651547188135|1519386624|1119051776|N/A|
|E2|qwen_2b|R3|200|200|0|200|1092.8351000002294|1612.0758299988659|2136.0|123.803593413285|1519386624|1115201536|N/A|
|E2|qwen_4b|R0|200|200|0|200|2259.4379000001936|3716.796225001781|2132.5|73.3122190868167|3037724672|1270845440|N/A|
|E2|qwen_4b|R1|200|200|0|200|2441.7370000010123|3729.803870000976|2152.0|73.30680936839482|3037724672|1262940160|N/A|
|E2|qwen_4b|R2|200|200|0|200|2463.79055000034|3750.32586500074|2136.5|73.47247409271615|3037724672|1281392640|N/A|
|E2|qwen_4b|R3|200|200|0|200|2435.6173500000295|3584.30839500088|2136.0|73.06018932358762|3037724672|1257025536|N/A|
|E2|qwen_9b|R0|200|200|0|200|3499.214600004052|5633.424994999222|2132.5|48.31747436169568|5472518144|1314332672|N/A|
|E2|qwen_9b|R1|200|200|0|200|3802.273150002293|5851.628159999382|2152.0|48.63827985758728|5472518144|1292869632|N/A|
|E2|qwen_9b|R2|200|200|0|200|3651.8319499991776|5693.930064998129|2136.5|48.48146534032514|5472518144|1299603456|N/A|
|E2|qwen_9b|R3|200|200|0|200|3633.6107500028447|5844.587905000661|2136.0|48.5589199593854|5472518144|1311154176|N/A|
|E3|qwen_2b|no_context|200|200|0|200|499.06815006397665|642.5733699463306|130.0|115.73922673795768|1507852288|1140375552|N/A|
|E3|qwen_2b|oracle|200|200|0|200|846.6622499981895|1682.9276049858868|305.0|115.0111092648878|1507852288|1235062784|N/A|
|E3|qwen_4b|no_context|200|200|0|200|488.00849996041507|822.8970249881967|130.0|74.91097741659564|3030384640|1508167680|N/A|
|E3|qwen_4b|oracle|200|200|0|200|1204.2621999280527|2740.9697648370625|305.0|77.37798897356822|3030384640|1536831488|N/A|
|E3|qwen_9b|no_context|200|200|0|200|920.8531498443335|1099.0948950638992|130.0|46.8663944910612|5463080960|1565446144|N/A|
|E3|qwen_9b|oracle|200|200|0|200|1773.7122499383986|4705.333670112301|305.0|50.8233085059064|5472518144|1565777920|N/A|
|E4|qwen_2b|hybrid|100|100|0|100|1054.241350000666|1325.0601499981713|2225.0|138.7954107898085|1519386624|1091829760|N/A|
|E4|qwen_4b|hybrid|100|100|2|100|1911.0877500024799|3102.275014999759|2225.0|69.73426950568151|3041918976|1332781056|N/A|
|E4|qwen_9b|hybrid|100|100|0|100|2841.8495500009158|4249.703399999998|2225.0|50.04312822664079|5470420992|1290838016|N/A|
|E5|lfm|bm25|100|100|13|100|1340.7836000042153|3721.963640003378|2156.0|195.92710281108623|980418560|621293568|N/A|
|E5|lfm|hybrid|100|100|16|100|1349.8997499991674|3430.769974994473|2149.5|191.4459854989504|980418560|620916736|N/A|
|E6|qwen_2b|simple|100|100|1|100|1043.5558999997738|1668.1713900048635|2014.5|126.86233683302234|1519386624|1092259840|N/A|
|E6|qwen_2b|structured|100|100|0|100|1051.9059499965806|1540.7547200000408|2147.5|126.50868539470622|1519386624|1091805184|N/A|
|E6|qwen_4b|simple|100|100|0|100|2191.1698000003526|3618.615545002467|2014.5|73.57622388556803|3041918976|1303154688|N/A|
|E6|qwen_4b|structured|100|100|0|100|2390.8031999999366|3657.7455599959644|2147.5|73.71647602624267|3041918976|1303134208|N/A|
|E7|qwen_2b|end|40|40|0|40|794.6764000007533|1687.6539000037146|684.0|135.64303085251157|1519386624|1091473408|N/A|
|E7|qwen_2b|middle|40|40|0|40|792.5202499973238|1769.590639998568|684.0|129.03303248316797|1519386624|1091608576|N/A|
|E7|qwen_2b|start|40|40|0|40|778.788750001695|1597.4224000023237|684.0|124.42171746894469|1519386624|1091477504|N/A|
|E7|qwen_4b|end|40|40|0|40|1407.5027500002761|2981.156334995464|684.0|76.61987107189101|3041918976|1306931200|N/A|
|E7|qwen_4b|middle|40|40|0|40|1523.9832499974|3056.665294995399|684.0|76.56890630175798|3041918976|1302224896|N/A|
|E7|qwen_4b|start|40|40|0|40|1469.122549999156|3261.4464150021367|684.0|73.1917892951188|3041918976|1303142400|N/A|
|ROUTE_DEV|qwen_2b|bm25|60|60|0|60|1075.3282000041509|1582.976900000721|2142.0|127.72950428242744|1519386624|1092276224|N/A|
|ROUTE_DEV|qwen_2b|hybrid|60|60|0|60|1148.7479500028712|1941.0777800057983|2151.0|127.34556445580898|1519386624|1091497984|N/A|
|ROUTE_DEV|qwen_2b|hybrid_rerank|60|60|0|60|1049.955949998548|1776.1477000029117|2135.0|130.59486628361043|1519386624|1091420160|N/A|
|ROUTE_TRAIN|qwen_2b|bm25|200|200|0|200|1056.5771500005212|1701.299035002739|2161.0|132.52878157035795|1519386624|1091833856|N/A|
|ROUTE_TRAIN|qwen_2b|hybrid|200|200|0|200|1052.996999998868|1887.4409299976833|2153.5|134.3716629814635|1519386624|1091645440|N/A|
|ROUTE_TRAIN|qwen_2b|hybrid_rerank|200|200|0|200|1052.7158500008227|1794.288755002707|2146.5|133.3622485300026|1519386624|1115578368|N/A|

## GPU R3迁移与可比性

```json
{
  "activation": "sigmoid",
  "aggregation": "windowed_max",
  "attention_implementation": "eager",
  "autocast": false,
  "backend": "transformers_sequence_classifier",
  "backend_versions": {
    "numpy": "2.5.4",
    "torch": "2.9.1+cu128",
    "transformers": "5.19.0"
  },
  "batch_size": 2,
  "campaign": "20261014_v2",
  "checkpoint_manifest": {
    "bytes": 3509,
    "path": "D:\\RAGScope_Runtime\\models\\bge-reranker-base\\2cfc18c9415c912f9d8155881c133215df768a70\\download_manifest.json",
    "sha256": "87a9da253900e640f2f29b934b52e3570e9b73756756780c10a267c9acb789bc"
  },
  "comparison_limit": "CPU TEST200 remains E2 input; full GPU global100 replaces partial CPU29 for global statistics. Routing CPU/GPU backend differences disclosed, never claim device equivalence.",
  "cpu_groups_preserved": [
    {
      "n_failed": 0,
      "n_saved": 100,
      "run_id": "E1_R512_hybrid_rerank_neighbor"
    },
    {
      "n_failed": 0,
      "n_saved": 200,
      "run_id": "TEST_CONTEXT_R512_hybrid_rerank_neighbor"
    },
    {
      "n_failed": 0,
      "n_saved": 29,
      "run_id": "E8_R512_hybrid_rerank_neighbor"
    }
  ],
  "cpu_threads": 4,
  "device": "cuda",
  "dtype": "torch.float32",
  "expected_gpu_rows": 360,
  "groups": [
    {
      "destination": "D:\\RAGScope_Runtime\\runs\\20261014_v2_retrieval\\records\\E8_R512_hybrid_rerank_neighbor_gpu.jsonl",
      "n_expected": 100,
      "prefix": "E8",
      "query_ids": [
        "qasper:c0355afc7871bf2e12260592873ffdb5c0c4c919",
        "qasper:e8c3f59313df20db0cdd49b84a37c44da849fe17",
        "qasper:d9949dd4865e79c53284932d868ca8fd10d55e70",
        "qasper:c17ece1dad42d92c78fca2e3d8afa9a20ff19598",
        "qasper:0bc305d6b90f77f835bc4c904b22a4be07f963b2",
        "qasper:fac273ecb3e72f2dc94cdbc797582d7225a8e070",
        "qasper:000549a217ea24432c0656598279dbb85378c113",
        "qasper:63d2e97657419a0185127534f4ff9d0039cb1a63",
        "qasper:43f43b135109ebd1d2d1f9af979c64ce550b5f0f",
        "qasper:4a201b8b9cc566b56aedb5ab45335f202bc41845",
        "qasper:1f40adc719d8ccda81e7e90525b577f5698b5aad",
        "qasper:0bde3ecfdd7c4a9af23f53da2cda6cd7a8398220",
        "qasper:f7ee48dd32c666ef83a4ae4aa06bcde85dd8ec4b",
        "qasper:a3705b53c6710b41154c65327b7bbec175bdfae7",
        "qasper:51e9f446d987219bc069222731dfc1081957ce1f",
        "qasper:ced63053eb631c78a4ddd8c85ec0f3323a631a54",
        "qasper:67c16ba64fe27838b1034d15194c07a9c98cdebe",
        "qasper:b948bb86855b2c0bfc8fad88ff1e29cd94bb6ada",
        "qasper:157284acedf13377cbc6d58c8f3648d3a62f5db5",
        "qasper:c79f168503a60d1b08bb2c9aac124199d210b06d",
        "qasper:9dd8ce48a2a59a63ae6366ab8b2b8828e5ae7f35",
        "qasper:fd8a8eb69f07c584a76633f8802c2746f7236d64",
        "qasper:6590055fb033cb32826f2afecb3d7f607dd97d57",
        "qasper:fc62549a8f0922c09996a119b2b6a8b5e829e989",
        "qasper:909ecf675f874421eecc926a9f7486475aa1423c",
        "qasper:7d8cd7d6c86349ef0bd4fdbd84c8dc49c7678f46",
        "qasper:0fee37ebe0a010cf8bd665fa566306d8e7d12631",
        "qasper:f8bba20d1781ce2b14fad28d6eff024e5a6c2c02",
        "qasper:252599e53f52b3375b26d4e8e8b66322a42d2563",
        "qasper:2ca3ca39d59f448e30be6798514709be7e3c62d8",
        "qasper:70512cc9dcd45157e40c8d1f85e82d21ade7645b",
        "qasper:740cc392c0c8bfadfe6b3a60c0be635c03e17f2a",
        "qasper:845bdcd900c0f96b2ae091d086fb1ab8bb1063f0",
        "qasper:21c89ee0281f093b209533453196306b9699b552",
        "qasper:5096aaea2d0f4bea4c12e14f4f7735e1aea1bfa6",
        "qasper:800fcd8b08d36c5276f9e5e1013208d41b46de59",
        "qasper:cdbbba22e62bc9402aea74ac5960503f59e984ff",
        "qasper:0bd3bea892c34a3820e98c4a42cdeda03753146b",
        "qasper:8cf5abf0126f19253930478b02f0839af28e4093",
        "qasper:0b5c599195973c563c4b1a0fe5d8fc77204d71a0",
        "qasper:cc8bcea4052bf92f249dda276acc5fd16cac6fb4",
        "qasper:35f48b8f73728fbdeb271b170804190b5448485a",
        "qasper:16edc21a6abc89ee2280dccf1c867c2ac4552524",
        "qasper:37e6ce5cfc9d311e760dad8967d5085446125408",
        "qasper:6683008e0a8c4583058d38e185e2e2e18ac6cf50",
        "qasper:df01e98095ba8765d9ab0d40c9e8ef34b64d3700",
        "qasper:d3cfbe497a30b750a8de3ea7f2cecf4753a4e1f9",
        "qasper:9baca9bdb8e7d5a750f8cbe3282beb371347c164",
        "qasper:7557f2c3424ae70e2a79c51f9752adc99a9bdd39",
        "qasper:b03249984c26baffb67e7736458b320148675900",
        "qasper:9595fdf7b51251679cd39bc4f6befc81f09c853c",
        "qasper:0a70af6ba334dfd3574991b1dd06f54fc6a700f2",
        "qasper:98b97d24f31e9c535997e9b6cb126eb99fc72a90",
        "qasper:71b07d08fb6ac8732aa4060ae94ec7c0657bb1db",
        "qasper:180c7bea8caf05ca97d9962b90eb454be4176425",
        "qasper:63eda2af88c35a507fbbfda0ec1082f58091883a",
        "qasper:de344aeb089affebd15a8c370ae9ab5734e99203",
        "qasper:c2037887945abbdf959389dc839a86bc82594505",
        "qasper:e9a0a69eacd554141f56b60ab2d1912cc33f526a",
        "qasper:2abf916bc03222d3b2a3d66851d87921ff35c0d2",
        "qasper:1fd31fdfff93d65f36e93f6919f6976f5f172197",
        "qasper:3588988f2230f3329d7523fbb881b20bf177280d",
        "qasper:5bfbc9ca7fd41be9627f6ef587bb7e21c7983be0",
        "qasper:5181527e6a61a9a192db5f8064e56ec263c42661",
        "qasper:b8bbdc3987bb456739544426c6037c78ede01b77",
        "qasper:4e59808a7f73ac499b9838d3c0ce814196a02473",
        "qasper:50c441a9cc7345a0fa408d1ce2e13f194c1e82a8",
        "qasper:9be9354eeb2bb1827eeb1e23a20cfdca59fb349a",
        "qasper:d6a815d24c46557827d8aca65d3ffd008ac1bc07",
        "qasper:d9354c0bb32ec037ff2aacfed58d57887a713163",
        "qasper:c035a011b737b0a10deeafc3abe6a282b389d48b",
        "qasper:d3fb0d84d763cb38f400b7de3daaa59ed2a1b0ab",
        "qasper:351f7b254e80348221e0654478663a5e53d3fe65",
        "qasper:df623717255ea2c9e0f846859d8a9ef51dc1102b",
        "qasper:ff36168caf48161db7039e3bd4732cef31d4de99",
        "qasper:425d17465ff91019eb87c28ff3942f781ba1bbcb",
        "qasper:ec2045e0da92989642a5b5f2b1130c8bd765bcc5",
        "qasper:2a9c7243744b42f1e9fed9ff2ab17c6f156b1ba4",
        "qasper:2bd702174e915d97884d1571539fb1b5b0b7123a",
        "qasper:10edfb9428b8a4652274c13962917662fdf84f8a",
        "qasper:a836ab8eb5a72af4b0a0c83bf42a2a14d1b38763",
        "qasper:0b5a7ccf09810ff5a86162d502697d16b3536249",
        "qasper:6371c6863fe9a14bf67560e754ce531d70de10ab",
        "qasper:f3b4e52ba962a0004064132d123fd9b78d9e12e2",
        "qasper:ea6edf45f094586caf4684463287254d44b00e95",
        "qasper:ba406e07c33a9161e29c75d292c82a15503beae5",
        "qasper:80bb07e553449bde9ac0ff35fcc718d7c161f2d4",
        "qasper:c8f8ecac23a991bceb8387e68b3b3f2a5d8cf029",
        "qasper:dd81f58c782169886235c48b8f9a08e0954dd3ae",
        "qasper:64c45fdb536ae294cf06716ac20d08b5fdb7944d",
        "qasper:a064d01d45a33814947161ff208abb88d4353b26",
        "qasper:c5ea4da3c760ba89194ad807bc1ef60e1761429f",
        "qasper:4a093a9af4903a59057a4372ac1b01603467ca58",
        "qasper:f4e16b185b506713ff99acc4dbd9ec3208e4997b",
        "qasper:9f065e787a0d40bb4550be1e0d64796925459005",
        "qasper:6b1a6517b343fdb79f246955091ff25e440b9511",
        "qasper:742d5e182b57bfa5f589fde645717ed0ac3f49c2",
        "qasper:dfdd309e56b71589b25412ba85b0a5d79a467ceb",
        "qasper:f1a50f88898556ecdba8e9cac13ae54c11835945",
        "qasper:72dbdd11b655b25b2b254e39689a7d912f334b71"
      ],
      "run_id": "E8_R512_hybrid_rerank_neighbor_gpu",
      "selection_provenance": [
        {
          "bytes": 359,
          "path": "D:\\RAGScope_Runtime\\runs\\20261014_v2_retrieval\\CANDIDATE_SELECTION.json",
          "sha256": "fdb2a034acfb932caec71aff4a67fc02fc9154aa9b35d47e78c18a7184bc8c74"
        },
        {
          "bytes": 1104,
          "path": "D:\\RAGScope_Runtime\\runs\\20261014_v2_retrieval\\CONTEXT_SELECTION.json",
          "sha256": "9f6a01a5b7c83309f664d37c043f605b6e87e5248142c19105f2d74ebf696f51"
        },
        {
          "bytes": 497365,
          "path": "D:\\RAGScope_Runtime\\corpora\\20261014_v2\\queries_global_test.jsonl",
          "sha256": "702048198abd4aa45861a1e28afbb28a9628cccdf0b07546e36619958226d301"
        },
        {
          "bytes": 18428,
          "path": "D:\\RAGScope_Runtime\\runs\\20261014_v2\\ROUTING_QUERY_MANIFEST.json",
          "sha256": "060bc704f5ebcc89903c1e5a3999ea63fb8bbafd2707e64d9a1b4c8bc2d9cd7f"
        }
      ],
      "source_artifacts": [
        {
          "bytes": 8972108,
          "path": "D:\\RAGScope_Runtime\\runs\\20261014_v2_retrieval\\records\\E8_R512_hybrid_neighbor.jsonl",
          "sha256": "6e2b59ced2177cc8643474ffc04c0111f3ba37e66316bb0579698fa0bc2871f6"
        },
        {
          "bytes": 17624,
          "path": "D:\\RAGScope_Runtime\\runs\\20261014_v2_retrieval\\records\\E8_R512_hybrid_neighbor.config.json",
          "sha256": "9d1c80e77746246f2df23b5fd9d7c6b45513fdf1811f9f0e549cb545a711766b"
        }
      ]
    },
    {
      "destination": "D:\\RAGScope_Runtime\\runs\\20261014_v2_retrieval\\records\\ROUTE_CONTEXT_TRAIN_R512_hybrid_rerank_neighbor_gpu.jsonl",
      "n_expected": 200,
      "prefix": "ROUTE_CONTEXT_TRAIN",
      "query_ids": [
        "qasper:c431c142f5b82374746a2b2f18b40c6874f7131d",
        "qasper:82fa2b99daa981fc42a882bb6db8481bdbbb9675",
        "qasper:e74ba39c35af53d3960be5a6c86eddd62cef859f",
        "qasper:eddb18109495976123e10f9c6946a256a55074bd",
        "qasper:08a5f8d36298b57f6a4fcb4b6ae5796dc5d944a4",
        "qasper:7aba5e4483293f5847caad144ee0791c77164917",
        "qasper:17f5f4a5d943c91d46552fb75940b67a72144697",
        "qasper:be6971827707afcd13af3085d0a775a0bd61c5dd",
        "qasper:5cb3d69607f60e1c5be2120462726a477ead9570",
        "qasper:86a93a2d1c19cd0cd21ad1608f2a336240725700",
        "qasper:3371d586a3a81de1552d90459709c57c0b1a2594",
        "qasper:cf93a209c8001ffb4ef505d306b6ced5936c6b63",
        "qasper:6adfa9eee76b96953a76c03356bf41d8a9378851",
        "qasper:9e805020132d950b54531b1a2620f61552f06114",
        "qasper:ac148fb921cce9c8e7b559bba36e54b63ef86350",
        "qasper:80f19be1cbe1f0ec89bbafb9c5f7a8ded37881fb",
        "qasper:c000a43aff3cb0ad1cee5379f9388531b5521e9a",
        "qasper:0cf6d52d7eafd43ff961377572bccefc29caf612",
        "qasper:b8b588ca1e876b3094ae561a875dd949c8965b2e",
        "qasper:58ad7e8f7190e2a4f1588cae9a7842c56b37694d",
        "qasper:736c74d2f61ac8d3ac31c45c6510a36c767a5d6d",
        "qasper:ca7e71131219252d1fab69865804b8f89a2c0a8f",
        "qasper:5bb96b255dab3e47a8a68b1ffd7142d0e21ebe2a",
        "qasper:774ead7c642f9a6c59cfbf6994c07ce9c6789a35",
        "qasper:a08b5018943d4428f067c08077bfff1af3de9703",
        "qasper:79cfd1b82c72d18e2279792c66a042c0e9dfa6b7",
        "qasper:9e1bf306658ef2972159643fdaf149c569db524b",
        "qasper:8e630c5a4a8ba0a4f5d8c483a2bf09c4ac8020ce",
        "qasper:5f0bb32d70ee8e4c4c59dc5c193bc0735fd751cc",
        "qasper:c30c3e0f8450b1c914d29f41c17a22764fa078e0",
        "qasper:a234bcbf2e41429422adda37d9e926b49ef66150",
        "qasper:f103789b85b00ec973076652c639bd31c605381e",
        "qasper:ccf7415b515fe5c59fa92d4a8af5d2437c591615",
        "qasper:7eb3852677e9d1fb25327ba014d2ed292184210c",
        "qasper:ee31c8a94e07b3207ca28caef3fbaf9a38d94964",
        "qasper:601f96770726a0063faf9bacd5db01c4af5add1f",
        "qasper:994ac7aa662d16ea64b86510fcf9efa13d17b478",
        "qasper:b622f57c4e429b458978cb8863978d7facab7cfe",
        "qasper:0e57a0983b4731eba9470ba964d131045c8c7ea7",
        "qasper:6c80bc3ed6df228c8ca6e02c0a8a1c2889498688",
        "qasper:0f60864503ecfd5b048258e21d548ab5e5e81772",
        "qasper:46563a1fb2c3e1b39a185e4cbb3ee1c80c8012b7",
        "qasper:c348a8c06e20d5dee07443e962b763073f490079",
        "qasper:33ccbc401b224a48fba4b167e86019ffad1787fb",
        "qasper:771b373d09e6eb50a74fffbf72d059ad44e73ab0",
        "qasper:ca4b66ffa4581f9491442dcec78ca556253c8146",
        "qasper:310e61b9dd4d75bc1bebbcb1dae578f55807cd04",
        "qasper:29983f4bc8a5513a198755e474361deee93d4ab6",
        "qasper:ba973b13f26cd5eb1da54663c0a72842681d5bf5",
        "qasper:1d860d7f615b9ca404c504f9df4231a702f840ef",
        "qasper:6dc9960f046ec6bd280a721724458f66d5a9a585",
        "qasper:f9751e0ca03f49663a5fc82b33527bc8be1ed0aa",
        "qasper:045dbdbda5d96a672e5c69442e30dbf21917a1ee",
        "qasper:351510da69ab6879df5ff5c7c5f49a8a7aea4632",
        "qasper:69e627942dfa7018d2933a88cea9543641d12d0d",
        "qasper:9785ecf1107090c84c57112d01a8e83418a913c1",
        "qasper:a999761aa976458bbc7b4f330764796446d030ff",
        "qasper:c10f38ee97ed80484c1a70b8ebba9b1fb149bc91",
        "qasper:3c0eaa2e24c1442d988814318de5f25729696ef5",
        "qasper:9714cb7203c18a0c53805f6c889f2e20b4cab5dd",
        "qasper:95abda842c4df95b4c5e84ac7d04942f1250b571",
        "qasper:4f9a8b50903deb1850aee09c95d1b6204a7410b4",
        "qasper:a3c6acf900126bc9bd9c50ce99041ea00761da6a",
        "qasper:62c4c8b46982c3fcf5d7c78cd24113635e2d7010",
        "qasper:5efed109940bf74ed0a9d4a5e97a535502b23d27",
        "qasper:e1132564b0dd916a522e7690bc7719d2bba3fe68",
        "qasper:587885bc86543b8f8b134c20e2c62f6251195571",
        "qasper:0428e06f0550e1063a64d181210795053a8e6436",
        "qasper:37db7ba2c155c2f89fc7fb51fffd7f193c103a34",
        "qasper:00ef9cc1d1d60f875969094bb246be529373cb1d",
        "qasper:0aa46c132515d8830a72f263812cdf7cbd5627c6",
        "qasper:4debd7926941f1a02266b1a7be2df8ba6e79311a",
        "qasper:7e328cc3cffa521e73f111d6796aaa9661c8eb07",
        "qasper:42a61773aa494f7b12838f71a949034c12084de1",
        "qasper:828615a874512844ede9d7f7d92bdc48bb48b18d",
        "qasper:e75f5bd7cc7107f10412d61e3202a74b082b0934",
        "qasper:b42323d60827ecf0d9e478c9a31f90940cfae975",
        "qasper:e67d2266476abd157fc8c396b3dfb70cb343471e",
        "qasper:1383ddd4619cf81227c72f3d9f30c10c47a0cdad",
        "qasper:eae13c9693ace504eab1f96c91b16a0627cd1f75",
        "qasper:cb8a6f5c29715619a137e21b54b29e9dd48dad7d",
        "qasper:83251fd4a641cea8b180b49027e74920bca2699a",
        "qasper:31b631a8634f6180b20a72477040046d1e085494",
        "qasper:d92f1c15537b33b32bfc436e6d017ae7d9d6c29a",
        "qasper:37c7c62c9216d6cf3d0858cf1deab6db4b815384",
        "qasper:c1ce652085ef9a7f02cb5c363ce2b8757adbe213",
        "qasper:e9f868f22ae70c7681c28228b6019e155013f8d6",
        "qasper:4e468ce13b7f6ac05371c62c08c3cec1cd760517",
        "qasper:8a7bd9579d2783bfa81e055a7a6ebc3935da9d20",
        "qasper:05238d1fad2128403577822aa4822ef8ca9570ac",
        "qasper:af8d3ee6a282aaa885e9126aa4bcb08ac68837e0",
        "qasper:39cfb8473c8be4e5d8ecc3227b800a10477c5f80",
        "qasper:6bcff3ef61aad6bf1280ea26ed79585e1b838e64",
        "qasper:8720c096c8b990c7b19f956ee4930d5f2c019e2b",
        "qasper:090f2b941b9c5b6b7c34ae18c2cc97e9650f1f0b",
        "qasper:eecf62e18a790bcfdd8a56f0c4f498927ff2fb47",
        "qasper:66bf0d61ffc321f15e7347aaed191223f4ce4b4a",
        "qasper:9bd938859a8b063903314a79f09409af8801c973",
        "qasper:4f1a5eed730fdcf0e570f9118fc09ef2173c6a1b",
        "qasper:da068b20988883bc324e55c073fb9c1a5c39be33",
        "qasper:a9b10e3db5902c6142e7d6a83253ad2a6cee77fc",
        "qasper:4c026715ee365c709381c5da770bdc8297eed19f",
        "qasper:f0e8f045e2e33a2129e67fb32f356242db1dc280",
        "qasper:f268b70b08bd0436de5310e390ca5f38f7636612",
        "qasper:9bcc1df7ad103c7a21d69761c452ad3cd2951bda",
        "qasper:f748cb05becc60e7d47d34f4c5f94189bc184d33",
        "qasper:f01aa192d97fa3cc36b6e316355dc5da0e9b97dc",
        "qasper:b85ab5f862221fac819cf2fef239bcb08b9cafc6",
        "qasper:4d4550533edb19c38cb876b1640e62e34e2b88e0",
        "qasper:871c34219eb623bde9ac3937aa0f28fc3ad69445",
        "qasper:cf15c4652e23829d8fb4cf2a25e64408c18734c1",
        "qasper:3ebdc15480250f130cf8f5ab82b0595e4d870e2f",
        "qasper:6ff240d985bbe96b9d5042c9b372b4e8f498f264",
        "qasper:06c340c7c2ad57d7621c3e8baba6a3d0ed9f4696",
        "qasper:61404466cf86a21f0c1783ce535eb39a01528ce8",
        "qasper:d28260b5565d9246831e8dbe594d4f6211b60237",
        "qasper:53bf6238baa29a10f4ff91656c470609c16320e1",
        "qasper:8f8f2b0046e1a78bd34c0c3d6b6cb24463a8ed7f",
        "qasper:4170ed011b02663f5b1b1a3c1f0415b7abfaa85d",
        "qasper:28b2a20779a78a34fb228333dc4b93fd572fda15",
        "qasper:84aef81dae38e0dca0ad041141df60ab9ac29761",
        "qasper:6adde6bc3e27a32eac5daa57d30ab373f77690be",
        "qasper:a891039441e008f1fd0a227dbed003f76c140737",
        "qasper:372fbf2d120ca7a101f70d226057f9639bf1f9f2",
        "qasper:9651fbd887439bf12590244c75e714f15f50f73d",
        "qasper:565d668947ffa6d52dad019af79289420505889b",
        "qasper:d32b6ac003cfe6277f8c2eebc7540605a60a3904",
        "qasper:b1f2db88a6f89d0f048803e38a0a568f5ba38fc5",
        "qasper:f3e96c5487d87557a661a65395b0162033dc05b3",
        "qasper:84a4a1f4695eba599d447e030c94f51e5f2f03bb",
        "qasper:de03e8cc1ceaf2108383114460219bf46e00423c",
        "qasper:d6a27c41c81f12028529e97e255789ec2ba39eaa",
        "qasper:e09dcb6fc163bba7d704178e7edba2e630b573c2",
        "qasper:5d85d7d4d013293b4405beb4b53fa79ac7c03401",
        "qasper:98ba7a7aae388b1a77dd6cab890977251d906359",
        "qasper:c9ee70c481c801892556eb6b9fd8ee38197923be",
        "qasper:3513682d4ee2e64725b956c489cd5b5995a6acf2",
        "qasper:11d2f0d913d6e5f5695f8febe2b03c6c125b667c",
        "qasper:5cd5864077e4074bed01e3a611b747a2180088a0",
        "qasper:cdc5a998cb73262594cdae1dda49576044da3d3d",
        "qasper:e8e6986365f899dead0768ecf7b1eca8a2699f2f",
        "qasper:edc43e1b75c0970b7003deeabfe3ad247cb1ed83",
        "qasper:f1c70baee0fd02b8ecb0af4b2daa5a56f3e9ccc3",
        "qasper:5774e019101415a43e0b5a780179fd897fc013fd",
        "qasper:b6ffa18d49e188c454188669987b0a4807ca3018",
        "qasper:65461516098ed63c45a567648e8e47c38ea7e58a",
        "qasper:766e2e35968ef7434b56330aa41957c5d5f8d0ee",
        "qasper:642c4704a71fd01b922a0ef003f234dcc7b223cd",
        "qasper:534f69c8c90467d5aa4e38d7c25c53dbc94f4b24",
        "qasper:abc5836c54fc2ac8465aee5a83b9c0f86c6fd6f5",
        "qasper:bbaf7cbae88c085faa6bbe3319e4943362fe1ad4",
        "qasper:edcde2b675cf8a362a63940b2bbdf02c150fe01f",
        "qasper:762f2527f85c3ae6bbdf1f331311930ef1e1fa51",
        "qasper:869aaf397c9b4da7ab52d6dd0961887ae08da9ae",
        "qasper:a24a7a460fd5e60d71a7e787401c68caa4702df6",
        "qasper:7d300176afa04947ac847135ac6ea2929908c0b0",
        "qasper:37016cc987d33be5ab877013ef26ec7239b48bd9",
        "qasper:442f8da2c988530e62e4d1d52c6ec913e3ec5bf1",
        "qasper:af7a9b56596f90c84f962098f7e836309161badf",
        "qasper:ba48c095c496d01c7717eaa271470c3406bf2d7c",
        "qasper:9447ec36e397853c04dcb8f67492ca9f944dbd4b",
        "qasper:2c20426c003f7e3053f8e6c333f8bb744f6f31f8",
        "qasper:9dd65dca9dffd2bf78ecc22b17824edc885d1fa2",
        "qasper:6407dae0c095c2c8e15e6769f03925aa4f0e902e",
        "qasper:fa5357c56ea80a21a7ca88a80f21711c5431042c",
        "qasper:ad67ca844c63bf8ac9fdd0fa5f58c5a438f16211",
        "qasper:5dc2f79cd8078d5976f2df9ab128d4517e894257",
        "qasper:d20d6c8ecd7cb0126479305d27deb0c8b642b09f",
        "qasper:6743c1dd7764fc652cfe2ea29097ea09b5544bc3",
        "qasper:18e915b917c81056ceaaad5d6581781c0168dac9",
        "qasper:ed11b4ff7ca72dd80a792a6028e16ba20fccff66",
        "qasper:7cf726db952c12b1534cd6c29d8e7dfa78215f9e",
        "qasper:73738e42d488b32c9db89ac8adefc75403fa2653",
        "qasper:43761478c26ad65bec4f0fd511ec3181a100681c",
        "qasper:a779d452d11f368c66f7b51f7190d0fe9402f505",
        "qasper:c2da598346b74541c78ecff5c9586b3857dd01b5",
        "qasper:a9cc4b17063711c8606b8fc1c5eaf057b317a0c9",
        "qasper:21a9f1cddd7cb65d5d48ec4f33fe2221b2a8f62e",
        "qasper:a3b1520e3da29d64af2b6e22ff15d330026d0b36",
        "qasper:fe578842021ccfc295209a28cf2275ca18f8d155",
        "qasper:59a5959a6abfb81b114e7bfaa945301349d20f0f",
        "qasper:52b113e66fd691ae18b9bb8a8d17e1ee7054bb81",
        "qasper:24c0f3d6170623385283dfda7f2b6ca2c7169238",
        "qasper:18482658e0756d69e39a77f8fcb5912545a72b9b",
        "qasper:56e58bdf0df76ad1599021801f6d4c7b77953e29",
        "qasper:7994b4001925798dfb381f9aa5c0545cdbd77220",
        "qasper:fbd47705262bfa0a2ba1440a2589152def64cbbd",
        "qasper:da544015511e535503dee2eaf4912a5e36c806cd",
        "qasper:e6c163f80a11bd057bbd0b6e1451ac82edddc78d",
        "qasper:6147846520a3dc05b230241f2ad6d411d614e24c",
        "qasper:3213529b6405339dfd0c1d2a0f15719cdff0fa93",
        "qasper:06095a4dee77e9a570837b35fc38e77228664f91",
        "qasper:5b99f74bb25bc88677621443bf065d96d84895ab",
        "qasper:7fdeef2b1c8f6bd5d7c3a44e533d8aae2bbc155f",
        "qasper:79443bf3123170da44396b0481364552186abb91",
        "qasper:9eb5b336b3dcb7ab63f673ba9ab1818573cce6c3",
        "qasper:b02d2d351bd2e49d4d59db0a8a6ef23cb90bfbc4",
        "qasper:0d0959dba3f7c15ee4f5cdee51682656c4abbd8f",
        "qasper:3a1bd3ec1a7ce9514da0cb2dfcaa454ba8c0ed14",
        "qasper:626873982852ec83c59193dd2cf73769bf77b3ed"
      ],
      "run_id": "ROUTE_CONTEXT_TRAIN_R512_hybrid_rerank_neighbor_gpu",
      "selection_provenance": [
        {
          "bytes": 359,
          "path": "D:\\RAGScope_Runtime\\runs\\20261014_v2_retrieval\\CANDIDATE_SELECTION.json",
          "sha256": "fdb2a034acfb932caec71aff4a67fc02fc9154aa9b35d47e78c18a7184bc8c74"
        },
        {
          "bytes": 1104,
          "path": "D:\\RAGScope_Runtime\\runs\\20261014_v2_retrieval\\CONTEXT_SELECTION.json",
          "sha256": "9f6a01a5b7c83309f664d37c043f605b6e87e5248142c19105f2d74ebf696f51"
        },
        {
          "bytes": 2440374,
          "path": "D:\\RAGScope_Runtime\\corpora\\20261014_v2\\queries_train.jsonl",
          "sha256": "da5513fab70f0e39b2afbf231fa8be32790eb2e7038785b45b6df4cc912eff43"
        },
        {
          "bytes": 18428,
          "path": "D:\\RAGScope_Runtime\\runs\\20261014_v2\\ROUTING_QUERY_MANIFEST.json",
          "sha256": "060bc704f5ebcc89903c1e5a3999ea63fb8bbafd2707e64d9a1b4c8bc2d9cd7f"
        }
      ],
      "source_artifacts": [
        {
          "bytes": 9934887,
          "path": "D:\\RAGScope_Runtime\\runs\\20261014_v2_retrieval\\records\\ROUTE_CONTEXT_TRAIN_R512_hybrid_neighbor.jsonl",
          "sha256": "1d154e7a5a8004c9e50dc36fcab47465a22fbe86d59b0db88d2d428d3691be14"
        },
        {
          "bytes": 23285,
          "path": "D:\\RAGScope_Runtime\\runs\\20261014_v2_retrieval\\records\\ROUTE_CONTEXT_TRAIN_R512_hybrid_neighbor.config.json",
          "sha256": "64dd0b12112fd94c2f5138be4f80e95b54aeb0c8cd5ab2717e8c8ceec384ad0e"
        }
      ]
    },
    {
      "destination": "D:\\RAGScope_Runtime\\runs\\20261014_v2_retrieval\\records\\ROUTE_CONTEXT_DEV_R512_hybrid_rerank_neighbor_gpu.jsonl",
      "n_expected": 60,
      "prefix": "ROUTE_CONTEXT_DEV",
      "query_ids": [
        "qasper:6becff2967fe7c5256fe0b00231765be5b9db9f1",
        "qasper:67ec8ef85844e01746c13627090dc2706bb2a4f3",
        "qasper:d9412dda3279729e95fcb35cbed09e61577a896e",
        "qasper:12f7fac818f0006cf33269c9eafd41bbb8979a48",
        "qasper:9a9d225f9ac35ed35ea02f554f6056af3b42471d",
        "qasper:79a44a68bb57b375d8a57a0a7f522d33476d9f33",
        "qasper:0101ebfbaba75fd47868ad0c796ac44ebc19c566",
        "qasper:a7d020120a45c39bee624f65443e09b895c10533",
        "qasper:22815878083ebd2f9e08bc33a5e733063dac7a0f",
        "qasper:ba539cab80d25c3e20f39644415ed48b9e4e4185",
        "qasper:abe2393415e533cb06311e74ed1c5674cff8571f",
        "qasper:1591068b747c94f45b948e12edafe74b5e721047",
        "qasper:82642d3111287abf736b781043d49536fe48c350",
        "qasper:869feb7f47606105005efdb6bea1c549824baea0",
        "qasper:ab54cd2dc83141bad3cb3628b3f0feee9169a556",
        "qasper:709feae853ec0362d4e883db8af41620da0677fe",
        "qasper:7784d321ccc64db5141113b6783e4ba92fdd4b20",
        "qasper:df95b3cb6aa0187655fd4856ae2b1f503d533583",
        "qasper:b4f881331b975e6e4cab1868267211ed729d782d",
        "qasper:5c5aeee83ea3b34f5936404f5855ccb9869356c1",
        "qasper:50cb50657572e315fd452a89f3e0be465094b66f",
        "qasper:2a564b092916f2fabbfe893cf13de169945ef2e1",
        "qasper:567dc9bad8428ea9a2658c88203a0ed0f8da0dc3",
        "qasper:f20a389ace2267aa61eddcc235535452ccdae0e6",
        "qasper:bc01853512eb3c11528e33003ceb233d7c1d7038",
        "qasper:79413ff5d98957c31866f22179283902650b5bb6",
        "qasper:c9bc6f53b941863e801280343afa14248521ce43",
        "qasper:4477bb513d56e57732fba126944073d414d1f75f",
        "qasper:d64383e39357bd4177b49c02eb48e12ba7ffd4fb",
        "qasper:585626d18a20d304ae7df228c2128da542d248ff",
        "qasper:0d9fcc715dee0ec85132b3f4a730d7687b6a06f4",
        "qasper:76121e359dfe3f16c2a352bd35f28005f2a40da3",
        "qasper:899ed05c460bf2aa0aa65101cad1986d4f622652",
        "qasper:9658b5ffb5c56e5a48a3fea0342ad8fc99741908",
        "qasper:adbf33c6144b2f5c40d0c6a328a92687a476f371",
        "qasper:6d1217b3d9cfb04be7fcd2238666fa02855ce9c5",
        "qasper:f7a89b9cd2792f23f2cb43d50a01b8218a6fbb24",
        "qasper:7bd6a6ec230e1efb27d691762cc0674237dc7967",
        "qasper:cd32a38e0f33b137ab590e1677e8fb073724df7f",
        "qasper:220d11a03897d85af91ec88a9b502815c7d2b6f3",
        "qasper:5aa12b4063d6182a71870c98e4e1815ff3dc8a72",
        "qasper:33554065284110859a8ea3ca7346474ab2cab100",
        "qasper:63c3550c6fb42f41a0c93133e9fca12ac00df9b3",
        "qasper:6411622cc8b2fbedbfa468859d453596d3bd2f03",
        "qasper:25c1c4a91f5dedd4e06d14121af3b5921db125e9",
        "qasper:9b76f428b7c8c9fc930aa88ee585a03478bff9b3",
        "qasper:8ce11515634236165cdb06ba80b9a36a8b9099a2",
        "qasper:8a7615fc6ff1de287d36ab21bf2c6a3b2914f73d",
        "qasper:0d34c0812f1e69ea33f76ca8c24c23b0415ebc8d",
        "qasper:5913930ce597513299e4b630df5e5153f3618038",
        "qasper:fc77d70c305fa80447b191248aba93da63ac3704",
        "qasper:71a0c4f19be4ce1b1bae58a6e8f2a586e125d074",
        "qasper:02428a8fec9788f6dc3a86b5d5f3aa679935678d",
        "qasper:7b4992e2d26577246a16ac0d1efc995ab4695d24",
        "qasper:23e2971c962bb6486bc0a66ff04242170dd22a1d",
        "qasper:6024039bbd1118c5dab86c41cce1175d99f10a25",
        "qasper:8910ee2236a497c92324bbbc77c596dba39efe46",
        "qasper:a267d620af319b48e56c191aa4c433ea3870f6fb",
        "qasper:2c59528b6bc5b5dc28a7b69b33594b274908cca6",
        "qasper:8f16dc7d7be0d284069841e456ebb2c69575b32b"
      ],
      "run_id": "ROUTE_CONTEXT_DEV_R512_hybrid_rerank_neighbor_gpu",
      "selection_provenance": [
        {
          "bytes": 359,
          "path": "D:\\RAGScope_Runtime\\runs\\20261014_v2_retrieval\\CANDIDATE_SELECTION.json",
          "sha256": "fdb2a034acfb932caec71aff4a67fc02fc9154aa9b35d47e78c18a7184bc8c74"
        },
        {
          "bytes": 1104,
          "path": "D:\\RAGScope_Runtime\\runs\\20261014_v2_retrieval\\CONTEXT_SELECTION.json",
          "sha256": "9f6a01a5b7c83309f664d37c043f605b6e87e5248142c19105f2d74ebf696f51"
        },
        {
          "bytes": 400020,
          "path": "D:\\RAGScope_Runtime\\corpora\\20261014_v2\\queries_dev.jsonl",
          "sha256": "75226422faf4c645ad7892838fa906c8c6b98d57e041da0ef9eb815c1a10578e"
        },
        {
          "bytes": 18428,
          "path": "D:\\RAGScope_Runtime\\runs\\20261014_v2\\ROUTING_QUERY_MANIFEST.json",
          "sha256": "060bc704f5ebcc89903c1e5a3999ea63fb8bbafd2707e64d9a1b4c8bc2d9cd7f"
        }
      ],
      "source_artifacts": [
        {
          "bytes": 2906945,
          "path": "D:\\RAGScope_Runtime\\runs\\20261014_v2_retrieval\\records\\ROUTE_CONTEXT_DEV_R512_hybrid_neighbor.jsonl",
          "sha256": "d0451df1e019c0e31636e8f7f96064442eba3a43ea95ab394f99a495e793815d"
        },
        {
          "bytes": 16281,
          "path": "D:\\RAGScope_Runtime\\runs\\20261014_v2_retrieval\\records\\ROUTE_CONTEXT_DEV_R512_hybrid_neighbor.config.json",
          "sha256": "6ac8a7437a40745e7b7b14a792217b69ee5de4e55e446803d4337f9974851fec"
        }
      ]
    }
  ],
  "historical_extra_partial_calls": 29,
  "historical_partial_in_final_gpu_denominator": false,
  "inherited_load_state": {
    "attempts": 2,
    "backend_failures": 0,
    "load_seconds": 68.23663799999849,
    "pid": 30108,
    "started_at": 1791944793.8861227,
    "status": "LOADED"
  },
  "max_pair_tokens": 512,
  "migration_hash": "3a986b31c041c47dc500729d824523cd6447657824bebcd48e24343e1db239c7",
  "protected_cpu_artifacts": [
    {
      "bytes": 5624622,
      "path": "D:\\RAGScope_Runtime\\runs\\20261014_v2_retrieval\\records\\E1_R512_hybrid_rerank_neighbor.jsonl",
      "sha256": "16b0874d0988ab11453e2d403b85d2b67eec6094b774eb6bf8940689bb80ad2e"
    },
    {
      "bytes": 18544,
      "path": "D:\\RAGScope_Runtime\\runs\\20261014_v2_retrieval\\records\\E1_R512_hybrid_rerank_neighbor.config.json",
      "sha256": "62d9d3518b5b641f7cfecd1d4ac953c326d249bd4a90101529deaa39f9f0d6f9"
    },
    {
      "bytes": 2098,
      "path": "D:\\RAGScope_Runtime\\runs\\20261014_v2_retrieval\\R3_GROUP_E1_VALIDATION.json",
      "sha256": "9d740aac9441a460d8cff0f23f200e45ff495b660bfc9d64eb4b2651000d2742"
    },
    {
      "bytes": 1858,
      "path": "D:\\RAGScope_Runtime\\runs\\20261014_v2_retrieval\\summaries\\E1_R512_hybrid_rerank_neighbor.json",
      "sha256": "0c874bc1fe16767127fb31bf3ff9ef0adc8d47a471abcf4a4383ced286ccb90d"
    },
    {
      "bytes": 11215893,
      "path": "D:\\RAGScope_Runtime\\runs\\20261014_v2_retrieval\\records\\TEST_CONTEXT_R512_hybrid_rerank_neighbor.jsonl",
      "sha256": "a4366ef11e137a239da889eb0ef8528ec5990976d8f4dbfe8efdc1377e6cbfbc"
    },
    {
      "bytes": 47604,
      "path": "D:\\RAGScope_Runtime\\runs\\20261014_v2_retrieval\\records\\TEST_CONTEXT_R512_hybrid_rerank_neighbor.config.json",
      "sha256": "3d49a5a4ef3a5b5e259643eff5a8d04d5b6352b82798ce935ac948e1d7580631"
    },
    {
      "bytes": 2169,
      "path": "D:\\RAGScope_Runtime\\runs\\20261014_v2_retrieval\\R3_GROUP_TEST_CONTEXT_VALIDATION.json",
      "sha256": "4d0ee2ba01788b0778ba182f67440ee25f24adf4685ea70995c0042071afdb7d"
    },
    {
      "bytes": 2128,
      "path": "D:\\RAGScope_Runtime\\runs\\20261014_v2_retrieval\\summaries\\TEST_CONTEXT_R512_hybrid_rerank_neighbor.json",
      "sha256": "92de6e75dfcacb1d03e3269909dafc80cfc3bcee74f478e062d4ebb78777d984"
    },
    {
      "bytes": 3068853,
      "path": "D:\\RAGScope_Runtime\\runs\\20261014_v2_retrieval\\records\\E8_R512_hybrid_rerank_neighbor.jsonl",
      "sha256": "f87d1a24c265e80506c8bc4a88972715cdd769319a3fe6b0e58dd11c0f5d23f4"
    },
    {
      "bytes": 37571,
      "path": "D:\\RAGScope_Runtime\\runs\\20261014_v2_retrieval\\records\\E8_R512_hybrid_rerank_neighbor.config.json",
      "sha256": "6c8ae9939be56ea6a6a3efddc9d49f0b2213ced23ca0990ef0d67eff1673ba8d"
    },
    {
      "bytes": 131,
      "path": "D:\\RAGScope_Runtime\\runs\\20261014_v2_retrieval\\R3_RUNTIME\\LOAD_STATE.json",
      "sha256": "ac6d2105f2f0144f10515a2faf418286f67e712372ccf1de2419943506f880f3"
    },
    {
      "bytes": 42834,
      "path": "D:\\RAGScope_Runtime\\runs\\20261014_v2_retrieval\\R3_RUNTIME\\RESOURCE_PAUSE_PID30108.json",
      "sha256": "8b6e9e388cab318355eed66315a3889c91e8ae6cf62b3d6559cd23bbb8b1d4b5"
    }
  ],
  "reason": "Explicit user GPU authorization; CPU runtime RAM guard after329; deadline-bound device migration",
  "revision": "2cfc18c9415c912f9d8155881c133215df768a70",
  "source_code_artifacts": [
    {
      "bytes": 13910,
      "path": "D:\\MCS_26S1课程资料\\个人项目\\RAGScope\\ragscope\\experiments\\rerank_gpu.py",
      "sha256": "91391bfbca782f2200febd2e2eda827e524d5ed93c83a09b884af403fed1adb5"
    },
    {
      "bytes": 21387,
      "path": "D:\\MCS_26S1课程资料\\个人项目\\RAGScope\\ragscope\\experiments\\rerank_campaign.py",
      "sha256": "7f5ba4c888d8949ba31d407d5538fd4483754a695ed630593ddbfa57e4c7bac2"
    },
    {
      "bytes": 6963,
      "path": "D:\\MCS_26S1课程资料\\个人项目\\RAGScope\\ragscope\\retrieval\\verified_reranker.py",
      "sha256": "5bf600542a5e0689f294657b5baf9f4aaee4e0cea608efd3c71b78781e573259"
    }
  ],
  "status": "FROZEN_NOT_EXECUTED",
  "tf32": false,
  "window_overlap": 32
}
```


## GPU R3完整性签收

```json
{
  "status": "PASS",
  "scope": "all_gpu_three_groups",
  "n_total": 360,
  "n_failed": 0,
  "worker_exited": true,
  "gpu_migration_complete": true
}
```


## 来源与实际构建

清单身份：`ac56e65060b49d8d536fdaf1584727a1f0e736b4ab1266c1845c0c1d6383f07c`；完整主矩阵N=5520。

## 本次实际只读构建命令

```json
[
  "D:\\RAGScope_Runtime\\envs\\core\\Scripts\\python.exe",
  "scripts/build_local_reports_v2.py",
  "--build",
  "--spec",
  "D:/RAGScope_Runtime/runs/20261014_v2/FINAL_LOCAL_REPORT_SOURCE_SPEC.json",
  "--output",
  "D:/RAGScope_Runtime/runs/20261014_v2/final_local_reports"
]
```


## 来源实际执行命令收据

```json
[
  {
    "receipt": {
      "bytes": 2588,
      "path": "D:\\RAGScope_Runtime\\runs\\20261014_v2\\MAIN_ANALYSIS_EXECUTION.json",
      "sha256": "75e71844bb6f0accca42c3bcd0068c9296bb839925c62bfa1f7b26f8a62352e3"
    },
    "observed": {
      "status": "COMPLETE_MAIN_ANALYSIS_ERRORS_FIGURES"
    },
    "actual_command_available": false,
    "missing_command_label": "UNAVAILABLE"
  },
  {
    "receipt": {
      "bytes": 3515,
      "path": "D:\\RAGScope_Runtime\\runs\\20261014_v2_lora\\train_2b_1024_seed42\\TRAINING_AUDIT.json",
      "sha256": "992dafabdab41a4c4c443a1e172c09ddff15024f2e7e9d717d27395021e60ae3"
    },
    "observed": {
      "status": "PASS_REAL_TRAINING_AUDIT"
    },
    "actual_command_available": false,
    "missing_command_label": "UNAVAILABLE"
  },
  {
    "receipt": {
      "bytes": 4149,
      "path": "D:\\RAGScope_Runtime\\runs\\20261014_v2_lora\\train_2b_1024_seed42\\paired_test100\\INDEPENDENT_COMPLETION_AUDIT.json",
      "sha256": "04c912f1e226f6e6a2b8cb93a4178cdd666082a70115ffaddad2601f7b53509f"
    },
    "observed": {
      "status": "COMPLETE_SAME_BF16_PAIR_AND_RELEASED"
    },
    "actual_command_available": false,
    "missing_command_label": "UNAVAILABLE"
  },
  {
    "receipt": {
      "bytes": 1100,
      "path": "D:\\RAGScope_Runtime\\runs\\20261014_v2_lora\\train_2b_1024_seed42\\paired_test100\\ROOT_OBSERVED_COMPLETION_WITNESS.json",
      "sha256": "826d1853efa29ea129fd59e343db8d4d80d8bc8a5cb816ce7e60ca86f2aadf71"
    },
    "observed": {
      "status": "COMPLETE_ROWS_PROCESS_EXIT_OBSERVED_ORIGINAL_TOOL_EXIT_UNAVAILABLE"
    },
    "actual_command_available": false,
    "missing_command_label": "UNAVAILABLE"
  },
  {
    "receipt": {
      "bytes": 1313,
      "path": "D:\\RAGScope_Runtime\\runs\\20261014_v2_routing\\online\\EXECUTION_START.json",
      "sha256": "dde383bf4cbaaa1906c2113ff3463c4c2cc93b1e7e072977336442aa639ff0c6"
    },
    "observed": {
      "argv": [
        "D:\\RAGScope_Runtime\\envs\\models\\Scripts\\python.exe",
        "scripts/run_online_routing_v2.py",
        "--campaign",
        "D:\\RAGScope_Runtime\\runs\\20261014_v2_generation",
        "--routing-dir",
        "D:\\RAGScope_Runtime\\runs\\20261014_v2_routing",
        "--output",
        "D:\\RAGScope_Runtime\\runs\\20261014_v2_routing\\online",
        "--mode",
        "run",
        "--embedding-url",
        "http://127.0.0.1:11436",
        "--embedding-identity",
        "D:\\RAGScope_Runtime\\runs\\20261014_v2\\embedding_identity.json",
        "--lease",
        "D:\\RAGScope_Runtime\\runs\\20261014_v2\\gpu_lease.json",
        "--service-receipt",
        "D:\\RAGScope_Runtime\\runs\\20261014_v2\\generation_service.json"
      ],
      "status": "STARTING_REPAIR_ATTEMPT2"
    },
    "actual_command_available": true,
    "missing_command_label": null
  },
  {
    "receipt": {
      "bytes": 3743,
      "path": "D:\\RAGScope_Runtime\\runs\\20261014_v2_routing\\online\\RELEASE_AFTER_ONLINE_COMPLETION.json",
      "sha256": "d8f7981a69a703ad495ccaa99a49cc6d371903116c72d9391668d7b632afe308"
    },
    "observed": {
      "status": "PASS_RELEASED_CPU_CLIENT"
    },
    "actual_command_available": false,
    "missing_command_label": "UNAVAILABLE"
  },
  {
    "receipt": {
      "bytes": 2756,
      "path": "D:\\RAGScope_Runtime\\runs\\20261014_v2\\FINAL_OWNED_RESOURCE_RELEASE.json",
      "sha256": "6086c4abf0c37efa729a33c562c061b9af72edb448b95dbd513d01ae213c2f31"
    },
    "observed": {
      "status": "ALL_PROJECT_GPU_CPU_UI_RESOURCES_RELEASED"
    },
    "actual_command_available": false,
    "missing_command_label": "UNAVAILABLE"
  },
  {
    "receipt": {
      "bytes": 3738,
      "path": "D:\\RAGScope_Runtime\\runs\\20261014_v2\\FINAL_LOCAL_REPORT_COMMAND_EVIDENCE.json",
      "sha256": "a620fb3825cc85a5e60ec4cc05cb09ee0afa772bd08675c0b0307682b80fbddd"
    },
    "observed": {
      "command": {
        "final_owned_resource_exit_codes": {
          "CPU_stop_request_command": 0,
          "CPU_supervisor64314_planned_stop": 0,
          "GPU_supervisor50644_planned_stop": 1,
          "UI35046_planned_interrupt": 1
        },
        "main_analysis_commands": [
          {
            "actual_exit_code": 0,
            "exec_session": 14359,
            "exit_witness": "Actual root tool completion; numerical commands logged their output, empty plot stdout has explicit observed completion witness",
            "log": {
              "bytes": 98,
              "path": "D:\\RAGScope_Runtime\\runs\\20261014_v2\\main_analysis_final.log",
              "sha256": "e7e186a49badbe7110fcea8472b744a5914658b01e4d939ddb9ff90a3599b6e6"
            },
            "script": "scripts\\analyze_generation_v2.py"
          },
          {
            "actual_exit_code": 0,
            "exec_session": 40205,
            "exit_witness": "Actual root tool completion; numerical commands logged their output, empty plot stdout has explicit observed completion witness",
            "log": {
              "bytes": 22,
              "path": "D:\\RAGScope_Runtime\\runs\\20261014_v2\\main_error_cases_final.log",
              "sha256": "37d1dc8f8354a777978f52c263df031b6687480ec1cc18835610e4b17efdf1ff"
            },
            "script": "scripts\\analyze_error_cases_v2.py"
          },
          {
            "actual_exit_code": 0,
            "exec_session": null,
            "exit_witness": "Actual root tool completion; numerical commands logged their output, empty plot stdout has explicit observed completion witness",
            "log": {
              "bytes": 418,
              "path": "D:\\RAGScope_Runtime\\runs\\20261014_v2\\main_figures_layout_fix.log",
              "sha256": "f5eabf5973284d0ab9d3d1b591b3b28f2d3e99520533928ce6f0e480ff6e3b46"
            },
            "script": "scripts\\plot_results_v2.py"
          }
        ],
        "online_actual_exit_code": 0,
        "online_argv": [
          "D:\\RAGScope_Runtime\\envs\\models\\Scripts\\python.exe",
          "scripts/run_online_routing_v2.py",
          "--campaign",
          "D:\\RAGScope_Runtime\\runs\\20261014_v2_generation",
          "--routing-dir",
          "D:\\RAGScope_Runtime\\runs\\20261014_v2_routing",
          "--output",
          "D:\\RAGScope_Runtime\\runs\\20261014_v2_routing\\online",
          "--mode",
          "run",
          "--embedding-url",
          "http://127.0.0.1:11436",
          "--embedding-identity",
          "D:\\RAGScope_Runtime\\runs\\20261014_v2\\embedding_identity.json",
          "--lease",
          "D:\\RAGScope_Runtime\\runs\\20261014_v2\\gpu_lease.json",
          "--service-receipt",
          "D:\\RAGScope_Runtime\\runs\\20261014_v2\\generation_service.json"
        ],
        "paired_independent_audit_status": "COMPLETE_SAME_BF16_PAIR_AND_RELEASED",
        "paired_original_exec_exit_code": null,
        "paired_original_exec_session": 4583,
        "training_audit_command": "UNAVAILABLE_IN_TRAINING_AUDIT_TOP_LEVEL"
      },
      "status": "ACTUAL_OBSERVED_RECEIPTS_ONLY"
    },
    "actual_command_available": true,
    "missing_command_label": null
  }
]
```


- `D:\RAGScope_Runtime\runs\20261014_v2_routing\DECISIONS_fixed_bm25.json` · SHA256 `6116e0ee228dff9427f2cd0bdc85fbd2208fd06013d50d1b9868f85d3e200937` · 31989 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_routing\DECISIONS_fixed_hybrid.json` · SHA256 `b7669bbb31084be7843df73804273ad4409b241a3a2afbd5043d111d9965a26a` · 32303 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_routing\DECISIONS_fixed_hybrid_rerank.json` · SHA256 `6ff7327c4e9e02a03fa7b0df0a76d141d1c1c80b0330f91c32c73cc55edcf39d` · 33456 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_routing\DECISIONS_rules.json` · SHA256 `95bb3d370a16b20328cba4bb816ee5d38a41ed6fa84178e5d681f1c26f591e1a` · 35098 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_routing\DECISIONS_linear.json` · SHA256 `031b25820f7a4408252bdfceb7f42cd6eae919f66449221eae0e4d85f520e9b5` · 64738 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_routing\DECISIONS_mlp.json` · SHA256 `39cc6984a702b1f016df1a0488911b3582db9f8ad4c5c7e3229cf27489dbbad4` · 65587 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_routing\DECISIONS_lstm.json` · SHA256 `780252e3c2ee7005ae7542ad4e3db9007fb9f541951adecf05feaf01265b8f5e` · 64716 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_routing\ROUTING_MANIFEST.json` · SHA256 `cc57c3db3609b24361f82e6b423d59b3f59be524a3258a019ae4872934be06c6` · 121754 bytes

- `D:\MCS_26S1课程资料\个人项目\RAGScope\reports\v2\ANALYSIS.json` · SHA256 `207c2fb8d4041a2d890acd5f82ab58070c184c35dfcb15c9eb73a836865938d1` · 306846 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_generation\EXPERIMENT_MANIFEST.json` · SHA256 `034c2e1a7a364de2e084c44fe7b6b61655b1d7d92d0e5eef5b390308e5f8517f` · 2775830 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_generation\FROZEN_INPUTS.json` · SHA256 `9983da3c70a1f98d28b4e6a81f0f114ea7cc240684cf3251731472031b25cbe0` · 148914449 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_generation\records\E4_qwen_2b.jsonl` · SHA256 `0702e9b1e8ebc0c9facd91f3f15c18ef5ee34810c461db1d1d7b587316bbcc10` · 8679849 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_generation\records\E4_qwen_2b.config.json` · SHA256 `10ad7a348d4aad6e586df47faee21cbb3a8bdc153f28b609dd1d99b91c11582c` · 127 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_generation\records\E4_qwen_4b.jsonl` · SHA256 `1721a7dcda289f45767b0fe85f6f44fa184cca1034d33c0985766185663a7fde` · 8730998 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_generation\records\E4_qwen_4b.config.json` · SHA256 `4cd95f4aefef51235e6c1dfec97b33c1e0771d7264748805d8decb76cd85f39e` · 127 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_generation\records\E4_qwen_9b.jsonl` · SHA256 `22604ad47f0c66e573ea400c73935e17c05f9f23f9f01829ccf6722272d5c1b8` · 8784716 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_generation\records\E4_qwen_9b.config.json` · SHA256 `74990917c72f2af9d91533e1323681f9a298fff5114b4dc658ee39a889533fae` · 127 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_generation\records\E5_lfm.jsonl` · SHA256 `84caa22b3abecafae0e8c9b1934c8b2ac95fc527ee0efea7dce80026015a6cc1` · 17143655 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_generation\records\E5_lfm.config.json` · SHA256 `9ecae2ce21c2e0da9664d97aa18f9cf0b3d9eb68d258bfacf4a17d06b95744ca` · 123 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_generation\records\E6_qwen_2b.jsonl` · SHA256 `0f235d929aa75416a5112d30c367a656c7e3c9bca777dad8b1053cc845128303` · 17010569 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_generation\records\E6_qwen_2b.config.json` · SHA256 `8868652da4cb004caf857a9399915816c7756ea3148fe6a991a5bfca1eb2dc9c` · 127 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_generation\records\E6_qwen_4b.jsonl` · SHA256 `e65b657e885f0e989246fd9edcea5610a1ac890cb3523bf2e3d7163d03dd2131` · 17178166 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_generation\records\E6_qwen_4b.config.json` · SHA256 `35ab289134d3e326a8d469c82a45034b4bda3b32d485077d50139a61e9980656` · 127 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_generation\records\E7_qwen_2b.jsonl` · SHA256 `b09e74b25a2f7969cb466f6383ccc49f4fd6200b23ea64b1df33952dd1e4e37e` · 3025089 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_generation\records\E7_qwen_2b.config.json` · SHA256 `2d6471d82a952100f5f1ace37d124406a642d8e44a92b44f8af9cb07a4d65549` · 127 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_generation\records\E7_qwen_4b.jsonl` · SHA256 `95ec7c0c6aeeee7ae75171a12e4ddbc425086effcf0c74e12418b1584a561044` · 3093070 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_generation\records\E7_qwen_4b.config.json` · SHA256 `d43083258af676ff66fa1501664cb49737fa7ae8a1c59fa9c512d9833324da5f` · 127 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_generation\records\ROUTE_DEV_qwen_2b.jsonl` · SHA256 `f25f465d95cb7fe78ee7d524b1739585713819aa1d01dd2259921ba25db4ff47` · 15285132 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_generation\records\ROUTE_DEV_qwen_2b.config.json` · SHA256 `03307223aca4e007d964dcd6c41cc354afabe42d38221b453f0e323c301d53d3` · 134 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_generation\records\ROUTE_TRAIN_qwen_2b.jsonl` · SHA256 `dbfc77898cc911869d086a97db390ffbe0bb0fd712bafea61acaf1bc3841a7d2` · 50500279 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_generation\records\ROUTE_TRAIN_qwen_2b.config.json` · SHA256 `cda89efd526abf42dfb02a7ef92cebbecd690aeac1553a22c5ad0dfbc4f01ef5` · 136 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_generation_e3\EXPERIMENT_MANIFEST.json` · SHA256 `0c0614c1f8dd2094c8c50356306654734ab4778c8aec5b571bb291f5d307cfac` · 514530 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_generation_e3\FROZEN_INPUTS.json` · SHA256 `fba2df638ea37c4b1475d30c671e86583c493c271c64c9d068e577af5ee5c222` · 2887112 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_generation_e3\records\E3_qwen_2b.jsonl` · SHA256 `3e15fd12901414bcf4b777971007cfdcb5700e5b06e4f318de04bdb4b58403ec` · 6480369 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_generation_e3\records\E3_qwen_2b.config.json` · SHA256 `ca9fd69448e393bf3079670f975e68cc6c4c3e697a619ed445e2366ffb4263c7` · 127 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_generation_e3\records\E3_qwen_4b.jsonl` · SHA256 `a7d5a59d88498d38491e20b0bc45fcd90790ec63279e75b46c59d85a2094630f` · 6544263 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_generation_e3\records\E3_qwen_4b.config.json` · SHA256 `020614257f5dbe45beda5dec0c3b93764f4c2290deba77b5b2ae168a58232479` · 127 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_generation_e3\records\E3_qwen_9b.jsonl` · SHA256 `fcaf1c05488d090240dcb1f7d1263f9e6e53e859753c2e249a414967094820b3` · 6660960 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_generation_e3\records\E3_qwen_9b.config.json` · SHA256 `6039d8bbd66fc17d7ae55e00e45f49e41519df071ba59796d8d174a9dbcbce0e` · 127 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_generation_e2\EXPERIMENT_MANIFEST.json` · SHA256 `fe9396d6c0e26853354b518f84379f7bbd69e59411dcd7dbaf29972ed90d10f1` · 980898 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_generation_e2\FROZEN_INPUTS.json` · SHA256 `48a7a63010a7a0c84a2c21deaf307875458c188d05a22643d7038c390a9ebd35` · 55998187 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_generation_e2\records\E2_qwen_2b.jsonl` · SHA256 `1cbfefabf14fc83cfc66debb3496e3bbc18bbac1259d3e5c290741542d0d96a9` · 68120430 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_generation_e2\records\E2_qwen_2b.config.json` · SHA256 `094ef4632c30e1fb90703d6823103ecfc96d242d5ef2c45f8f67f5cd8ae71735` · 127 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_generation_e2\records\E2_qwen_4b.jsonl` · SHA256 `da5112b01a96589e6ee3503f1a9274bbc73e69ba0d760996bf3676f256714de5` · 68840897 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_generation_e2\records\E2_qwen_4b.config.json` · SHA256 `003ab422def667882b0a52f8557a475e908f1903647f708ea3b9d30cff507ed2` · 127 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_generation_e2\records\E2_qwen_9b.jsonl` · SHA256 `5777b7fd2181227ad07d61b17757c1ef8d310e107c0e3a61add9d77984eb1514` · 69353339 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_generation_e2\records\E2_qwen_9b.config.json` · SHA256 `261de6b8a6445e77f7ddd65c263b7cc516377594668078321f653a6fab289df0` · 127 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\RETRIEVAL_STATISTICS.json` · SHA256 `c300b1480b3a7095430b5026b9241c3ce962fb08ddadf2fc45f2037d0f0cbbff` · 26817 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\R3_GPU_COMPLETION_VALIDATION.json` · SHA256 `b10cb5b52bb54606eada1fccfbedcb0a302c22a4bfdd6acd0c5f77563fbdef89` · 5124 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\R3_GPU_MIGRATION.json` · SHA256 `697ce7ac3e0c213072359f03232a3b81d5ca4d8e33f0b4907f7ce58af77bbc51` · 26779 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\records\E8_R512_hybrid_rerank_neighbor_gpu.jsonl` · SHA256 `70e39c961c76b0ccbfcde55bfc578a2fc9a11931c4e8aef545df2084193cd4cf` · 10765957 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\records\E8_R512_hybrid_rerank_neighbor_gpu.config.json` · SHA256 `d75e5e63c894ae542a7878d3a6d6d12c7ddeef6cd5d74ce05676e8118cba2575` · 38024 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\summaries\E8_R512_hybrid_rerank_neighbor_gpu.json` · SHA256 `396a1da3eff337e6b946c9d92f30d02b5b8746b3f918d0373f740fc6bcfebf3d` · 2152 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\warmups\E8_R512_hybrid_rerank_neighbor_gpu.jsonl` · SHA256 `753dceba8b9ba6ab74e83e806dc3ca0577990a1b5ff9c90bb51cda10fdabdfee` · 3982 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\warmups\E8_R512_hybrid_rerank_neighbor_gpu.config.json` · SHA256 `b7879744f907029e4fe764c72c7c8dc13ddd8d78e96ffcac46d3209b606ded76` · 324 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\records\ROUTE_CONTEXT_TRAIN_R512_hybrid_rerank_neighbor_gpu.jsonl` · SHA256 `2cdd36c14955f627be84959b277456b390a1634994d12e2ff3420cc820be43af` · 11688786 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\records\ROUTE_CONTEXT_TRAIN_R512_hybrid_rerank_neighbor_gpu.config.json` · SHA256 `917d8e71f1515e9fe9b91a894420a1857cd3614ca21a3121a6f51dc7aa7c08c0` · 49375 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\summaries\ROUTE_CONTEXT_TRAIN_R512_hybrid_rerank_neighbor_gpu.json` · SHA256 `cdb8e8776bf127ca37186bcaa10a0c00d5d64a93657d4767d2692bb7255a49de` · 2437 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\warmups\ROUTE_CONTEXT_TRAIN_R512_hybrid_rerank_neighbor_gpu.jsonl` · SHA256 `1ef277754e3bb33cdc896b1a0bb462a35f52b483f6714d08cd85022e08e51fdd` · 3984 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\warmups\ROUTE_CONTEXT_TRAIN_R512_hybrid_rerank_neighbor_gpu.config.json` · SHA256 `55fdf6807d1018f2ffea4134e25edb481751ea2fac63ee4e2bd50c734f0d8157` · 324 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\records\ROUTE_CONTEXT_DEV_R512_hybrid_rerank_neighbor_gpu.jsonl` · SHA256 `253d110197567cfe077e237428d50ac1fdb8c362a3c58f3244e969cf952ca0f2` · 3404204 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\records\ROUTE_CONTEXT_DEV_R512_hybrid_rerank_neighbor_gpu.config.json` · SHA256 `bdd67e2719943b29b970b8e16d5532c06f54e2781e000916291f63a6f8b909ff` · 35360 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\summaries\ROUTE_CONTEXT_DEV_R512_hybrid_rerank_neighbor_gpu.json` · SHA256 `201f3ba8ddfb0a58b950760a584180f4630c6f751e03b79ff78c969a3cafaa3c` · 2030 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\warmups\ROUTE_CONTEXT_DEV_R512_hybrid_rerank_neighbor_gpu.jsonl` · SHA256 `05ba1aa1ad8bf91eda46654fefd5d36e3ee50d0c1c1c202566ac32ef697f6f4a` · 3984 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\warmups\ROUTE_CONTEXT_DEV_R512_hybrid_rerank_neighbor_gpu.config.json` · SHA256 `5f55669ef2a3f7fc575eea43a63f7b9d320259e201ed1a665e2a160b5fb10837` · 324 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\records\E4_CONTEXT_R512_hybrid_neighbor.jsonl` · SHA256 `b84f5dbec226213cdc84c7dbdc79796f70301fd7a193e7bebc4efc68293041b7` · 5014461 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\records\E4_CONTEXT_R512_hybrid_neighbor.config.json` · SHA256 `743a9ec6154071ff14a7ac0ea9398fa3ad7a4b67a832f6cc2fbf323beeb9fc95` · 54501 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\summaries\E4_CONTEXT_R512_hybrid_neighbor.json` · SHA256 `65dd38034b596867d82658d4ba4db140374bada98a1bb35c8c611a58e2912334` · 912 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\records\E8_R512_bm25_neighbor.jsonl` · SHA256 `0637c9f6f31abe7ee80cdb392c502a18bfcc1b46de7203cfd13154c1eceb69f2` · 8903485 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\records\E8_R512_bm25_neighbor.config.json` · SHA256 `3112f89265af1670afb09e0c9f98406ab079172c3aa3f96b900c768b15c076bb` · 17182 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\summaries\E8_R512_bm25_neighbor.json` · SHA256 `79fe2417b330a620cdeed113ca24d2ef51b507b8b42ae06f5d814fb78ec7b35c` · 893 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\records\E8_R512_dense_neighbor.jsonl` · SHA256 `6acccdb9000213a8b1ee24d77967d3e6cd61fa2ea9bfe9ab31a02f4ce4793c31` · 8881943 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\records\E8_R512_dense_neighbor.config.json` · SHA256 `5e1ef9be465d2eed56ecba2d54c2fbfae2982ae3ce6cae729dca2871a0aec2be` · 17622 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\summaries\E8_R512_dense_neighbor.json` · SHA256 `c80837475ba04a4ba69be771e1609f1c09af93f40d180497785330e0950a1c87` · 895 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\records\E8_R512_hybrid_neighbor.jsonl` · SHA256 `6e2b59ced2177cc8643474ffc04c0111f3ba37e66316bb0579698fa0bc2871f6` · 8972108 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\records\E8_R512_hybrid_neighbor.config.json` · SHA256 `9d1c80e77746246f2df23b5fd9d7c6b45513fdf1811f9f0e549cb545a711766b` · 17624 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\summaries\E8_R512_hybrid_neighbor.json` · SHA256 `6c4cfce815f63adafb69cc9c71b17c2509882423fee8e56f7a94fdb6fc2341a5` · 955 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\records\TEST_CONTEXT_F512_O0_bm25_neighbor.jsonl` · SHA256 `e0ec15b9aea2a06ad4c86ff8f798410990dd83274fab68e3b2106fa719a1825c` · 9991259 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\records\TEST_CONTEXT_F512_O0_bm25_neighbor.config.json` · SHA256 `12118f445a17adb1efe636fee21ff21bc7ad149445be93d8878fc2d1a5e66b4f` · 24597 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\summaries\TEST_CONTEXT_F512_O0_bm25_neighbor.json` · SHA256 `a460cbe5687960ecc94bcc6c3ab7e94fc1dd695789379eea684965d2e6f8693d` · 1002 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\records\TEST_CONTEXT_F512_O0_dense_neighbor.jsonl` · SHA256 `65cbe75f02a773588276b2e04d4d61aa2a2195b19e0b7cd663c6c8c28bfb6805` · 10151337 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\records\TEST_CONTEXT_F512_O0_dense_neighbor.config.json` · SHA256 `7451f8fd53b6233dcf39eda6c0c1125a22d6cfbbc23fe02ab58600874734879a` · 25037 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\summaries\TEST_CONTEXT_F512_O0_dense_neighbor.json` · SHA256 `606d7fb27c47bd7b3495d7da7bb6abe01b02164792682da818f03b0a05858d37` · 1003 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\records\TEST_CONTEXT_F512_O0_hybrid_neighbor.jsonl` · SHA256 `8af1439521c2b1e999123dfd91946d88e77cca62e28a0a45711530bb92ba509a` · 10163027 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\records\TEST_CONTEXT_F512_O0_hybrid_neighbor.config.json` · SHA256 `3a688fe9bcd7f20fc92d2001b5ddc1b68e763f25b331752c3a0bcca43a634fcd` · 25039 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\summaries\TEST_CONTEXT_F512_O0_hybrid_neighbor.json` · SHA256 `3373a2793a96ac028a4a26878d8dc9599bc0c5f0a0cf871b95fd61a320a4709e` · 1011 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\records\TEST_CONTEXT_R512_bm25_neighbor.jsonl` · SHA256 `190265c482f0a5e204444f2640ee6a0e665a7d21e7d34ac736d0e3f582bfeed9` · 9540251 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\records\TEST_CONTEXT_R512_bm25_neighbor.config.json` · SHA256 `90212fa892843dac879e138cb3d4bb8729c93777f487d3f62207e15514f383b9` · 22192 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\summaries\TEST_CONTEXT_R512_bm25_neighbor.json` · SHA256 `f5ced61e290ba0c085c02735ae8edf076c7215bdab7a6fcefa5e00638103cd06` · 998 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\records\TEST_CONTEXT_R512_dense_neighbor.jsonl` · SHA256 `b150f7c440bb732de3cd413ef1a2193468190c3a8fff472b1ff6499d13d94170` · 9658299 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\records\TEST_CONTEXT_R512_dense_neighbor.config.json` · SHA256 `e9672dc97bcf49c2c9103ea67336c9788679cc0e5c2bb1586083bb8a17750b58` · 22632 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\summaries\TEST_CONTEXT_R512_dense_neighbor.json` · SHA256 `ab78b8a5ebed9d1a79c57bcac4b991e10db86f1e7a506fd7bd969a5565dd2afc` · 999 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\records\TEST_CONTEXT_R512_hybrid_neighbor.jsonl` · SHA256 `2123821c39540f813a7b3f7f16b174a69961a67a4fad30e38496a8636d24fae2` · 9690519 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\records\TEST_CONTEXT_R512_hybrid_neighbor.config.json` · SHA256 `372893678bbb559070395d1b69c8ce598b7d50747678b861412374621593607a` · 22634 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\summaries\TEST_CONTEXT_R512_hybrid_neighbor.json` · SHA256 `02124c184c8b76d26f63d77a5a27d71e12a2a46524d477b80a2d4881b4639e47` · 1001 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\records\TEST_CONTEXT_R512_hybrid_rerank_neighbor.jsonl` · SHA256 `a4366ef11e137a239da889eb0ef8528ec5990976d8f4dbfe8efdc1377e6cbfbc` · 11215893 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\records\TEST_CONTEXT_R512_hybrid_rerank_neighbor.config.json` · SHA256 `3d49a5a4ef3a5b5e259643eff5a8d04d5b6352b82798ce935ac948e1d7580631` · 47604 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\summaries\TEST_CONTEXT_R512_hybrid_rerank_neighbor.json` · SHA256 `92de6e75dfcacb1d03e3269909dafc80cfc3bcee74f478e062d4ebb78777d984` · 2128 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\CANDIDATE_SELECTION.json` · SHA256 `fdb2a034acfb932caec71aff4a67fc02fc9154aa9b35d47e78c18a7184bc8c74` · 359 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\CONTEXT_SELECTION.json` · SHA256 `9f6a01a5b7c83309f664d37c043f605b6e87e5248142c19105f2d74ebf696f51` · 1104 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_lora\train_2b_1024_seed42\training_result.json` · SHA256 `a154622c7c23d656545050f16b92697481f3c50c13a5312fe101127614273aea` · 4731 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_lora\train_2b_1024_seed42\TRAINING_AUDIT.json` · SHA256 `992dafabdab41a4c4c443a1e172c09ddff15024f2e7e9d717d27395021e60ae3` · 3515 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_lora\train_2b_1024_seed42\steps.jsonl` · SHA256 `6eb8227af30206a94c2eed25cc606d192be0b45cd45c49ff0d8cea31cc25c87c` · 98874 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_lora\train_2b_1024_seed42\steps.config.json` · SHA256 `e77817d6f35cc0eeb5dd25695d58e10e3d69506cf9835c589afc014a154bf38e` · 49878 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_lora\train_2b_1024_seed42\initialization_receipts.jsonl` · SHA256 `00a4c345bffcd6718fa7654c4cea83f343d20e917afc1be4dd6e4f9ccb3cf79a` · 1350 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_lora\train_2b_1024_seed42\resume_from_step_000005.json` · SHA256 `df6723741270b63e23794e9bdab23cef0042adbafbeb5d586a13181e64b9e268` · 1630 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_lora\train_2b_1024_seed42\resume_from_step_000010.json` · SHA256 `02c8c27e68c934310a287354054b12bc38be4e7cc56027b8b0ce869a3b677b06` · 1631 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_lora\train_2b_1024_seed42\checkpoints\step-000120\manifest.json` · SHA256 `03a73f69648da97caac90e09b1512cd3e7b4ba75ac3be83dca0315e1578c4a58` · 1152 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_lora\train_2b_1024_seed42\checkpoints\step-000121\manifest.json` · SHA256 `652fa464f2e0f45cf6f454e0a3c87401d48c93c3aeae49e2eb4b5948ff4d0681` · 1152 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_lora\train_2b_1024_seed42\checkpoints\step-000122\manifest.json` · SHA256 `509feca3e68f4ac12c1318c289102ae94809d3bb5e62f86506362187f140faba` · 1152 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_lora\train_2b_1024_seed42\checkpoints\step-000122\adapter\adapter_model.safetensors` · SHA256 `a532dc22942d84e4dceabacb64b936c191b37ebdfa9fda092f3f87464d0bbae4` · 14798456 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_lora\train_2b_1024_seed42\paired_test100\SUMMARY.json` · SHA256 `c47d9fb6bcdf975c4f0219b33bc4b47bfa52dcd78de6dab9bf837c2a74df3de9` · 2763 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_routing\online\SUMMARY.json` · SHA256 `d3fabde7044ecaf04c38dffbb9c763161c3d473ba00f92c0084de2c7df653378` · 1946 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_routing\online\records.jsonl` · SHA256 `2c76fc69bedeea7c7f36316e15c53122d33282bbe006b73facba4dcc068d7b93` · 11658417 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_routing\online\ONLINE_MANIFEST.json` · SHA256 `fbf8c5b9f2fecc18d1f43bd900b16edb8e42a059104d0c03a7ba26948981d7dd` · 7543 bytes

- `D:\MCS_26S1课程资料\个人项目\RAGScope\ragscope\routing\online_campaign.py` · SHA256 `896720edb18d12b73b0c78bd0c2be75eb3882d7f7ea654f195d9c20d5fb7f033` · 18707 bytes

- `D:\RAGScope_Runtime\models\bge-reranker-base\2cfc18c9415c912f9d8155881c133215df768a70\download_manifest.json` · SHA256 `87a9da253900e640f2f29b934b52e3570e9b73756756780c10a267c9acb789bc` · 3509 bytes

- `D:\RAGScope_Runtime\corpora\20261014_v2\queries_global_test.jsonl` · SHA256 `702048198abd4aa45861a1e28afbb28a9628cccdf0b07546e36619958226d301` · 497365 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2\ROUTING_QUERY_MANIFEST.json` · SHA256 `060bc704f5ebcc89903c1e5a3999ea63fb8bbafd2707e64d9a1b4c8bc2d9cd7f` · 18428 bytes

- `D:\RAGScope_Runtime\corpora\20261014_v2\queries_train.jsonl` · SHA256 `da5513fab70f0e39b2afbf231fa8be32790eb2e7038785b45b6df4cc912eff43` · 2440374 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\records\ROUTE_CONTEXT_TRAIN_R512_hybrid_neighbor.jsonl` · SHA256 `1d154e7a5a8004c9e50dc36fcab47465a22fbe86d59b0db88d2d428d3691be14` · 9934887 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\records\ROUTE_CONTEXT_TRAIN_R512_hybrid_neighbor.config.json` · SHA256 `64dd0b12112fd94c2f5138be4f80e95b54aeb0c8cd5ab2717e8c8ceec384ad0e` · 23285 bytes

- `D:\RAGScope_Runtime\corpora\20261014_v2\queries_dev.jsonl` · SHA256 `75226422faf4c645ad7892838fa906c8c6b98d57e041da0ef9eb815c1a10578e` · 400020 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\records\ROUTE_CONTEXT_DEV_R512_hybrid_neighbor.jsonl` · SHA256 `d0451df1e019c0e31636e8f7f96064442eba3a43ea95ab394f99a495e793815d` · 2906945 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\records\ROUTE_CONTEXT_DEV_R512_hybrid_neighbor.config.json` · SHA256 `6ac8a7437a40745e7b7b14a792217b69ee5de4e55e446803d4337f9974851fec` · 16281 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\records\E1_R512_hybrid_rerank_neighbor.jsonl` · SHA256 `16b0874d0988ab11453e2d403b85d2b67eec6094b774eb6bf8940689bb80ad2e` · 5624622 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\records\E1_R512_hybrid_rerank_neighbor.config.json` · SHA256 `62d9d3518b5b641f7cfecd1d4ac953c326d249bd4a90101529deaa39f9f0d6f9` · 18544 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\R3_GROUP_E1_VALIDATION.json` · SHA256 `9d740aac9441a460d8cff0f23f200e45ff495b660bfc9d64eb4b2651000d2742` · 2098 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\summaries\E1_R512_hybrid_rerank_neighbor.json` · SHA256 `0c874bc1fe16767127fb31bf3ff9ef0adc8d47a471abcf4a4383ced286ccb90d` · 1858 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\R3_GROUP_TEST_CONTEXT_VALIDATION.json` · SHA256 `4d0ee2ba01788b0778ba182f67440ee25f24adf4685ea70995c0042071afdb7d` · 2169 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\records\E8_R512_hybrid_rerank_neighbor.jsonl` · SHA256 `f87d1a24c265e80506c8bc4a88972715cdd769319a3fe6b0e58dd11c0f5d23f4` · 3068853 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\records\E8_R512_hybrid_rerank_neighbor.config.json` · SHA256 `6c8ae9939be56ea6a6a3efddc9d49f0b2213ced23ca0990ef0d67eff1673ba8d` · 37571 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\R3_RUNTIME\LOAD_STATE.json` · SHA256 `ac6d2105f2f0144f10515a2faf418286f67e712372ccf1de2419943506f880f3` · 131 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_retrieval\R3_RUNTIME\RESOURCE_PAUSE_PID30108.json` · SHA256 `8b6e9e388cab318355eed66315a3889c91e8ae6cf62b3d6559cd23bbb8b1d4b5` · 42834 bytes

- `D:\MCS_26S1课程资料\个人项目\RAGScope\ragscope\experiments\rerank_gpu.py` · SHA256 `91391bfbca782f2200febd2e2eda827e524d5ed93c83a09b884af403fed1adb5` · 13910 bytes

- `D:\MCS_26S1课程资料\个人项目\RAGScope\ragscope\experiments\rerank_campaign.py` · SHA256 `7f5ba4c888d8949ba31d407d5538fd4483754a695ed630593ddbfa57e4c7bac2` · 21387 bytes

- `D:\MCS_26S1课程资料\个人项目\RAGScope\ragscope\retrieval\verified_reranker.py` · SHA256 `5bf600542a5e0689f294657b5baf9f4aaee4e0cea608efd3c71b78781e573259` · 6963 bytes

- `D:\MCS_26S1课程资料\个人项目\RAGScope\ragscope\experiments\rerank_validation.py` · SHA256 `f8a82e999b4339437bac7471a45e1013d07b0c03eb0c6f0b1d1986685770f307` · 12973 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_lora\train_2b_1024_seed42\paired_test100\baseline.jsonl` · SHA256 `2dca04ab3b466e518239ccbc76b2176580e50416adba08ac80d08493e9d7faea` · 5508512 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_lora\train_2b_1024_seed42\paired_test100\baseline.config.json` · SHA256 `88b077cc4988c4a2677890ddb83b8b8fa2d459855eaf82a18d6422e5bd6742a7` · 5590 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_lora\train_2b_1024_seed42\paired_test100\adapted.jsonl` · SHA256 `58dc8ef8885d0695f170ef9ea65caab18dc3b3473edb655aa8e034db19e3affd` · 5576500 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_lora\train_2b_1024_seed42\paired_test100\adapted.config.json` · SHA256 `faf6629e7d6ff85fae182f98c3b132863de89f8502fdf8edeeeb363e50de3013` · 5818 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_lora\train_2b_1024_seed42\paired_test100\FROZEN_EVALUATION_INPUTS.json` · SHA256 `5fc064e350500e4049b3ee8a355296f421038626a1151706b9760e213623c092` · 6981574 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_routing\online\records.config.json` · SHA256 `232ae268ae66b12bed70d9305b45c834d8b7ce4e11362eeddefc584c22612f3a` · 4553 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_routing\REPLAY_fixed_bm25.json` · SHA256 `3d16bccf8483069874e8752f12db3018dad200b751afd67e52eb509b4911c1fd` · 281735 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_routing\REPLAY_fixed_hybrid.json` · SHA256 `f92e0ece531a50da1e148a093983e25445d23d2e76b1a7c15f8daf1981b3f9d0` · 282608 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_routing\REPLAY_fixed_hybrid_rerank.json` · SHA256 `f891838c87e164069960a2eae320bf63487328b532d660a4be2ad5fc86dcdebd` · 285915 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_routing\REPLAY_rules.json` · SHA256 `5067a895914a32861e50e10d8b61beb614ce222c82818af915287991909a34c3` · 285508 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_routing\REPLAY_linear.json` · SHA256 `794994e8cde1aac993c1dcff733c551a2448795a3844f218f7d28535ee16df96` · 300749 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_routing\REPLAY_mlp.json` · SHA256 `bc2f26995ae6241cf855e24c03bac74fd00202041008e04f62f6d8debf8e9920` · 300250 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_routing\REPLAY_lstm.json` · SHA256 `ddc11e0c2cb956317754a9126b7371148467ffca6c0d74ba687cc92bb7e36210` · 300100 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_routing\REPLAY_laya.json` · SHA256 `195f8177441b717a7c2ee7489af2608ea02ff3710eb6768f6a577c9e2e853fe7` · 631855 bytes

- `D:\RAGScope_Runtime\envs\models\laya_sdk_installation.json` · SHA256 `5a82013d2877af5d6c66f6e439918b90798bdc33834d0938d0d679a41d7eb47b` · 12066 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2\MAIN_ANALYSIS_EXECUTION.json` · SHA256 `75e71844bb6f0accca42c3bcd0068c9296bb839925c62bfa1f7b26f8a62352e3` · 2588 bytes

- `D:\MCS_26S1课程资料\个人项目\RAGScope\reports\v2\GENERATION_TABLES.md` · SHA256 `153ee09cf127ed19917451757131c9034fea1670d72ab82ca94c12fc0a02a5ad` · 3480 bytes

- `D:\MCS_26S1课程资料\个人项目\RAGScope\reports\v2\ERROR_CASES.json` · SHA256 `95ee7484ceeba7359ddfdb951f0882b02512d87fc1ea9f635eb04ec5e81183ba` · 22636739 bytes

- `D:\MCS_26S1课程资料\个人项目\RAGScope\reports\v2\ERROR_CASES.md` · SHA256 `a1eb70ef31b5d8c6a6a35c6984476f5ad1ba4478efe0388f54d40135584676f3` · 9286 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2\MAIN_FIGURES_VISUAL_QA.json` · SHA256 `caa7e70a0d5032810196541494ec6d27d7a514115f150c181bd96c173062b215` · 3368 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2\main_analysis_final.log` · SHA256 `e7e186a49badbe7110fcea8472b744a5914658b01e4d939ddb9ff90a3599b6e6` · 98 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2\main_error_cases_final.log` · SHA256 `37d1dc8f8354a777978f52c263df031b6687480ec1cc18835610e4b17efdf1ff` · 22 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2\main_figures_layout_fix.log` · SHA256 `f5eabf5973284d0ab9d3d1b591b3b28f2d3e99520533928ce6f0e480ff6e3b46` · 418 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2\sign_final_main_stages.py` · SHA256 `ca0ac9d1610ed8c07b3c4370aecf8c55e76bb78512bab317124cd45c1c50613c` · 5528 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_lora\train_2b_1024_seed42\paired_test100\INDEPENDENT_COMPLETION_AUDIT.json` · SHA256 `04c912f1e226f6e6a2b8cb93a4178cdd666082a70115ffaddad2601f7b53509f` · 4149 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_lora\train_2b_1024_seed42\paired_test100\ACTUAL_GPU_PREFLIGHT.json` · SHA256 `10a458657b7848ce8fd0b4441d0c3bb072a9637e8f14789b19938c3ba4182a62` · 1904 bytes

- `D:\RAGScope_Runtime\models\Qwen3.5-2B\15852e8c16360a2fea060d615a32b45270f8a8fc\model_manifest.json` · SHA256 `07973094192c7b4de7a92203eaf37554ec02182a8694a7c23bc734e8cf8a523a` · 6248 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_lora\train_2b_1024_seed42\paired_test100\warmup_baseline.jsonl` · SHA256 `83a73b2099e20069b05ae269d492c99d934ca011d63e0a9fb80252aa8e46ee7b` · 175374 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_lora\train_2b_1024_seed42\paired_test100\warmup_baseline.config.json` · SHA256 `443167a65dfb36c105f58771aee4cb5860a306f7f8373b58331734c6f57379ac` · 5618 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_lora\train_2b_1024_seed42\paired_test100\warmup_adapted.jsonl` · SHA256 `d682bf07a4cc0e845b9c91b6af3fdc812999baf3675816e5824997b674651784` · 174422 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_lora\train_2b_1024_seed42\paired_test100\warmup_adapted.config.json` · SHA256 `983a8c2df1ec245b0e5187072b63eb7602c1801f5c3f18ca38c79cb217d0a759` · 5846 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2\recovery\generation_post_reboot_20261014\audit_paired_bf16.py` · SHA256 `aeeae02577e1904832df59356f7f7be9840844ce24daa644c6b28f8e44d8b9d3` · 7866 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_lora\train_2b_1024_seed42\paired_test100\ROOT_OBSERVED_COMPLETION_WITNESS.json` · SHA256 `826d1853efa29ea129fd59e343db8d4d80d8bc8a5cb816ce7e60ca86f2aadf71` · 1100 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_routing\online\EXECUTION_START.json` · SHA256 `dde383bf4cbaaa1906c2113ff3463c4c2cc93b1e7e072977336442aa639ff0c6` · 1313 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_routing\online\REMOTE_GPU_REBIND_VALIDATION.json` · SHA256 `866ec682768ef57da2db74c68403e136afd98b296150fd447c352ad34003dadb` · 5198 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2\online_service_handoff\ROOT_ONLINE_SERVICE_HANDOFF.json` · SHA256 `527408f3b9a229cbd5fb467fdfb1f258c2c4131ea4fbbf89d9efd0f80ded8ccf` · 1445 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_routing\online\RELEASE_AFTER_ONLINE_COMPLETION.json` · SHA256 `d8f7981a69a703ad495ccaa99a49cc6d371903116c72d9391668d7b632afe308` · 3743 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_routing\online\INDEPENDENT_COMPLETION_AUDIT.json` · SHA256 `57f895d3329ebb6670b536c0841b9ecb6ebe9675cbd3240ddc660814c4a1b708` · 2843 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_routing\online\WORKER_EXECUTION.json` · SHA256 `c28fd32cca66ef36e709e8e6c683a6658afc30251e9368310abd31d22d6ca7be` · 2324 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2_routing\online\resource_samples.jsonl` · SHA256 `0caef468cddb4c3c4d1844d09ac66761f5109259b99920b2403d486468659d49` · 13309 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2\logs\online_formal_remote_gpu_fixed_20261014.log` · SHA256 `7a2beb66b3215216e79db969d054cbf1a4d1ff085e56a4a6c7e14d36349bb381` · 7764 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2\logs\online_independent_audit_20261014.log` · SHA256 `76e23e82578fd47685fa79c89e7bec617493ff3bbd5edc0ee0d405c310b3e958` · 988 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2\FINAL_OWNED_RESOURCE_RELEASE.json` · SHA256 `6086c4abf0c37efa729a33c562c061b9af72edb448b95dbd513d01ae213c2f31` · 2756 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2\generation_service.json` · SHA256 `2eba3148fc331b7e2fc01e0818703044daf71180adcd739951ff97be0b157cb6` · 760 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2\online_embedding\SERVICE_LIFETIME.json` · SHA256 `fbab7ff8515ff111d85f2495708bf3b5725d8598975e9e37717d9668b925aaa2` · 2556 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2\online_embedding\STOP_REQUEST.json` · SHA256 `7297940305c89882a1edab2184da962ea0b34eb97db9caf28376e86a8e32995e` · 329 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2\ROOT_PLANNED_UI_GPU_STOP_REQUEST.json` · SHA256 `56443bec1f57fc351793969b1d7e1dfe572a5813c32164384d10017247ae13dd` · 1258 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2\UI_FINAL_BROWSER_QA.json` · SHA256 `978163ba98f5380714ae8f42ecf55d5b671cd9cc543ecf6e33964705b1628c1c` · 5447 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2\gpu_lease.json` · SHA256 `884715ec4556624cd069b461b1b9cfc3483f0bab32f6f77c6cf2165f6d6b0cac` · 1287 bytes

- `D:\RAGScope_Runtime\runs\20261014_v2\FINAL_LOCAL_REPORT_COMMAND_EVIDENCE.json` · SHA256 `a620fb3825cc85a5e60ec4cc05cb09ee0afa772bd08675c0b0307682b80fbddd` · 3738 bytes

## 解释边界

- 所有结果均为原运行条件的描述性观测；多重对照为探索性，不据未校正区间声称显著或因果。

- 非流式调用的 TTFT 保持 N/A；device-total 为整个设备采样，包含其他进程且可能漏瞬时峰值。

- GPU R3 的 E8/ROUTE 成本与原 CPU E2 R3、CPU 在线部署不同，不能直接作为部署加速比。

- 重放 latency 是已记录组件成本之和，actual_online 才是实际请求总耗时；加载与驻留条件分别保留。

- 失败保留完整分母；不适用或缺失测量为 N/A/UNAVAILABLE，不补零来代替缺测。

- 人工审核状态 NOT_REVIEWED；局部报告生成不代表最终验收或默认路由获准启用。
