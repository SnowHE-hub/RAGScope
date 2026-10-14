# RAGScope v2 检索局部表

## 检索统计沿冻结选择与原分母报告

最终配置由dev选择：结构组R512、简单对照F512_O0，扩展均为neighbor；test不参与选型。下表是已有12组最终检索统计和5个预声明探索性配对对照的投影，按论文/来源聚类的2,000次bootstrap已经实际运行，本次构建没有重算或增加组合。失败保留原分母，不可答的证据指标为N/A。

GPU全局/ROUTE重排与封存CPU测试重排有设备/后端成本差异，不能把它们直接做部署加速比。单论文候选池不足50时保留实际数量；“候选深度50”不是每题实际有50块。未校正的探索性区间不能证明普遍显著改善。

来源整体状态：PASS；没有据test重新选型。

下表直接读取现有bootstrap统计，不重新定义指标或选择对照。实际N含失败；证据指标对不可答为N/A，聚类数随适用指标变化。

|run_id|实际N|失败|指标|mean|95% low|95% high|适用N|cluster N|resamples|
|---|---|---|---|---|---|---|---|---|---|
|E4_CONTEXT_R512_hybrid_neighbor|100|0|continuous_coverage|0.97|0.9215686274509803|1.0|100|29|2000|
|E4_CONTEXT_R512_hybrid_neighbor|100|0|full_reference_coverage|0.97|0.9215686274509803|1.0|100|29|2000|
|E4_CONTEXT_R512_hybrid_neighbor|100|0|full_unit_recall|0.97|0.9215686274509803|1.0|100|29|2000|
|E8_R512_bm25_neighbor|100|0|continuous_coverage|0.16|0.0891089108910891|0.23764907740774063|100|56|2000|
|E8_R512_bm25_neighbor|100|0|full_reference_coverage|0.16|0.0891089108910891|0.23764907740774063|100|56|2000|
|E8_R512_bm25_neighbor|100|0|full_unit_recall|0.16|0.0891089108910891|0.23764907740774063|100|56|2000|
|E8_R512_dense_neighbor|100|0|continuous_coverage|0.18|0.10890551011622901|0.247842920353982|100|56|2000|
|E8_R512_dense_neighbor|100|0|full_reference_coverage|0.18|0.10890551011622901|0.247842920353982|100|56|2000|
|E8_R512_dense_neighbor|100|0|full_unit_recall|0.18|0.10890551011622901|0.247842920353982|100|56|2000|
|E8_R512_hybrid_neighbor|100|0|continuous_coverage|0.20033333333333336|0.12138010235540757|0.28623517786561264|100|56|2000|
|E8_R512_hybrid_neighbor|100|0|full_reference_coverage|0.19|0.11204501915708813|0.2755272108843536|100|56|2000|
|E8_R512_hybrid_neighbor|100|0|full_unit_recall|0.20033333333333336|0.12138010235540757|0.28623517786561264|100|56|2000|
|E8_R512_hybrid_rerank_neighbor_gpu|100|0|continuous_coverage|0.14833333333333334|0.0873015873015873|0.2193877551020408|100|56|2000|
|E8_R512_hybrid_rerank_neighbor_gpu|100|0|full_reference_coverage|0.14|0.07920792079207921|0.21177361853832438|100|56|2000|
|E8_R512_hybrid_rerank_neighbor_gpu|100|0|full_unit_recall|0.14833333333333334|0.0873015873015873|0.2193877551020408|100|56|2000|
|TEST_CONTEXT_F512_O0_bm25_neighbor|200|0|continuous_coverage|0.6404634235660085|0.5671515580846226|0.7120267083203877|182|72|2000|
|TEST_CONTEXT_F512_O0_bm25_neighbor|200|0|full_reference_coverage|0.5824175824175825|0.5027568219423957|0.6594934408092301|182|72|2000|
|TEST_CONTEXT_F512_O0_bm25_neighbor|200|0|full_unit_recall|0.6160714285714286|0.5418839958299704|0.6896994324045407|182|72|2000|
|TEST_CONTEXT_F512_O0_dense_neighbor|200|0|continuous_coverage|0.8096183800653903|0.7542490304660718|0.8628380388883229|182|72|2000|
|TEST_CONTEXT_F512_O0_dense_neighbor|200|0|full_reference_coverage|0.7582417582417582|0.6953969205236954|0.8223387799237077|182|72|2000|
|TEST_CONTEXT_F512_O0_dense_neighbor|200|0|full_unit_recall|0.7742673992673992|0.7138070276497696|0.8347408527226962|182|72|2000|
|TEST_CONTEXT_F512_O0_hybrid_neighbor|200|0|continuous_coverage|0.7310011340475766|0.6715582590129872|0.788922535122235|182|72|2000|
|TEST_CONTEXT_F512_O0_hybrid_neighbor|200|0|full_reference_coverage|0.6703296703296703|0.6042780748663101|0.7333615819209037|182|72|2000|
|TEST_CONTEXT_F512_O0_hybrid_neighbor|200|0|full_unit_recall|0.6992521367521367|0.63847038512445|0.7595819621554916|182|72|2000|
|TEST_CONTEXT_R512_bm25_neighbor|200|0|continuous_coverage|0.7048992673992673|0.6336589735362426|0.7736105889724308|182|72|2000|
|TEST_CONTEXT_R512_bm25_neighbor|200|0|full_reference_coverage|0.6703296703296703|0.5918313922427609|0.7454545454545455|182|72|2000|
|TEST_CONTEXT_R512_bm25_neighbor|200|0|full_unit_recall|0.7048992673992673|0.6336589735362426|0.7736105889724308|182|72|2000|
|TEST_CONTEXT_R512_dense_neighbor|200|0|continuous_coverage|0.8365384615384616|0.777482962662651|0.8891759899250233|182|72|2000|
|TEST_CONTEXT_R512_dense_neighbor|200|0|full_reference_coverage|0.8076923076923077|0.7421143174250832|0.8663448844884488|182|72|2000|
|TEST_CONTEXT_R512_dense_neighbor|200|0|full_unit_recall|0.8365384615384616|0.777482962662651|0.8891759899250233|182|72|2000|
|TEST_CONTEXT_R512_hybrid_neighbor|200|0|continuous_coverage|0.8192307692307692|0.7568282757818858|0.8743128209468991|182|72|2000|
|TEST_CONTEXT_R512_hybrid_neighbor|200|0|full_reference_coverage|0.7967032967032966|0.7304880108071597|0.8550105421686747|182|72|2000|
|TEST_CONTEXT_R512_hybrid_neighbor|200|0|full_unit_recall|0.8192307692307692|0.7568282757818858|0.8743128209468991|182|72|2000|
|TEST_CONTEXT_R512_hybrid_rerank_neighbor|200|0|continuous_coverage|0.6978021978021978|0.6222459454796412|0.766174178004535|182|72|2000|
|TEST_CONTEXT_R512_hybrid_rerank_neighbor|200|0|full_reference_coverage|0.6593406593406593|0.5777619047619047|0.7340440905177715|182|72|2000|
|TEST_CONTEXT_R512_hybrid_rerank_neighbor|200|0|full_unit_recall|0.6978021978021978|0.6222459454796412|0.766174178004535|182|72|2000|

## 缺失组

```json
[]
```


## 预声明探索性配对对照

```json
{
  "global_hybrid_minus_bm25": {
    "classification": "exploratory",
    "id": "global_hybrid_minus_bm25",
    "left": [
      "structured",
      "hybrid"
    ],
    "metrics": {
      "continuous_coverage": {
        "confidence": 0.95,
        "contrast": "left_minus_right",
        "high": 0.10000724637681156,
        "low": -0.013408280150548192,
        "mean": 0.04033333333333333,
        "method": "paper_cluster_percentile",
        "n_clusters": 56,
        "n_resamples": 2000,
        "n_rows": 100,
        "seed": 42
      },
      "full_reference_coverage": {
        "confidence": 0.95,
        "contrast": "left_minus_right",
        "high": 0.088257134536983,
        "low": -0.02128231525966598,
        "mean": 0.03,
        "method": "paper_cluster_percentile",
        "n_clusters": 56,
        "n_resamples": 2000,
        "n_rows": 100,
        "seed": 42
      },
      "full_unit_recall": {
        "confidence": 0.95,
        "contrast": "left_minus_right",
        "high": 0.10000724637681156,
        "low": -0.013408280150548192,
        "mean": 0.04033333333333333,
        "method": "paper_cluster_percentile",
        "n_clusters": 56,
        "n_resamples": 2000,
        "n_rows": 100,
        "seed": 42
      }
    },
    "right": [
      "structured",
      "bm25"
    ],
    "status": "PASS",
    "task": "global"
  },
  "global_hybrid_minus_dense": {
    "classification": "exploratory",
    "id": "global_hybrid_minus_dense",
    "left": [
      "structured",
      "hybrid"
    ],
    "metrics": {
      "continuous_coverage": {
        "confidence": 0.95,
        "contrast": "left_minus_right",
        "high": 0.08890350877192975,
        "low": -0.04364194577352472,
        "mean": 0.02033333333333333,
        "method": "paper_cluster_percentile",
        "n_clusters": 56,
        "n_resamples": 2000,
        "n_rows": 100,
        "seed": 42
      },
      "full_reference_coverage": {
        "confidence": 0.95,
        "contrast": "left_minus_right",
        "high": 0.08002020202020191,
        "low": -0.05406633906633907,
        "mean": 0.01,
        "method": "paper_cluster_percentile",
        "n_clusters": 56,
        "n_resamples": 2000,
        "n_rows": 100,
        "seed": 42
      },
      "full_unit_recall": {
        "confidence": 0.95,
        "contrast": "left_minus_right",
        "high": 0.08890350877192975,
        "low": -0.04364194577352472,
        "mean": 0.02033333333333333,
        "method": "paper_cluster_percentile",
        "n_clusters": 56,
        "n_resamples": 2000,
        "n_rows": 100,
        "seed": 42
      }
    },
    "right": [
      "structured",
      "dense"
    ],
    "status": "PASS",
    "task": "global"
  },
  "paper_hybrid_minus_bm25": {
    "classification": "exploratory",
    "id": "paper_hybrid_minus_bm25",
    "left": [
      "structured",
      "hybrid"
    ],
    "metrics": {
      "continuous_coverage": {
        "confidence": 0.95,
        "contrast": "left_minus_right",
        "high": 0.1726591506847865,
        "low": 0.06602283427495292,
        "mean": 0.11433150183150183,
        "method": "paper_cluster_percentile",
        "n_clusters": 72,
        "n_resamples": 2000,
        "n_rows": 182,
        "seed": 42
      },
      "full_reference_coverage": {
        "confidence": 0.95,
        "contrast": "left_minus_right",
        "high": 0.1860496041563582,
        "low": 0.07367826345524828,
        "mean": 0.12637362637362637,
        "method": "paper_cluster_percentile",
        "n_clusters": 72,
        "n_resamples": 2000,
        "n_rows": 182,
        "seed": 42
      },
      "full_unit_recall": {
        "confidence": 0.95,
        "contrast": "left_minus_right",
        "high": 0.1726591506847865,
        "low": 0.06602283427495292,
        "mean": 0.11433150183150183,
        "method": "paper_cluster_percentile",
        "n_clusters": 72,
        "n_resamples": 2000,
        "n_rows": 182,
        "seed": 42
      }
    },
    "right": [
      "structured",
      "bm25"
    ],
    "status": "PASS",
    "task": "paper"
  },
  "paper_hybrid_minus_dense": {
    "classification": "exploratory",
    "id": "paper_hybrid_minus_dense",
    "left": [
      "structured",
      "hybrid"
    ],
    "metrics": {
      "continuous_coverage": {
        "confidence": 0.95,
        "contrast": "left_minus_right",
        "high": 0.03314601566098567,
        "low": -0.07063188848754284,
        "mean": -0.01730769230769231,
        "method": "paper_cluster_percentile",
        "n_clusters": 72,
        "n_resamples": 2000,
        "n_rows": 182,
        "seed": 42
      },
      "full_reference_coverage": {
        "confidence": 0.95,
        "contrast": "left_minus_right",
        "high": 0.0378429788484136,
        "low": -0.062164648910411624,
        "mean": -0.01098901098901099,
        "method": "paper_cluster_percentile",
        "n_clusters": 72,
        "n_resamples": 2000,
        "n_rows": 182,
        "seed": 42
      },
      "full_unit_recall": {
        "confidence": 0.95,
        "contrast": "left_minus_right",
        "high": 0.03314601566098567,
        "low": -0.07063188848754284,
        "mean": -0.01730769230769231,
        "method": "paper_cluster_percentile",
        "n_clusters": 72,
        "n_resamples": 2000,
        "n_rows": 182,
        "seed": 42
      }
    },
    "right": [
      "structured",
      "dense"
    ],
    "status": "PASS",
    "task": "paper"
  },
  "paper_structured_minus_simple_hybrid": {
    "classification": "exploratory",
    "id": "paper_structured_minus_simple_hybrid",
    "left": [
      "structured",
      "hybrid"
    ],
    "metrics": {
      "continuous_coverage": {
        "confidence": 0.95,
        "contrast": "left_minus_right",
        "high": 0.13766717250583574,
        "low": 0.04007023242620717,
        "mean": 0.08822963518319256,
        "method": "paper_cluster_percentile",
        "n_clusters": 72,
        "n_resamples": 2000,
        "n_rows": 182,
        "seed": 42
      },
      "full_reference_coverage": {
        "confidence": 0.95,
        "contrast": "left_minus_right",
        "high": 0.18079246303642263,
        "low": 0.07099547511312218,
        "mean": 0.12637362637362637,
        "method": "paper_cluster_percentile",
        "n_clusters": 72,
        "n_resamples": 2000,
        "n_rows": 182,
        "seed": 42
      },
      "full_unit_recall": {
        "confidence": 0.95,
        "contrast": "left_minus_right",
        "high": 0.17107317722901516,
        "low": 0.06676859318996416,
        "mean": 0.11997863247863248,
        "method": "paper_cluster_percentile",
        "n_clusters": 72,
        "n_resamples": 2000,
        "n_rows": 182,
        "seed": 42
      }
    },
    "right": [
      "simple",
      "hybrid"
    ],
    "status": "PASS",
    "task": "paper"
  }
}
```


## 原统计协议

```json
{
  "classification": "exploratory",
  "comparisons": [
    {
      "id": "paper_hybrid_minus_bm25",
      "left": [
        "structured",
        "hybrid"
      ],
      "right": [
        "structured",
        "bm25"
      ],
      "task": "paper"
    },
    {
      "id": "paper_hybrid_minus_dense",
      "left": [
        "structured",
        "hybrid"
      ],
      "right": [
        "structured",
        "dense"
      ],
      "task": "paper"
    },
    {
      "id": "paper_structured_minus_simple_hybrid",
      "left": [
        "structured",
        "hybrid"
      ],
      "right": [
        "simple",
        "hybrid"
      ],
      "task": "paper"
    },
    {
      "id": "global_hybrid_minus_bm25",
      "left": [
        "structured",
        "hybrid"
      ],
      "right": [
        "structured",
        "bm25"
      ],
      "task": "global"
    },
    {
      "id": "global_hybrid_minus_dense",
      "left": [
        "structured",
        "hybrid"
      ],
      "right": [
        "structured",
        "dense"
      ],
      "task": "global"
    }
  ],
  "confidence": 0.95,
  "failure_policy": "retained_zero_for_answerable_evidence_metrics; unanswerable_not_applicable",
  "frozen_before_final_neural_test_contexts": true,
  "interpretation": "Unadjusted exploratory percentile intervals; no confirmatory significance or deployment conclusion. Missing runs remain unavailable; no test-based reselection.",
  "metrics": [
    "full_reference_coverage",
    "continuous_coverage",
    "full_unit_recall"
  ],
  "n_resamples": 2000,
  "qasper_cluster": "paper_doc_id",
  "seed": 42,
  "squad_cluster": "article_cluster_id",
  "statistic": "question_weighted_mean_with_entire_source_document_clusters_resampled",
  "version": "retrieval_statistics_v1"
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
