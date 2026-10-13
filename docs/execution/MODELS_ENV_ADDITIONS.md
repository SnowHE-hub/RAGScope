# CPU模型环境后续补充

2026-10-14。初始38包环境与lock保留在`D:/RAGScope_Runtime/envs/models/environment_manifest.initial_preparation.json`及内容寻址requirements旧lock。新增固定Laya SDK0.4.6、psutil7.2.2后，当时40包`uv pip check`通过。该lock SHA-256 `0a9f751bbbcd21195abb98e2ec1bdbfedc72ea2a2aaeb54e414dff38590d7a85`，清单与实装metadata同步；仍为torch2.9.1+cpu，不涉及CUDA驱动或原环境。

Laya SDK来自已审阅[官方仓库固定提交](https://github.com/NandhaKishorM/laya/tree/c73908d96ff73c0062c340a11bf7c54ec7d8bc4d)，下载47个必要源码/构建/license文件，共1,241,303字节，逐文件Git blob SHA与固定revision tree核对。源码manifest位于`D:/RAGScope_Runtime/downloads/laya_sdk/c73908d96ff73c0062c340a11bf7c54ec7d8bc4d/source_manifest.json`。使用本地源码`uv pip install --no-deps --no-build-isolation`安装，避免改变已经锁定的模型依赖。

安装后通过importlib.metadata查位置/版本并读取文件，不import laya/Torch；44个安装后的laya源码/配置文件逐个SHA相同。凭据为models环境的`laya_sdk_installation.json`，日志`D:/RAGScope_Runtime/logs/laya_sdk_install.log`。Laya权重尚未加载，SDK安装不构成真实Laya路由实验完成。

psutil用于E0真实模型进程自身的每秒RSS与host RAM采样，避免将Windowsvenv启动器的少量RSS误当成模型进程。安装日志`models_psutil_install.log`。E0仅CPU执行，Laya与BGE-base未并载。

为R3正式应用入口另补diskcache5.6.3、loguru0.7.3、tiktoken0.14.0、rank-bm25 0.2.2及其Windows计时依赖，未改已有模型包。当时45包`uv pip check`通过；该lock SHA为`0add3cfa267fd2f4c18dae8692b62bb60eb028679b2e08949abad54a4e9ff902`。实测导入retrieval_campaign/verified_reranker成功，sys.modules内Torch与Laya均未导入，未加载任何额外模型。上述40包中间版本lock同样已按hash保留。

为最终同进程CPU重排UI，按主执行器授权补入与core一致的Streamlit1.65.0及缺少的26个依赖。安装前确认没有models环境活动模型进程，保存45包manifest/lock原字节和哈希至`history/ui_streamlit_20261014/`。通过实际core metadata递归核验依赖约束，提取项目`requirements-v2.lock`中的确切版本和wheel哈希；先以旧45包约束、`--offline --no-deps --require-hashes --dry-run`确认只新增27包，再使用同一命令参数安装。全程复用D盘uv缓存，没有重新下载模型或更改core/train环境。

当前72包`uv pip check`通过，45个原包版本逐一确认完全未变；包括Torch2.9.1+cpu、Transformers4.57.6、sentence-transformers5.2.0、HF Hub0.36.2。新完整lock SHA为`4248ab17be28f682c46c8425870a5e5361ca55ad19a37ef5342dc246356addee`。Laya SDK凭据SHA仍为`5a82013d2877af5d6c66f6e439918b90798bdc33834d0938d0d679a41d7eb47b`，其44个源文件核验未变。实际Streamlit及verified_reranker模块导入成功，Torch/Transformers/Laya没有导入或初始化GPU；模型/路由输入配置没有改变。

完整安装收据为`D:/RAGScope_Runtime/envs/models/history/ui_streamlit_20261014/COMPLETION.json`，新env manifest引用其SHA。日志为`models_streamlit_dry_run.log`、`models_streamlit_install.log`、`models_streamlit_pip_check.log`和`models_streamlit_import_and_versions.log`。依赖安装通过不等于真实R3 UI已验收，后者仍需就绪index与资源窗口。
