# CPU模型环境后续补充

2026-10-14。初始38包环境与lock保留在`D:/RAGScope_Runtime/envs/models/environment_manifest.initial_preparation.json`及内容寻址requirements旧lock。新增固定Laya SDK0.4.6、psutil7.2.2后，当前40包`uv pip check`通过。新lock SHA-256 `0a9f751bbbcd21195abb98e2ec1bdbfedc72ea2a2aaeb54e414dff38590d7a85`，清单与实装metadata已同步；仍为torch2.9.1+cpu，不涉及CUDA驱动或原环境。

Laya SDK来自已审阅[官方仓库固定提交](https://github.com/NandhaKishorM/laya/tree/c73908d96ff73c0062c340a11bf7c54ec7d8bc4d)，下载47个必要源码/构建/license文件，共1,241,303字节，逐文件Git blob SHA与固定revision tree核对。源码manifest位于`D:/RAGScope_Runtime/downloads/laya_sdk/c73908d96ff73c0062c340a11bf7c54ec7d8bc4d/source_manifest.json`。使用本地源码`uv pip install --no-deps --no-build-isolation`安装，避免改变已经锁定的模型依赖。

安装后通过importlib.metadata查位置/版本并读取文件，不import laya/Torch；44个安装后的laya源码/配置文件逐个SHA相同。凭据为models环境的`laya_sdk_installation.json`，日志`D:/RAGScope_Runtime/logs/laya_sdk_install.log`。Laya权重尚未加载，SDK安装不构成真实Laya路由实验完成。

psutil用于E0真实模型进程自身的每秒RSS与host RAM采样，避免将Windowsvenv启动器的少量RSS误当成模型进程。安装日志`models_psutil_install.log`。E0仅CPU执行，Laya与BGE-base未并载。

为R3正式应用入口另补diskcache5.6.3、loguru0.7.3、tiktoken0.14.0、rank-bm25 0.2.2及其Windows计时依赖，未改已有模型包。45包`uv pip check`通过；当前最终lock SHA为`0add3cfa267fd2f4c18dae8692b62bb60eb028679b2e08949abad54a4e9ff902`。实测导入retrieval_campaign/verified_reranker成功，sys.modules内Torch与Laya均未导入，未加载任何额外模型。上述40包中间版本lock同样已按hash保留。
