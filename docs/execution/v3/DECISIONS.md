# v3 决策记录

2026-10-15T01:45:06.552349+11:00：完整任务书/预算/诊断已核SHA并保存。纯审计CPU-only四线程，新campaign独立；旧v2与用户文件只读保护。先回原journal重算，后评分校准和有界实验；H01–H12均是待验证假设。S6保留1800请求，所有失败/预热/裁判计账。

2026-10-14T15:42:56.373763+00:00：实际CPU原journal审计5520/39/32，独立F1/EM全部一致；官方v0.3 evaluator重算5720条（含LoRA200），实际F1差异0。LoRA互斥分解：新增拒答23题贡献-0.0979273，改善28题+0.1239351，仍回答变差23题-0.0514381，格式/请求3题-0.0127749，不变15题，另8不可答；重构净差-0.0382051。EOS tokenizer248046与model.config248044不同仅为待验证线索，20数值mask检查PASS不能替代实际监督自检。
旧全局题真实文件为queries_global_test.jsonl，原SOURCE_INVENTORY错误候选名缺失保留并在身份补充报告解析；E8 Hybrid100全部可答/完整证据19，适用分母确为100。模型110声明文件全部HASH匹配，硬链接去重实际核19292924146字节；5环境读取未升级。首次身份审计把项目内stopwords来源误当模型缓存越界，原日志保留，一次修复后exit0。
封存新D_confirm100/50论文、H_QA200/100论文，扫描230旧journals保守排除510论文，包括本轮schema读取露出的all_queries第一条所在论文。H_scope/H_tasks仍待构造，未称S0全部完成。裁判60组构造在实际判分前发现购买不排除出售的歧义，V1保留，另立V2明确否定；120自然盲评上下文companion已完成，人工NOT_REVIEWED。
语义环境v3_metrics复用train/reports只读site-packages，仅在新env安装依赖；pip分发bert-score0.3.13但包内部__version__0.3.12，记录两者，不默改身份。GPU初次守卫误将Codex宿主ChatGPT.exe当未知，保留拒绝日志，核PID11276/create1791922058.5791745及WindowsApps/OpenAI.Codex准确路径后修复。第二次GPU已算64/6342对，RAM2850594816低于3GiB守卫而exit1/释放GPU到12MiB；不放宽门槛、不清理用户进程。唯一兼容替代为v3_metrics_cpu复用models/reports CPU torch2.9.1/TF4.57.6，独立cpu命名结果完整6342对，不混入GPUpartial。当前live exec31866；总新生成请求0，S6额度1800完整保留。

2026-10-14T17:26:36.485169+00:00：RoBERTa默认GPU64/6342、CPU64缓存832/6342、CPU8缓存232/6342均实际触3GiB守卫退出1；三个完整行前缀和日志保留、不拼合、默认计分标BLOCKED_RESOURCE。NLI base较小，同CPU库实际130/130构造关系判断、60/60组通过，actualexit0、2620 encoder tokens、42.53秒；不是自然语言准确率或人评。
只做一次编码器方法替代：复用已核的MoritzLaurer DeBERTa-v3-base NLI权重，官方BERTScore支持对应base架构9层，FP32/CPU/encoder batch4/outer8固定先声明，明确不是英文默认RoBERTa且与NLI共享权重，不算两个独立裁判。全6342对/5720旧答案实际exit0、342932 encoder tokens、0截断、1070.41秒。R2语义proxy 2B/4B/9B=.66054/.53194/.47452；Oracle=.69983/.63792/.57300；LoRA=.65368→.54406，均不是事实准确率。原WordF1未重写。
LoRA语义差值-.1096233；按原WordF1互斥组，新拒答23贡献-.1466141、Word改善28贡献+.0648651、Word仍回答变差23贡献-.0117090、Word不变15贡献-.0022472、格式失败3贡献-.0139180，贡献重构PASS。23题仍须独立question/context充分性评判，不能把标注覆盖直接当YES。
已冻结558问题/3057独特原答文本的原始引用片段分割输入；没有gold/reference或方案名发给分割器，尚0模型调用。模型数字符位置不可靠风险须在首次调用前用quote-only协议另立配置，Python定位原文并校验全字符/否定/条件覆盖，不把看过gold的重写当旧答案。S1协议诊断实际32失败（含LFM29）全部不能仅清理包裹恢复；空上下文schema只限制cite数量，不强制abstain，严格材料prompt才限制已有知识。
固定HF 2B revision15852e8c...的generation_config.json公开端点404，旧目录未修改；不能把EOS差异说成漏下载bug。NLI自然样本审核、原子事实/支持/引用、23新拒答核对仍PENDING，S2-S6尚未运行。全30定向功能测试actualexit0，不能代替收益。接下来新增v3私有Ollama生命周期/预算适配；旧serve_generation硬编码v2，不能直接执行写旧run。当前无本项目驻留模型，新LLM请求0/S6完整保留1800。

2026-10-15 S1：QA 拼接裁判未通过工程门槛；采用参考不可见的窄机械完句并实际完成80/80构造校准，其他片段保留UNKNOWN。CPU裁判与GPU分割并发导致371组请求前RAM阻断，保留全部证据；第二次有界调度修复仅续跑预算未发出的组，资源不足即停批。详见 S1_RESOURCE_AND_QA_AUDIT.md。

S0 辅助题：40 道合成范围题已封存（20指定论文、20无指代）；30篇任务来源原文已冻结，均来自原预留50篇，独立核验与主分割241篇论文零交叉。初版辅助守卫错误地读 paper_ids 默认空集合，已保留初版脚本与收据；新增验证使用真实 doc_ids 且失败封闭，确认实际交叉为0，未改变任何题或主封存。H_tasks参考尚待生成，不能称任务题库完成。

S1 规模差异实际分解：R2的9B相对2B文字差-.1496723，替代语义差-.1860228；8条可答拒答即使全部变满分，最多+.043956，不能解释全差。yes/no的篇幅代理偏差已定位：20题中9B极性前缀17次与任一参考相符、2B16次，WordF1却约.05356/.8。5题参考极性有分歧；在另报的15题一致参考子集中分别12/15和11/15，WordF1约.04843/.73333。这个诊断不验证解释正确或证据支持，不替换原评分，优先进入S2简洁答案对照。

LoRA原始后端输出P50 67.5→176 tokens，mean 79.45→149.23；重复4gram平均下降而非上升。原停止原因、TTFT和decode速度没有记录，不补造；长度偏移相容于耗时上升，但不能单独解释3.62倍或判定EOS bug。

分割最终实际558请求/548通过/10契约失败，session57682 actualexit2；服务卸载且实际退出，原进程保持不动。GPU NLI实际210构造校准全通过、session46755 actualexit0。12条评分冒烟没有可校准关系而全UNKNOWN，原空缓存摘要bug导致session29339 exit1，修复后保留原12记录并exit0。全量GPU在session80992、流式修复66941均因3GiB守卫停止（最后仅差约15MiB），不降低门槛、不改WSL；唯一兼容改为已校准CPU同FP32模型，session24620真实运行，GPU部分保留且不与CPU分数混合。

后续CPU24620实际exit1（335评分/230关系），有界NLI全量到BLOCKED，不再反复加载；所有原结果、UNKNOWN和未执行分母保存。9B上下文校准59/60；首次104条后RAM停批，单次resume仅补13条，117完成。23LoRA新增拒答按独立材料proxy拆为充分8、不足12、不确定3。

S2完整24题三模型G0–G3矩阵864实际结果，通过独立计分与token一致核验；新增720、旧G0复用144、另72真实夹具。G1对4B/9B文字改善而2B下降；G2的52失败大多错引用，不把JSON可解析说成恢复成功。官方固定卡2B非思考text采样1.0/top_p1.0/presence2.0；4B/9B general为.7/.8/1.5，全部参数Go接口源已核验。相邻reasoning段落曾覆盖general解析值，已在首次真实调用前修复并功能测试。

H_tasks30论文teacher请求完成：16初版有效任务，20论文未通过原文/完整句检查。独立任务拆分、只允许空白差异的源匹配后冻结20有效任务，29无效分支仍保留，未改词语/标点、未补造参考；全部人造AI标注、NOT_REVIEWED、非操作者盲测，不并入官方QA总分。

S2补充真实42调用完成：36完整Oracle核心+同论文低词汇重合非标注材料，6额外prior；最终S2共840新请求。长度最大差2.567%，2题真实检索无gold evidence位置，明确以中间位置作替代。12题子集4B padded-clean文字F1下降.15051；2B下降.03522、9B下降.00417。旧暴露诊断，不宣称总体因果或holdout改善。

第一次补充运行session80469 actualexit1：错误传3GiB给另含1GiB余量的require_ram，总门槛变4GiB。修正为2+1GiB，实际已完成1条复用，41条续跑，无重发。复用封装首行record_hash包含旧hash的缺陷被独立校验发现；精确验证旧hash参加运算且全部原推理字段相同后，新建42条verified投影；两个原journal/config/log原样保留，不重算模型输出或改变分母。

S3零生成取证500旧记录：独立三项证据度量与原记录零差；9519窗口/6056候选无512预算、全文字符覆盖或降序异常。不能指控只看第一段。top2均值重用同一真实窗口分数，在当前两组selected证据覆盖差均0；这不证明最终上下文质量不变。

S3旧同预算400条core-first：论文内Hybrid连续覆盖+2.400pp、重排+4.031pp，原全库Hybrid -2.5pp、GPU重排+2.367pp；各组改善/退步逐题保留。论文内Hybrid完整参考覆盖-1.087pp，禁止仅选有利指标宣布部署。必须新D_confirm再验证。

唯一新增重排Qwen3-Reranker-0.6B固定e61197ed45024b0ed8a2d74b80b4d909f1255473，D盘1,207,484,795字节校验。官方前后缀/no-yes final logits，BF16 GPU实际12组24评分全部排序正确；只是构造夹具，不是自然检索胜出。非生成评分对预算独立200000 ledger，当前24对；不消耗生成tokens。单GPU进程实际exit0、未恢复用户原训练。

新源文档级BM25实际构建1585论文并重载校验。D_search120题/240真实查询：原问题top1/5/10=.275/.4/.475；明确加入可见选定论文的原题名后=.991667/1/1。题名来自用户scope对应源文档，属于明确不同的输入条件；不得写作原E8提升。Dense文档级与全库块旁路仍待实现。

本轮新功能回归5/5实际通过。v2基线d431f646全356受控文件逐一比对，内容差0；25文件只有既存CRLF/LF差异，不宣称byte-identical。第一次临时保护审计的git show使用Windows反斜杠失败，没有写出PASS；专用audit_v2_preservation修复路径处理后actualexit0，完整证据保存。

S3源准备初始85266 actualexit1，4284/4755后可用内存3011874816B低于3GiB；所有原产物保留。模型工作终态后同身份单次resume51493 actualexit0，仅补471项，4755齐全。三组R512/F512_O0/P256_1024真实块数19673/17194/41662，前缀含原题名/章节但引用与生成保持canonical原文。

现有Qwen3-Embedding-0.6B官方Q8_0权重SHA06507c7b…3e439已硬链接复用注册项目专用tag，无新权重下载、无旧tag覆盖；私有GPU后端0.34.4严格不截断。四组构造向量夹具真实通过、12×1024向量/缓存复读完全一致、api/ps证实GPU驻留。新GPU/Ollama身份与旧CPU/llama身份分开。六预选原CPU索引向量桥接cos最小.999703，只是数值proxy，不证明全体排序/成本等价。

真实文档Dense1585与开发query向量220完成；只D_search/D_confirm输入，没读H_QA问题或gold。原未定位问题D_search/D_confirm文档top5匹配：BM25 .4/.39，Dense .3333/.22，等权融合 .4083/.39。歧义原问题不作为唯一目标优化依据；不给指定论文任务增加全库查找，全球块旁路仍必须实现。

四实际稀疏索引采用与原ArrayIndex一致的BM25参数/分词，压缩实现逐分数对rank_bm25对照通过2功能测试。D_search120题/101证据适用同2048+neighbor：完整参考覆盖R512body .613861、R512metadata .603960、F512meta .495050、P256meta .564356。P256连续覆盖较高但完整集合更差，不能只选有利指标。按预设完整集合/单元召回/简单性优先选R512meta进入一次有限Dense确认，不声称胜出。

实际向量编码是非生成模型请求，输入/输出向量/后端耗时在EMBEDDING_REQUESTS与embedding_responses单独落盘，源输入集合固定；生成LLM预算仍1605/6000，S6预留1800不动。稠密块索引只做R512body与已预选metadata两条，GPU串行；每请求核验owner/lease，未知GPU消费者检查固定5秒，无取消3GiB守卫。

S5独立CPU诊断：全部972样本数值mask/EOS/JSON/引用ID可见性0失败；固定20样本实际解码包含原assistant目标及im_end。841样本是第一合法人工参考证据、131负例取第一真实段落。训练prompt P50 350.5/最大891，部署BF16配对实际输入P50 2147.5/最大2318。引用ID合法不等于事实支持；分布差异不等于因果。eval/use_cache=True已存在；实际runtime EOS与8夹具监督冒烟、24成对adapter/merged/定长延迟仍待GPU，不增加正式训练。

块Dense初始81708已actualexit1，完整批次4352向量与额外缓存全部保留；可用2974576640B跌破3GiB，属于守卫停批，不声明GPU显存OOM或确定内存泄漏。CPU tokenizer审计在该时段重叠，但不据时间相关认定因果。所有审计进程已终态后31519同身份单次resume，只补缺失缓存/批次；后续不给长GPU编码并行加载CPU模型或tokenizer。若同样阻断再出现，最多一个资源兼容替代，再记录BLOCKED，不无限重启。

第二次块编码31519 actualexit1，可用2822684672B低于3GiB，完整10816块/169批次保留，额外成功缓存亦保留。此次未并行CPU tokenizer，不能只归因于审计。最后且唯一资源兼容：resource_v3固定每1024向量新子进程/私有后端，数学模型/输入/4K不截断参数不变，已成功缓存直接校验复用；原两个日志/旧批次journal不覆盖。任一分段失败终态BLOCKED，不再retry。新S3编码仍单GPU串行，目标两条件/19673块各一遍，未扩实验组合。

全库Dense兼容59311 actualexit1/segment11，有界尝试已尽；V3完整189批次/12096向量保留，原10816及额外缓存不删。第三异常只service.parent_exception RuntimeError，Windows子进程输出未进重定向日志，具体原因UNKNOWN，不能直接写RAM leak。所有服务EXITED；WSL --list --running明确没有运行分发，没启动/重启WSL。当前前台/浏览器/系统安全进程内存占用只读核验，未停止。

独立PAPER_GIVEN小足迹替代完成141开发论文、真实scope向量；这不等于全库Dense完成。仅加载用户scope对应文件，20论文固定子进程，复用可核验的已有缓存；保持同一GPU向量空间，生成输入不读gold。440两条件上下文冻结，实际最大prompt2318+输出512<4096，无预检失败。B0名称明确frozen v2 policy with GPU embedding bridge，不能声称原CPU部署成本复现。

S3首40题真实80生成+3预热全部成功；独立word重算零差，32可答：neighbor .389403→core-first .453152（delta+.06375，论文bootstrap95% [.000116,.151739]，仅开发诊断）。完整参考覆盖.6875→.75、连续.6875→.7734375，两条件各1可答拒答。frozen-context P50不是在线E2E；关键事实/支持/引用语义仍未测，promotion阻断。明确漏斗与word诊断收益支持扩100开发确认，不允许直接发布。

D_confirm25952在77正式+3预热后RAM2491547648B守卫actualexit1，原成功全部保留。唯一修复：stream/fresh-process20固定输入，不改数学协议，45993续跑未执行123；旧记录去除wrapper record_hash/config_hash/recorded_at后保留原值作provenance，避免重现嵌套旧hash缺陷。每子进程日志直接捕获，不丢stderr。

S4可见输入三维规则标签220题已冻结，selected60；规则只识别compare5/explain49/fact161/other5，未识别summary，置信概率/材料充分性均null。没有人工确认、不伪造少数类、不按路径结果定义任务；60工作流真实效用仍待执行，禁止提前训练router。

2026-10-15：S3 D_confirm 独立100题审计完成。core-first文字F1增量约+0.01235、论文聚类区间跨零；主要claim/support/citation指标仍未知，不能发布。BGE真实12组夹具11组正确，max/top2同GPU权重，仅工程校准；句柄已失效且进程不存在，历史OS exit code保持UNKNOWN。S5固定D_confirm排序前24题，三条件同BF16/eager/greedy512、prompt-only，不依据输出选题。旧adapter只作诊断。EOS沿用实际generation_config，逐条保存输出IDs；8条监督烟雾采用官方配对答案及train:引用命名空间核对，最多1 optimizer step，不是正式重训。

2026-10-15 S5：同24个D_confirm基础/未合并各24+1预热均actualexit0。两者im_end后均仅2token，正常EOS；基础8条引用失败全保留。输出P50 67→171，冻结生成P50 5.749→19.880秒。固定64-token微基准各3正式+1预热实际完成，未合并约8.5–8.6tokens/s低于基础约12.5；同步回调开销同协议，不能当在线E2E。合并数值门槛失败（max_abs0.50–0.625/relativeL2约0.033–0.050），原门槛不放宽，未写合并快照。首次监督烟雾1步因warmup初始LR0无更新；新_v2沿用8夹具两次形成2 optimizer steps、228张量更新、均值loss−0.061489、完整恢复核验PASS，不是泛化成绩。原失败checkpoint/journal保留。任务书第11.4指定Markdown/JSON报告并禁上传，因此报告技能仅用于证据与结构，不生成MCP/HTML/Sites旁支。

2026-10-15 S4终端：session27341 actualexit0，180正式+9独立预热，全0格式失败。独立效用审计actualexit0，60题×4效用（W_clarify对指定scope复用W_local真实响应，0额外生成）。52可答；W_local文字F1 .363394、W_compare .340127、W_summary .360794。预标映射相对最佳固定混合短QA代理效用−.012080；hindsight .419707仅不可部署上界。没有总结标签、比较仅5题、关键点proxy未知，停止新router训练（NOT_RUN_GATE），不把生成时间当onlineE2E。仍须自然重排、任务关键点、S6锁定终验及UI/报告，不因该审计PASS声称整个Goal完成。

2026-10-15 自然重排与关键点诊断：BGE首40题727窗口对、Qwen463完整对、BGE确认100题1823窗口对均actualexit0。D_search max打包完整覆盖+6.25pp；D_confirm max−14.13pp/top2−13.04pp，拒绝候选，不扩生成组合。60真实索引机制查找actualexit0：specified跳过文档层，ambiguous澄清，显式分支有全库BM25旁路；冻结query vector成本不算onlineE2E。6整篇教师请求只有3合法任务（2summary/1compare），原17失败分支与兼容9失败任务保留；两阶段兼容不新增标签。6正式任务答案+1预热，3组实际原子切分全通过。首次NLI在已加载后重复预留2GiB触发guard；小固定cohort的resident兼容保留加载前3GiB，已冻结关系输入额外64MiB及1GiB余量，不增加任务/改变阈值；实际132关系/61395encoder tokens，initialexit1与compatibilityexit0均保留，不重启旧全量评分。新的D_confirm200答案另实际完成100原子分组（10失败保留）及244NLI关系；92可答主claimF1只有B0 14/C1 13确定，均值null，不能选known子集，质量门槛不通过。预算2269/6000，最终1800保留。下一独立工作是online成本、CLI/UI和条件终验/验收；默认v2，人审NOT_REVIEWED。
