# Research Learning Workflow

**AI 辅助的论文精读与研究学习工作流 · MIT 开源**

A source-grounded workflow for paper reading, teaching experiments, and evidence-based feedback.

从论文原文出发，将主题讲义、数学与代码、独立作答和反馈组织成可追溯的学习过程。本仓库公开现有项目 skill、材料处理工具和教学代码，供参考与改编。

[项目介绍](https://github.com/by-lastime/research-learning-workflow-showcase) · [MIT 许可证](LICENSE) · [论文双语阅读器](https://github.com/by-lastime/bilingual-paper-reader)

## 仓库内容

| 目录 | 内容 |
|---|---|
| `.agents/skills/yang-lab-research-tutor/` | 项目 skill、教学和来源规则、评估与推进方式 |
| `资料库/工具/` | 材料获取、正文整理及 OCR 工具 |
| `资料库/07_代码实践/` | Attention / DDPM 等局部教学代码、已有运行记录和环境说明 |
| `资料库/04_正式学习路线/` | 工作流、环境方案与讲图标准 |

## 如何使用

1. 阅读 `资料库/04_正式学习路线/10_MD单文件学习工作流.md`，了解协作方式。
2. 阅读 skill 主文件，再按自己的项目调整其 `references/project-map.md`。
3. 根据具体教学脚本的导入与环境说明准备 Python 依赖；PyTorch 相关示例需自行安装适合本机的版本。
4. 按自己的论文与问题选择小实验，核对运行结果及其适用范围。

这里保留了原项目的目录结构和 skill 名称，但不包含完整课程资料。项目地图中的论文、讲义、学习入口与作答文件需要自行接入；不要将缺少的个人状态编造成真实记录。材料抓取或 OCR 工具按需使用，不是运行所有示例的前置条件。

## 内容范围

原论文、网页全文、用户作答、诊断结果和个人学习状态没有纳入。已有运行记录属于当时环境下的局部教学验证，不能视为完整论文复现或新的性能结论；本轮未重新执行实验。

`SOURCE_SNAPSHOT.json` 标明快照来源文件及当前校验和，公开整理修改的文件另保留原始校验和。本机原项目未改变，未配置自动同步。

## 许可

仓库内原创代码与工作流文档使用 [MIT](LICENSE)。第三方依赖保留各自许可证；引用论文与外部材料不因本仓库开源而变更许可，使用者应自行准备有权使用的材料。
