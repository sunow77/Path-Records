**<font size=8>fast.ai</font>**

# [Practical Deep Learning 2022 Part2](https://course.fast.ai/) & [Fastbook](https://github.com/fastai/fastbook)

# 9: Stable Diffusion

## 9.1 GPU平台

当然除了GPU外，还有TPU，但是一般用不上。

- TPU：大于10亿参数的transformer模型；使用TensorFlow或JAX框架
- GPU：使用PyTorch；参数在几亿参数之内

### 9.1.1 GPU类型

**Quadro 系列（专业工作站显卡）：**用于 CAD、3D 渲染、科学计算等专业领域。

**GeForce 系列（消费级显卡）：**用于游戏、内容创作和部分 AI 推理任务。

**Tesla 系列（数据中心显卡）：**用于 AI 训练、深度学习、科学计算等数据中心任务。

**NVIDIA DGX 系列（AI 超级计算机）：**用于大规模 AI 模型训练和推理，属于整机解决方案。

- 云端 GPU 平台的主流显卡：

| 用途 / 档位         | 典型显卡 | 架构                  | 显存     | 主要用途                                      |
| ------------------- | -------- | --------------------- | -------- | --------------------------------------------- |
| 入门 / 推理         | T4、L4   | Turing / Ada Lovelace | 16–24 GB | AI 推理、小模型训练、视频处理、轻量深度学习   |
| 中高端 / 小型训练   | A10、A30 | Ampere                | 24–40 GB | 中型 AI 模型训练、深度学习开发                |
| 高端 / 大规模训练   | A100     | Ampere                | 40–80 GB | 大型 AI 模型训练、HPC、科研计算               |
| 旗舰 / 超大模型训练 | H100     | Hopper                | 80 GB    | 超大规模 AI 模型训练（GPT 类、图像生成）、HPC |

不过这只是国外的供应商策略，国内有很多供应商会租赁消费级显卡跑模型，效果也不错，价格便宜很多

- Tesla 系列分类：

| GPU 型号 | 架构         | 发布时间 | 显存           | 功耗                      | 训练性能 | 推理性能 | 适用场景               |
| -------- | ------------ | -------- | -------------- | ------------------------- | -------- | -------- | ---------------------- |
| **A100** | Ampere       | 2020-05  | 40/80 GB HBM2e | 400W（PCIe）/500W（SXM）  | 高       | 高       | 大规模 AI 训练、HPC    |
| **H100** | Hopper       | 2022-09  | 80 GB HBM3     | 350W（PCIe）/700W（SXM5） | 极高     | 极高     | 超大规模 AI 训练、推理 |
| **H200** | Hopper       | 2024     | 141 GB HBM3e   | 未公开                    | 极高     | 极高     | 下一代 AI 训练、推理   |
| **T4**   | Turing       | 2018-09  | 16 GB GDDR6    | 70W                       | 中       | 高       | AI 推理、小规模训练    |
| **L4**   | Ada Lovelace | 2024     | 24 GB GDDR6    | 120W                      | 中       | 高       | AI 推理、视频处理      |
| **L40S** | Ada Lovelace | 2025     | 24 GB GDDR6    | 350W                      | 高       | 极高     | AI 推理、图形渲染      |

### 9.1.2 Google Colab

[Colab主页](https://colab.research.google.com/)

[Colab 付费服务价格](https://colab.research.google.com/signup?utm_source=resource_tab&utm_medium=link&utm_campaign=payg_learn_more)

[Colab使用教程（超级详细版）及Colab Pro/Pro+评测 - 知乎](https://zhuanlan.zhihu.com/p/527663163)

| 方案              | 月费（大致）                                                 | “Compute Units”额度*                                         | 可用 GPU 类型 / 资源                                         | 大致运行时间 / RAM /备注                                     |
| ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **免费版 (Free)** | $0                                                           | 无明确 “额度” 或非常少                                       | 通常为较旧 GPU（如 K80）或 T4 等；强 GPU 不保证。([Paperspace by DigitalOcean Blog](https://blog.paperspace.com/alternative-to-google-colab-pro/?utm_source=chatgpt.com)) | 每次运行最多约 **12 小时**（或者更短）([谷歌研究](https://research.google.com/colaboratory/faq.html?utm_source=chatgpt.com))；RAM 普遍 ~12 GB 左右；资源 “不保证” ([谷歌研究](https://research.google.com/colaboratory/faq.html?utm_source=chatgpt.com)) |
| **Pro 版**        | 约 $9.99/月（视地区）([Paperspace by DigitalOcean Blog](https://blog.paperspace.com/alternative-to-google-colab-pro/?utm_source=chatgpt.com)) | 大约 **100 Compute Units/月** ([Medium](https://medium.com/%40jprachir/reality-check-if-you-are-opting-for-google-colaboratory-colab-2c9d36d3c0bd?utm_source=chatgpt.com)) | GPU 类型更好，如 T4、P100 等较新设备可能性更高 ([Paperspace by DigitalOcean Blog](https://blog.paperspace.com/alternative-to-google-colab-pro/?utm_source=chatgpt.com))；RAM 增加（可能至 ~32 GB）([Paperspace by DigitalOcean Blog](https://blog.paperspace.com/alternative-to-google-colab-pro/?utm_source=chatgpt.com)) | 运行时间比免费版更长，但仍有断开、资源调度不保证的情况([RCpedia](https://rcpedia.stanford.edu/blog/2024/03/28/train-machine-learning-models-on-colab-gpu/?utm_source=chatgpt.com)) |
| **Pro+ 版**       | 约 $49.99/月（视地区）([Paperspace by DigitalOcean Blog](https://blog.paperspace.com/alternative-to-google-colab-pro/?utm_source=chatgpt.com)) | 大约 **500 Compute Units/月** ([Revolgy](https://www.revolgy.com/insights/blog/introducing-google-workspace-colab-pro-and-pro-subscriptions?utm_source=chatgpt.com)) | 更强的 GPU（如可能获得 V100、甚至 A100）机率更高 ([Medium](https://medium.com/%40jprachir/reality-check-if-you-are-opting-for-google-colaboratory-colab-2c9d36d3c0bd?utm_source=chatgpt.com))；RAM 上限更高，例如 ~52 GB 左右 ([Paperspace by DigitalOcean Blog](https://blog.paperspace.com/alternative-to-google-colab-pro/?utm_source=chatgpt.com)) | 支持“后台运行”功能（即关闭浏览器也可继续运行）在某些地区/版本中可用 ([Revolgy](https://www.revolgy.com/insights/blog/introducing-google-workspace-colab-pro-and-pro-subscriptions?utm_source=chatgpt.com)) |

### 9.1.3 vast.ai

[Vast.ai | Console](https://cloud.vast.ai/?_gl=1*13c6bi4*_gcl_au*MTk1NzIxODU4MS4xNzYxMjcwMjQ2*_ga*MTU3NTc4NzYzNS4xNzYxMjcwMjQ4*_ga_DG15WC8WXG*czE3NjEyNzAyNDckbzEkZzAkdDE3NjEyNzAyNDckajYwJGwwJGgxNTEzODc1NjQ1)

### 9.1.4 AutoDL

感觉国外的GPU平台性价比不高，上网还麻烦，于是找一些国内的平台

[AutoDL算力云 | 弹性、好用、省钱，GPU算力零售价格新标杆](https://www.autodl.com/home)

[AutoDL帮助文档](https://www.autodl.com/docs/video/)

其它：

[国内AutoDL等几款GPU租用平台使用体验如何？ - 知乎](https://www.zhihu.com/question/488792295/answer/1897605554848903488)

## 9.2 Hugging face & Fastai

- 两者应用

| 阶段             | 使用工具                     | 目的                                                         |
| ---------------- | ---------------------------- | ------------------------------------------------------------ |
| ① 找模型         | 🧠 Hugging Face Hub           | 搜索并下载别人已经预训练好的模型（例如 BERT、ViT、Whisper 等）。 |
| ② 适配与准备数据 | 📘 fastai                     | 构建 DataLoader、批量化预处理、可视化样本。                  |
| ③ 训练 / 微调    | 🧠 fastai + HF Transformers   | 将 HF 模型封装进 fastai 的 `Learner`，利用 fastai 的学习率查找、回调系统、训练日志等。 |
| ④ 上传 / 部署    | 🚀 Hugging Face Hub 或 Spaces | 上传 fine-tuned 模型，生成在线推理接口或网页 Demo。          |

- 两者比较

| 目标                       | 用哪个框架更方便                      | 原因                                  |
| -------------------------- | ------------------------------------- | ------------------------------------- |
| 改模型结构                 | ✅ Hugging Face (Transformers)         | 模型是 PyTorch 类，直接继承修改最自然 |
| 改训练过程 / 微调流程      | ✅ fastai                              | 封装更友好，调参效率高                |
| 做大规模训练 / 部署        | ✅ Hugging Face Trainer / Hub / Spaces | 更标准、更可移植                      |
| 做小规模实验 / 教学 / 研究 | ✅ fastai                              | 快速试错、可解释性好                  |

- 其它模型库

| 仓库名称               | 侧重点                        | 特点                                       |
| ---------------------- | ----------------------------- | ------------------------------------------ |
| Hugging Face Model Hub | NLP、计算机视觉、语音生成等   | 开源模型仓库，适用于多模态、NLP、视觉模型  |
| PyTorch Hub            | 计算机视觉（ResNet、VGG、等） | PyTorch 预训练模型，主要面向计算机视觉任务 |
| TensorFlow Hub         | 计算机视觉、NLP、音频、视频   | TensorFlow 预训练模型，适合 TF 用户        |
| Google Model Garden    | 研究模型（NLP、CV等）         | 包含 Google 研究团队的各类开源模型         |
| Open Model Zoo         | 计算机视觉、嵌入式设备        | Intel 优化模型，专注嵌入式平台的加速       |
| Kaggle Datasets        | 数据集、模型                  | 大量数据集和模型代码分享                   |
| Facebook AI Model Zoo  | NLP、计算机视觉、强化学习等   | Facebook 开源的 AI 研究和模型              |

## 9.3 Stable Diffusion的核心思想

![](D:\Git\a\Path-Records\img\p2-09-3-1.png)

$X$代表图像，它可能是包含噪声的，也可能是不包含噪声的；

输入给$f$；

输出为$p(X)$，$p(X)$代表“在自然界中，图像$X$的无噪声版本出现的概率”，也就是说，输入越清晰，$p(X)$越接近1；

在对这个模型进行backward的时候，可以关注$ ∇_X⁡p(X)$,它代表在图像$X$这个状态下，$p(X)$的梯度；

也就是说，我们可以通过按照$ ∇_X⁡p(X)$梯度方向变动图像$X$，就可以使它越来越接近无噪声版本。

## 9.4 U-Net，$f$

