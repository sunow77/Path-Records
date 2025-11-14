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

| 阶段             | 使用工具                   | 目的                                                         |
| ---------------- | -------------------------- | ------------------------------------------------------------ |
| ① 找模型         | Hugging Face Hub           | 搜索并下载别人已经预训练好的模型（例如 BERT、ViT、Whisper 等）。 |
| ② 适配与准备数据 | fastai                     | 构建 DataLoader、批量化预处理、可视化样本。                  |
| ③ 训练 / 微调    | fastai + HF Transformers   | 将 HF 模型封装进 fastai 的 `Learner`，利用 fastai 的学习率查找、回调系统、训练日志等。 |
| ④ 上传 / 部署    | Hugging Face Hub 或 Spaces | 上传 fine-tuned 模型，生成在线推理接口或网页 Demo。          |

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

输出为$p(X)$，$p(X)$代表“在自然界中，图像$X$的无噪声版本出现的概率”，也就是说，输入越清晰、越接近清晰版本图像，$p(X)$越接近1；

在对这个模型进行backward的时候，可以关注$ ∇_X⁡p(X)$,它代表在图像$X$这个状态下，$p(X)$的梯度；

也就是说，我们可以通过按照$ ∇_X⁡p(X)$梯度方向变动图像$X$，就可以使它越来越接近无噪声版本。

## 9.4 UNet~$f$~去噪

![](D:\Git\a\Path-Records\img\p2-09-4-1.png)

带有噪声的图=好图+噪声$N(0,σ)$

inputs: 带有噪声的图片

outputs：噪声

loss: $MSE()$

训练好后，好图=inputs-outputs

- 感知损失： MSE这种loss是像素级的，它造成后果是图像确实接近好图，但是会模糊、会失真，因此引入*perceptual loss*。perceptual loss是追求图片在感知层面上更接近真实图片，它不在MSE像素级比较，而是在高层特征空间比较，也就是比较某些层的特征图（比如纹理、边缘、形状等）

- UNet (网络结构的形状像字母U)：分为传统UNet和Stable diffusion的噪声预测器

  - 二者相同点：

    | 特性            | 传统 U-Net               | Stable Diffusion 的 U-Net      |
    | --------------- | ------------------------ | ------------------------------ |
    | Encoder         | 下采样特征，提取语义信息 | 下采样噪声特征，提取“潜在语义” |
    | Decoder         | 上采样特征，恢复空间信息 | 上采样去噪特征，重建潜空间分布 |
    | Skip Connection | 传递空间细节             | 同样保留层级特征，用于去噪还原 |

  - 二者不同点：

    | 对比维度    | 传统 U-Net（如图像分割）   | Stable Diffusion 的 U-Net                                    |
    | ----------- | -------------------------- | ------------------------------------------------------------ |
    | 任务目标    | 像素级预测（例如语义分割） | 对噪声图像逐步去噪（Diffusion过程）                          |
    | 输入        | 原始图像                   | 噪声图像 `x_t`（加噪后的潜空间图像） + 时间步 `t` + 文本嵌入 |
    | 输出        | 分割图、去噪图像等         | 预测当前噪声的分布（score 或 ε）                             |
    | Bottleneck  | 普通卷积层                 | 带 Attention 的 Transformer Block（Self + Cross Attention）  |
    | Feature类型 | 空间特征（像素）           | 潜空间特征（latent space）                                   |
    | 跳跃连接    | 拼接 encoder/decoder 特征  | 仍拼接，但通道更多、语义更复杂                               |
    | 时间信息    | 无                         | 有（每个 block 都加上 timestep embedding）                   |
    | 文本信息    | 无                         | 有（通过 cross-attention 融合 CLIP 文本特征）                |

## 9.5 VAE~图像解码

![](D:\Git\a\Path-Records\img\p2-09-5-1.png)

在这张图上，需要知道的是Encoder和Decoder交汇的地方那个特征层，就是Latent space潜空间，它通常是3D/4D张量，形状为$(batch~size, C, H, W)$，可以加速UNet。这在数学上不是必要的东西，但是我们太缺算力了，所以用它更加划算。

- **VAE**：是Autoencoder的一种，Autoencoder可以包括下面这些，我猜测Jeremy讲的是最简单的Autoencoder，VAE不是图上的。
  
  | 类型                            | 特点                          | 应用               |
  | ------------------------------- | ----------------------------- | ------------------ |
  | Vanilla Autoencoder             | 最基本的结构，MSE损失重建输入 | 图像压缩、异常检测 |
  | Denoising Autoencoder (DAE)     | 训练时添加噪声，增强鲁棒性    | 图像去噪、信号重建 |
  | Sparse Autoencoder              | 强制隐藏层稀疏激活            | 特征提取           |
  | Variational Autoencoder (VAE)   | 生成式模型，学习概率分布      | 图像生成、风格迁移 |
  | Convolutional Autoencoder (CAE) | 卷积结构                      | 图像重建、医学影像 |
  | Sequence Autoencoder            | RNN/Transformer结构           | 文本、时间序列     |
  
  目前Autoencoder应用的领域有很多，这个词已经不太用了，它的思想“从数据中学习潜在表示并重构原始输入”在不同的领域有了不同的表达：
  | 现代模型                | 与 Autoencoder 的关系                |
  | ----------------------- | ------------------------------------ |
  | VAE / Diffusion         | 直接继承编码-解码结构                |
  | BERT / GPT Encoder部分  | 自编码思想（重建输入或预测部分输入） |
  | AE-GNN / AE-Transformer | 图结构或序列的自编码                 |
  | SimCLR / BYOL / MAE     | 自监督学习中的“编码-重建”本质上是 AE |
  
- **Score function**：在概率论中指“在这个点上，数据分布上升最快的方向”，如果你能学会这个梯度，就能一步步“爬回”到真实数据分布的高概率区域。而在diffusion模型中，有两种模型：①噪声预测模型（Noise Prediction）：预测每步加的噪声$ \epsilon$；②score-based 模型（Score Function）：预测 $∇_{x_t}log⁡ p(x_t)$。他们其实都是噪声预测，只是表达形式不同。

- 因为VAE的Variational，它的潜空间z并不是一个固定向量，而是一个分布。VAE Encoder输出两个向量μ（均值） 和 σ（方差），即z ~ N(μ, σ²)，z = μ + σ * ε（其中 ε ~ N(0, 1)），于是每次采样到的z都略有不同，会给生成图像带来一些随机性。而且，这样在训练UNet的过程中，就可以保证输入的z是连续的、平滑的。不过需要说明的是，VAE的z给图像生成带来的随机性并不是颠覆性的，生成图像每次都不同的主要原因是初始生成的全噪声潜空间特征不同。

## 9.6 CLIP

在**9.4**中，我们说UNet的inputs是带有噪声的图片（经过了**9.5**，它变成了noisy latents），但在训练的时候，我们不只输入latents，我们还会直接告诉UNet这个latents是什么图，因此UNet的inputs就成为了$(noisy~latents, descriptions)$。

当descriptions特别简单的时候，我们可以用one-hot来编码，但当它特别复杂的时候，我们就很难这样编码。所以引入了一个CLIP（Contrastive Language–Image Pretraining）。

![](D:\Git\a\Path-Records\img\p2-09-6-1.png)

inputs: images、text

outputs: embeddings (一般是1D或2D张量，类似$(batch~size, feature~dim)$)

loss: contrastive loss (见上图红绿部分)

## 9.7 time step

在扩散模型中，我们模拟一个“逐步加噪声的过程”，这个过程被离散化成若干个时间步t：

- t=0：表示最干净的图像（原始潜空间图像 z₀）
- t=T：表示完全被噪声覆盖的图像（近似纯噪声）
- 中间的 t 值：表示“部分加噪”的状态

训练步骤如下：

1. 从原始潜空间图像 `z₀` 开始
2. 随机选择一个时间步 `t ∈ [1, T]`
3. 根据噪声调度函数（βₜ），加上噪声：$x_t=\sqrt{α_t}·z_0 + \sqrt{1 - α_t}·ε$ （其中 ε 是高斯噪声）
4. 把 `(x_t, t, text embedding)` 喂进 UNet
5. 让 UNet 预测噪声 `ε_θ(x_t, t)`，用 MSE 损失优化

在推理阶段，我们反过来用 time step 进行“逐步去噪”：

1. 从纯噪声 `x_T` 开始
2. 按照从大到小的时间步（T → 0）循环
3. 每一步调用 UNet：$ε_θ(x_t, t, c)$，预测当前噪声，并计算去噪后的 `x_{t-1}`
4. 最终得到干净的潜空间图像 `x_0`

> 所以 time step 也定义了 **采样步数**，决定生成速度与质量。
>  例如：
>
> - 1000步（完整扩散） → 高质量但慢
> - 50步（快速采样） → 速度快但略模糊
> - 也就是说模型要经过UNet去噪t次，使用采样器↓

| 采样器                  | 典型步数  | 特点                               |
| ----------------------- | --------- | ---------------------------------- |
| DDPM                    | 1000 步   | 最原始，精度高但慢                 |
| DDIM                    | 50–250 步 | 去噪更快，质量相近                 |
| DPM++ / Euler / Heun 等 | 10–50 步  | 最快，配合大模型仍能生成高质量图像 |

使用采样器的原因是，UNet预测的噪声是x_t与z_0之间的噪声，而每次经过UNet去一次噪声，我们是想去掉x_t与x_t-1的噪声

## 9.8 总结

UNet的inputs：latent space、text embedding、time step

![](D:\Git\a\Path-Records\img\p2-09-7-1.png)

### 9.8.1 训练流程

- 典型流程

1. CLIP 文本 encoder
   - 用图文对 `(图A, 文B)` 训练，得到文本 embedding `c`，让文本 embedding 对应图像语义。
   - 训练完成后，你有一个固定的文本 encoder，可以把任意文本映射到语义向量空间。
2. VAE 编码图像
   - 将真实图像 `A` 通过 VAE encoder → 潜空间 `z_0`
   - 对 `z_0` 加噪声 → 得到 `x_t`（噪声潜空间）
3. UNet 训练
   - 训练数据 `(x_t, c, t)` → 目标是预测噪声 `ε`，即学习 mapping，恢复出原始潜空间 `z_0`

> 核心点：训练中 x_t 与 c 是严格对应的

- 另一种流程

1. CLIP 文本 encoder
   - 用图文对 `(图A, 文B)` 训练，得到文本 embedding `c`，让文本 embedding 对应图像语义。
   - 训练完成后，你有一个固定的文本 encoder，可以把任意文本映射到语义向量空间。
2. VAE 编码图像
   - 将真实图像 `A_0` 通过 VAE encoder → 潜空间 `za_0`
   - 对 `za_0` 加噪声 → 得到 `xa_t`（噪声潜空间）
3. UNet 训练
   - 训练数据 `(xa_t, c, t)` → 目标是预测噪声 `ε`，即学习 mapping，恢复出原始潜空间 `za_0`

> 核心点：训练中 xa_t 与 c 不是严格对应的，此时就要求文B和图A_0的语义是匹配的

- 现在一般还是沿用典型流程，以下再重复一遍

1. 准备图文对 `(A, B)`
   - A：真实图像
   - B：对应的文本描述
2. VAE 编码图像
   - `A → z_0` （潜空间表示）
   - 对 `z_0` 添加噪声 → 得到 `x_t`（当前时间步的噪声潜空间）
3. 文本编码
   - `B → c`（CLIP 或类似文本 encoder 的 embedding）
4. UNet 输入
   - `(x_t, c, t)` → 目标预测噪声 `ε(x_t, c, t)`
   - 损失函数通常是 MSE：预测噪声 vs 实际加的噪声

- 扩展/增强

| 扩展方法                              | 用途                                              |
| ------------------------------------- | ------------------------------------------------- |
| 数据增强（旋转、裁剪、颜色 jitter）   | 增加训练多样性                                    |
| 语义增强（同义词、prompt paraphrase） | 文本 embedding 对应同一图像不同文本，提高模型泛化 |
| 潜空间扰动（轻微噪声、插值）          | 增加去噪鲁棒性，避免过拟合到固定 z_0              |
| 多样本条件（同一个文本对应多张图像）  | 提高生成多样性，但仍保证语义对齐                  |

# 10: Diving Deeper

## 10.1 Classifier-Free Guidance Diffusion

### 10.1.1 最初的“有分类器指导（Classifier Guidance）”

最早的做法来自 OpenAI 的 *Guided Diffusion*（2021）。 当时，他们训练了一个额外的分类器（classifier），在每个时间步告诉模型： “这个图像像不像提示词描述的那类图像？”模型会沿着分类器梯度的方向调整去噪方向，使图像更符合语义。
 数学上：
$$
ϵ_{guided}=ϵ_θ(x_t,t)−s⋅σ_t∇_{x_t}log⁡p_ϕ(y∣x_t)
$$
- $ϵ_θ(x_t,t)$：普通去噪预测
- $∇_{x_t}log⁡p_ϕ(y∣x_t)$：分类器输出的梯度
- $s$：控制指导强度的权重
这样做的问题是：还得额外训练一个分类器；分类器必须适配每个噪声级别；太复杂了。
如果直接用$ϵ_θ(x_t,t,c)$做结果，就会导致模型太过于遵从文本指令，最终的图片也特别不自然。
### 10.1.2 Classifier-Free Guidance（无分类器指导）
2021年底，论文 《Classifier-Free Diffusion Guidance》（Ho & Salimans, Google Research）提出： 既然可以让 UNet 自己学会“有条件”和“无条件”的生成， 那我们根本不需要外部分类器！
- **训练阶段**
让模型有时看到条件（比如文本 embedding），有时不看到条件（喂一个空 embedding），获得了$ϵ_θ(x_t,t,c) and ϵ_θ(x_t,t,∅)$
也就是说，它能学会两种情况：有 prompt 时的去噪；没 prompt（自由生成）时的去噪。
- **推理阶段**
我们可以混合两种预测结果：
$$
ϵ_{final}=ϵ_{uncond}+w⋅(ϵ_{cond}−ϵ_{uncond})
$$
其中：
  - $ϵ_{cond}=ϵ_θ(x_t,t,c)$：有条件预测（遵守提示词）；

  - $ϵ_{uncond}=ϵ_θ(x_t,t,∅)$：无条件预测（自由生成）；

  - $w$：Guidance Scale（通常设为 5～7）。当 $w>1$，图像会更贴合文本描述，但也可能变得不自然。

- **Stable Diffusion 中的实现**

  - CLIP 提供文本 embedding c

  - UNet 同时运行两次：

    1. 一次输入 $x_t,t,c=promp$

    2. 一次输入 $x_t,t,c=empty$

  - 然后混合两者结果，用上面的公式得到最终噪声预测。

## 10.2 Distillation 蒸馏

蒸馏的目标很多，这里只讨论步数蒸馏；步数蒸馏的方法也很多，这里只讲逐步蒸馏progressive distillation，这是目前扩散模型中最经典、最系统、也是最成功的一种蒸馏方式之一。

逐步蒸馏（Progressive Distillation）的核心思想是：让学生一次走多个 time steps，且是一层一层教学的。

- 采样

基于大量随机生成样本，直接从教师模型中采样出许多去噪轨迹；

随机选取中间的时间步对$ (x_t,x_{t−2})$；

- 训练

学生模型以 $x_t$为输入，以教师模型输出的“目标图” $x_{t−2}^{teacher}$为监督信号：
$$
\hat{y} =f_ϕ(x_t​,t,Δt=2)~~~其中没有text~embedding~c是因为默认对齐输入
$$
损失函数（L2损失-MSE均方误差）：
$$
L=∥\hat{y}​−y∥_2^2​
$$

- 逐步

| 模型       | 教师是谁 | 每步跳多少（相对原始时间步） | 采样序列举例    |
| ---------- | -------- | ---------------------------- | --------------- |
| A（原始）  | —        | 1                            | 200,199,198,…,0 |
| B（学生₁） | A        | 2                            | 200,198,196,…,0 |
| C（学生₂） | B        | 4                            | 200,196,192,…,0 |
| D（学生₃） | C        | 8                            | 200,192,184,…,0 |
| …          | …        | …                            | …               |

> 需要说明的是，不论ABCDE什么模型，它的结构都是UNet，参数量并没有任何减少，它的收益在于推理速度而非模型轻量化。

## 10.3 Scheduler 采样器

初期采样器是基于原始扩散过程的，后来越来越追求效率，于是出现了基于数值积分的采样器（当然我猜还有其它的）致力于缩短步数。以K-LMS为例理解所谓数值积分：
$$
x_{t+1} = x_t + Δt * (a*f_t + b*f_{t-1} + c*f_{t-2} + ...)
$$

| 采样器类型 | 主要目标         | 基于数值积分？ | 典型步数 |
| :--------- | :--------------- | :------------- | :------- |
| DDPM       | 忠实还原训练过程 | 否             | 1000     |
| DDIM       | 确定性生成、插值 | 否             | 20-100   |
| LMS        | 加速采样         | 是             | 20-50    |
| DPM-Solver | 极速采样         | 是             | 10-20    |
| Euler      | 简单稳定         | 是             | 30-100   |

采样器发展历程：从过程忠实 → 效率优先
       1. 初期：DDPM (1000步) - 忠实但慢
       2. 改进：DDIM (50-100步) - 可调整步数
       3. 加速：LMS/Euler (20-50步) - 数值方法
       4. 极速：DPM-Solver (10-20步) - 专用ODE求解器
       5. 最新：UniPC (10-25步) - 统一框架

## 10.4 Looking inside the pipeline

```python
from transformers import CLIPTextModel, CLIPTokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16) #分词并将词映射为数字
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).to("cuda")  #将数字映射为有距离意义的语义向量， 将模型或张量移动到 GPU 上进行计算

from diffusers import AutoencoderKL, UNet2DConditionModel
# Here we use a different VAE to the original release, which has been fine-tuned for more steps
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema", torch_dtype=torch.float16).to("cuda")
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", torch_dtype=torch.float16).to("cuda")

#实例化采样器
beta_start,beta_end = 0.00085,0.012
'''
# Noising schedule
plt.plot(torch.linspace(beta_start**0.5, beta_end**0.5, 1000) ** 2)
plt.xlabel('Timestep')
plt.ylabel('β');
'''
from diffusers import LMSDiscreteScheduler
scheduler = LMSDiscreteScheduler(beta_start=beta_start, beta_end=beta_end, beta_schedule="scaled_linear", num_train_timesteps=1000) #采样器

#参数
prompt = ["a photograph of an astronaut riding a horse"]
height = 512
width = 512
num_inference_steps = 70 #采样器设置时间步
guidance_scale = 7.5 #有没有prompt的lanterns差
batch_size = 1

#分词+嵌入
text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
'''
text_input['input_ids']
tensor([[49406,   320,  8853,   539,   550, 18376,  6765,   320,  4558, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407]])
text_input['attention_mask']
tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0]])
可以看到后面那些0都是我们不关心的东西，只有1才是我们关心的token
'''
text_embeddings = text_encoder(text_input.input_ids.to("cuda"))[0].half()
#其中text_encoder()通常是一个长度为2的tuple，[0]为text的embeddings，形状为[batch_size1, sequence_length77, hidden_size768]，[1]为对[0]进行了池化输出，形状为[batch_size1, hidden_size768]
#.half()是用半精度提高速度
max_length = text_input.input_ids.shape[-1] #77
uncond_input = tokenizer(
    [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
)
uncond_embeddings = text_encoder(uncond_input.input_ids.to("cuda"))[0].half()
uncond_embeddings.shape #torch.Size([1, 77, 768])
text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

#初始化latents
torch.manual_seed(100)
latents = torch.randn((batch_size, unet.in_channels, height // 8, width // 8)) #//8是因为VAE就是可以缩小计算量，共缩小64倍
latents = latents.to("cuda").half()
latents.shape #torch.Size([1, 4, 64, 64])

#初始化scheduler
scheduler.set_timesteps(num_inference_steps)

#缩放latents
latents = latents * scheduler.init_noise_sigma #scheduler.init_noise_sigma是一个数字，latents经过缩放从一个N(0,1)的正太分布变成了N(0,scheduler.init_noise_sigma^2)的正态分布，也就是x_T，这个大小就是scheduler熟悉的大小了

'''
scheduler.timesteps
tensor([999.0000, 984.5217, 970.0435, 955.5652, 941.0870, 926.6087, 912.1304,
        897.6522, 883.1739, 868.6957, 854.2174, 839.7391, 825.2609, 810.7826,
        796.3043, 781.8261, 767.3478, 752.8696, 738.3913, 723.9130, 709.4348,
        694.9565, 680.4783, 666.0000, 651.5217, 637.0435, 622.5652, 608.0870,
        593.6087, 579.1304, 564.6522, 550.1739, 535.6957, 521.2174, 506.7391,
        492.2609, 477.7826, 463.3043, 448.8261, 434.3478, 419.8696, 405.3913,
        390.9130, 376.4348, 361.9565, 347.4783, 333.0000, 318.5217, 304.0435,
        289.5652, 275.0870, 260.6087, 246.1304, 231.6522, 217.1739, 202.6957,
        188.2174, 173.7391, 159.2609, 144.7826, 130.3043, 115.8261, 101.3478,
         86.8696,  72.3913,  57.9130,  43.4348,  28.9565,  14.4783,   0.0000],
       dtype=torch.float64)
这里可以看到通过这个scheduler，我们虽然1000步加噪，但在去噪可以从999直接跳到985步
scheduler.sigmas
tensor([14.6146, 13.3974, 12.3033, 11.3184, 10.4301,  9.6279,  8.9020,  8.2443,
         7.6472,  7.1044,  6.6102,  6.1594,  5.7477,  5.3709,  5.0258,  4.7090,
         4.4178,  4.1497,  3.9026,  3.6744,  3.4634,  3.2680,  3.0867,  2.9183,
         2.7616,  2.6157,  2.4794,  2.3521,  2.2330,  2.1213,  2.0165,  1.9180,
         1.8252,  1.7378,  1.6552,  1.5771,  1.5031,  1.4330,  1.3664,  1.3030,
         1.2427,  1.1852,  1.1302,  1.0776,  1.0272,  0.9788,  0.9324,  0.8876,
         0.8445,  0.8029,  0.7626,  0.7236,  0.6858,  0.6490,  0.6131,  0.5781,
         0.5438,  0.5102,  0.4770,  0.4443,  0.4118,  0.3795,  0.3470,  0.3141,
         0.2805,  0.2455,  0.2084,  0.1672,  0.1174,  0.0292,  0.0000])
这是上面的每个iteration迭代的噪声数量，也就是上面的时间步下对应的加噪总数量
'''

#
from tqdm.auto import tqdm #tqdm是一个进度条，用了它就会在训练时生成进度条，不用它也不影响代码正确性
for i, t in enumerate(tqdm(scheduler.timesteps)): #其中i是0~69，t是scheduler.timesteps中的数
    input = torch.cat([latents] * 2) #因为有prompt和无prompt两个版本
    input = scheduler.scale_model_input(input, t)  #将input缩放到t对应的输入大小上
    # predict the noise residual
    with torch.no_grad(): pred = unet(input, t, encoder_hidden_states=text_embeddings).sample
    # perform guidance
    pred_uncond, pred_text = pred.chunk(2)
    pred = pred_uncond + guidance_scale * (pred_text - pred_uncond)
    # compute the "previous" noisy sample
    latents = scheduler.step(pred, t, latents).prev_sample
```

