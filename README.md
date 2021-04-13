# 嵌入式端的神经网络算法部署和实现

介绍关于 ARM NN、CMSIS NN 和 K210 等嵌入式端的 NN 部署和实现。神经网络的调教（训练）还是在 PC 端，神经网络参数训练好之后，在嵌入式端进行部署，使其接收网络给定数据、经过网络计算再给出结果，这么一个过程。

------

## MCU 端

### K210

[Github 仓库](https://github.com/Staok/Awesome-K210)

收集关于 K210 的 MaixPy 开发和 SDK IDE 开发等的软硬件入门资料，帮助初学者快速了解、学习和入门 K210。

这款芯片的生态已经起来了，相关的开发板、kendryte 官方和 sipeed 官方的资料和例程、各路网友大佬的例程、网络训练以及模型开源等等已经非常丰富。甚至[北航高校已经应用部署到无人机产品上](https://github.com/LZBUAV/K210_Python)了，描述如下。

>   该项目是Kendryte K210 AI芯片应用程序的集合，其中包括面部检测，颜色检测，目标检测和分类，QR码和Apriltag码检测以及与ArduPilot飞行软件的通信。 最后，我们可以将这些应用程序部署到无人机终端，使无人机更加智能。
>
>   所实现的应用主要分为两类，第一个是机器视觉应用，该类应用基于openmv机器视觉库；第二类是深度学习应用，该类主要基于Tensorflow和yolov2。详细功能参见用户手册。
>
>    本K210项目为Vision_MAV项目的配套项目，Vision_MAV项目旨在设计并实现一个依托深度学习和图像处理技术的基于视觉的微型无人机系统，能够实现在无GPS环境下的自主视觉导航、目标检测与追踪，该项目由北航无人系统研究院李大伟副教授课题组创立并进行研究，并将在项目没有保密需求后进行开源。本仓库的K210项目是Vision_MAV的一个配套项目，基于[嘉楠科技公司](https://canaan-creative.com/)生产的边缘AI芯片[K210](https://canaan-creative.com/product/kendryteai)，来实现目标检测与追踪，为Vision_MAV项目提供一个可选的视觉解决方案。该项目采用了一块[矽速科技公司](https://www.sipeed.com/)生产的MAXI DOCK K210评估板，来验证K210芯片的AI计算能力。在本项目中，采用传统机器视觉方法实现了最大色块识别、二维码识别、Apriltag码识别、圆形识别，采用深度学习方法实现了人脸识别、人体识别、口罩识别等，并开发了K210和[Ardupilot](https://github.com/ArduPilot/ardupilot)飞控固件之间的[MAVlink](https://github.com/ArduPilot/mavlink)通讯接口，来实时的将K210视觉模组解算出的目标位置信息发送给飞控来控制无人机运动。

### CMSIS-NN

CMSIS (Cortex Microcontroller Software Interface Standard) 是针对 Cortex-M MCU 和 Cortex-A5/A7/A9 处理器的一组软件应用框架，是标准化软件组件库，抽离出公共部分组件和启动文件等，简化开发，并提供很多高效的软件实现。下图示意其组成部件。

![CMSIS Version 5.3](assets/cmsis.png)

[CMSIS 的 Github 仓库](https://github.com/ARM-software/CMSIS_5)；[CMSIS 的使用文档](https://arm-software.github.io/CMSIS_5/General/html/index.html)。

其中这里最关心 CMSIS-NN。

#### 介绍

Collection of efficient neural network kernels，Neural networks and machine learning are functions that are being pushed into the end node if IoT applications.

The neural network kernels of the [**CMSIS-NN**](https://arm-software.github.io/CMSIS_5/NN/html/index.html) library help to maximize the performance and minimize the memory footprint of neural networks on Cortex-M processor cores. 

#### 提供的函数库

The library is divided into a number of functions each covering a specific category：

-   Convolution Functions
-   Activation Functions
-   Fully-connected Layer Functions
-   Pooling Functions
-   Softmax Functions
-   Basic math Functions

The functions can be classified into two segments：

-   Legacy functions supporting ARM's internal symmetric quantization(8 bits).
-   Functions that support TensorFlow Lite framework with symmetric quantization(8 bits).

The legacy functions can be identified with their suffix of _q7 or _q15 and are no new development is done there. The article in [2] describes in detail how to run a network using the legacy functions.

The functions supporting TensorFlow Lite framework is identified by the _s8 suffix and can be invoked from TFL micro. The functions are bit exact to TensorFlow Lite. Refer to the TensorFlow's documentation in [3] on how to run a TensorFlow Lite model using optimized CMSIS-NN kernels.

#### 源码、手册和例程

[CMSIS-NN 官方 Github 仓库，包含手册、例程等](https://github.com/ARM-software/CMSIS_5/tree/develop/CMSIS/NN)；

[官方教程集：CMSIS-NN 在 Arm Cortex-M 的应用](https://developer.arm.com/solutions/machine-learning-on-arm/developer-material/how-to-guides#cortex-m)；

[安富莱的 CMSIS-NN 开源教程和例程（暂时还没出）](http://www.armbbs.cn/forum.php?mod=viewthread&tid=94547)。

#### 总结

这里 CMSIS-NN 是结合 CMSIS DSP库，二者一块完成一些机器学习基本算子函数的库，可以在ARM Cortex M 的 MCU 开发中直接 include 然后调用 API 进行神经网络搭建；还可以使用 CMSIS-NN kernels 来运行 TensorFlow Lite 模型。

####  CMSIS-DSP 的介绍

CMSIS-DSP: Fast implementation of digital signal processing

Developing a real-time digital signal processing (DSP) system is not trivial as the DSP algorithms heavily rely on complex mathematical operations that are even time-critical.

[**CMSIS-DSP**](https://arm-software.github.io/CMSIS_5/DSP/html/index.html) library is a rich collection of DSP functions that are optimized by ARM for the various Cortex-M processor cores. CMSIS-DSP is widely used in the industry and enables also optimized C code generation from [MATLAB®](https://www.mathworks.com/hardware-support/arm-cortex-m-cmsis.html). The [ASN Filter Designer](https://www2.keil.com/mdk5/cmsis/asnfd) generates CMSIS-DSP code that you can directly use in your application.

关于 CMSIS-DSP 的学习和使用，"安富莱"进行了不少的摸索和验证，并出了几百页的教程，对于初学者足够够的了。

[安富莱的 硬汉嵌入式论坛](http://www.armbbs.cn/)；[安富莱官网](http://www.armfly.com/)；[安富莱的 CMSIS-DSP 开源教程和例程](http://www.armbbs.cn/forum.php?mod=viewthread&tid=94547)；[CMSIS-DSP 官方例子（Github）](https://github.com/ARM-software/CMSIS_5/tree/develop/CMSIS/DSP/Examples/ARM)。

### TinyML Projects

[Tiny Machine Learning 项目主页](https://hanlab.mit.edu/projects/tinyml/)

TinyML Projects 分为两个部分：

-   [MCUNet (inference, microcontrollers)](https://hanlab.mit.edu/projects/tinyml/mcunet/)
-   [TinyTL (on-device learning, memory-efficient transfer learning)](https://hanlab.mit.edu/projects/tinyml/tinyTL/)

论文、PPT、海报和源码等均已开源，商业权被大公司以超超高价买下。

MCUNet 实验结论：

<img src="assets/exp.png" alt="img" style="zoom:80%;" />

------

## ARM Cortex-A 端

### ARM NN

ARM NN 是 ARM 公司 在 Cortex-A 嵌入式端  "[AI and Machine Learning](https://developer.arm.com/solutions/machine-learning-on-arm)" 主题的 关键项目，官方介绍如下：

>   The machine learning platform libraries – Arm NN and Arm Compute Library – bridge the gap between existing neural network (NN) frameworks, such as TensorFlow, TensorFlow Lite, Caffe and ONNX, and the underlying IP.
>
>   They enable efficient translation of these NN frameworks, allowing them to run efficiently – without modification – across Arm Cortex-A CPUs, Arm Mali GPUs and the Arm ML processor.
>
>   ![ML Platform flow chart image](assets/ml-home-graphic-814-75fde4.webp)

[ARM NN 的 Github 仓库](https://github.com/arm-software/armnn)

>   The Arm NN SDK is a set of open-source software and tools that enables machine learning workloads on power-efficient devices. It provides a bridge between existing neural network frameworks and power-efficient Cortex-A CPUs, Arm Mali GPUs and Arm Ethos NPUs.
>
>   ![imageARMNN](assets/imageARMNN.png)

ARM NN 是 使用C++语言，可以直接载入如 tensorflow lite 框架生成的神经网络模型文件，然后对模型进行分析和优化，使之底层调用适合 ARM 内核的指令或者 NPU单元 实现运算加速，然后再使用 ARM NN 的 API 载入优化过的模型，进行推理计算，得到结果。

官方生态文章和实例：

-   [ARM NN 官方的应用文章](https://developer.arm.com/solutions/machine-learning-on-arm/developer-material/how-to-guides#armnn)
-   [ARM NN 官方的在树莓派上使用](https://developer.arm.com/solutions/machine-learning-on-arm/developer-material/how-to-guides#mlrpi)
-   [ARM 官网 - AI and Machine Learning 白皮书 各种应用介绍](https://developer.arm.com/solutions/machine-learning-on-arm/developer-material/white-papers-and-research-papers)

网友实例：

-   [被低估的ArmNN（一）如何编译](https://zhuanlan.zhihu.com/p/71369040)
-   [【树莓派/目标检测】(二)armnn编译](https://blog.csdn.net/qq_33446100/article/details/114024776)
-   看来目前还不是很多

### PyArmNN

[PyArmNN 的 Github 仓库](https://github.com/NXPmicro/pyarmnn-release)

PyArmNN 是 ARM NN 的 Python 实现，使用 Python 语言，干的活和 ARM NN 一样。

网友实例：

-   [Accelerating ML inference on X-Ray detection at edge using Raspberry Pi with PyArmNN](https://community.arm.com/developer/ip-products/processors/b/ml-ip-blog/posts/ml-inference-x-ray-detection-edge-raspberry-pi-pyarmnn)
-   看来目前还不是很多

## 总结

stm32 这种 ARM Cortex-M 的单片机应该用 CMSIS-NN 去复现（或者运行 TensorFlow Lite）神经网络模型；

而到了ARM Cortex-A 系列的运行 Linux 的平台，就使用 ARM NN 这个库，导入 tf 或者 Pytorch 的模型，进行优化加速。

所以都属于转化，主要还是算法设计。

## 其他平台

不太熟悉，只是罗列。

![其他嵌入式AI开发平台](assets/其他嵌入式AI开发平台.jpg)