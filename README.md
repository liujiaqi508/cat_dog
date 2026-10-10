# cat_dog 猫狗图像分类项目
基于PyTorch ResNet18实现猫狗二分类，使用Gradio搭建网页可视化预测界面。

## 环境依赖
torch==2.12.1
torchvision==0.27.1
opencv-python
gradio
numpy

## 项目文件说明
- train.py：模型训练脚本，训练完成可启动Gradio网页
- predict.py：加载训练好的模型，单独运行预测网页
- .gitignore：Git忽略配置文件，不上传数据集和模型权重

## 运行步骤
1. 准备数据集，划分train训练集、val验证集
2. 运行train.py训练模型，生成best_model.pth权重文件
3. 运行predict.py，打开网页上传图片识别猫狗

## 实验结果
训练Loss曲线、验证集准确率，网页端图片识别演示。
