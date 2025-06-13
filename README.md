# HX-ANiMe-KaO

基于 ResNet18 的 二次元表情识别 模型 (带训练 和 WebUI)

1. 下载 数据集
    - [素晴4人组 颜艺表情包](https://www.bilibili.com/video/BV1JMjUzFEUw/)
        - 百度网盘链接: https://pan.baidu.com/s/1SQcxc8f6cHliL14uYZ1Xxw?pwd=2233
        - 夸克网盘: https://pan.quark.cn/s/8548aff280b0 提取码: qM1n

然后放到:

```bash
./dataset/mae
    - 阿库娅 # 示例
    - 惠惠
```

中, 仅需要不抠图的版本

2. 运行 [训练代码](run.py)

3. 启动 [WebUI](web_ui.py)
