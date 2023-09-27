本模型通过命名实体识别（Named entity recognition， NER），实现从快递单中提取有效字段信息。

> 例如 `“张三18600009172广东省深圳市南山区八马路与东平行路交叉口北40米”`，
>
> 将识别出`('广东省', 'A1')('深圳市', 'A2')('南山区', 'A3')('八马路与东平行路交叉口北40米', 'A4')('张三', 'P')('18600009172', 'T')`的字段信息。



**序列标注（Sequence labeling）**

| 抽取实体/字段 | 符号 | 抽取结果     |
| :------------ | :--- | :----------- |
| 姓名          | P    | 张三         |
| 电话          | T    | 18625584663  |
| 省            | A1   | 广东省       |
| 市            | A2   | 深圳市       |
| 区            | A3   | 南山区       |
| 详细地址      | A4   | 百度国际大厦 |

根据BIO标注法，标签集合可以定义为

| 标签 | 定义                       |
| :--- | :------------------------- |
| P-B  | 姓名起始位置               |
| P-I  | 姓名中间位置或结束位置     |
| T-B  | 电话起始位置               |
| T-I  | 电话中间位置或结束位置     |
| A1-B | 省份起始位置               |
| A1-I | 省份中间位置或结束位置     |
| A2-B | 城市起始位置               |
| A2-I | 城市中间位置或结束位置     |
| A3-B | 县区起始位置               |
| A3-I | 县区中间位置或结束位置     |
| A4-B | 详细地址起始位置           |
| A4-I | 详细地址中间位置或结束位置 |
| O    | 无关字符                   |



**数据准备**

- 训练集train.txt
- 验证集dev.txt
- 测试集test.txt

每行数据由两列组成（训练集中除第一行是 `text_a\tlabel`），以制表符分隔，第一列是 utf-8 编码的中文文本，以 `\002` 分割，第二列是对应序列标注的结果，以 `\002` 分割。



读入数据，使用预训练模型将字符映射到token_id，构造数据加载器。

```mermaid
graph LR;
	读入数据 ---> Tokenizer ---> DataLoader
```



**训练模型**

双向门控循环单元（Gate Recurrent Unit，BIGRU）是一种经典的循环神经网络（RNN，Recurrent Neural Network），用于对句子等序列信息进行建模。

<img src="https://cdn.jsdelivr.net/gh/TaraTang2003/picgo-typora/202309281008788.png" style="zoom: 80%;" />

条件随机场（CRF，Conditional Random Fields)属于概率图模型中的无向图模型，可以用来解决序列标注任务中标签之间的依赖性这一问题。

<img src="https://cdn.jsdelivr.net/gh/TaraTang2003/picgo-typora/202309281010512.png" style="zoom:67%;" />



**训练成果**

```
Epoch: 0 ; Loss 1022.0720825195312
Epoch: 1 ; Loss 926.0482177734375
Epoch: 2 ; Loss 767.9647216796875
Epoch: 3 ; Loss 737.5621337890625
Epoch: 4 ; Loss 589.0906982421875
Epoch: 5 ; Loss 645.0921630859375
Epoch: 6 ; Loss 554.5932006835938
Epoch: 7 ; Loss 403.1741943359375
Epoch: 8 ; Loss 430.857421875
Epoch: 9 ; Loss 391.172119140625
```



**模型调参**



**模型评估**


