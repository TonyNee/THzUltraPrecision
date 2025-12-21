### model.py:

#### （1）结构
单层20节点

#### （2）效果
数据集范围：10-40 GHz



#### （3）日志

```shell
[19:59:23 root@Aliyun-ECS ~/nitao/01coding/01python]$python3 bpnn_train.py 
使用设备: gpu
Epoch 100/10000, Training Loss: 0.000841
Epoch 200/10000, Training Loss: 0.000837
Epoch 300/10000, Training Loss: 0.000837
Epoch 400/10000, Training Loss: 0.000837
Epoch 500/10000, Training Loss: 0.000837
Epoch 600/10000, Training Loss: 0.000837
Epoch 700/10000, Training Loss: 0.000837
Epoch 800/10000, Training Loss: 0.000837
Epoch 900/10000, Training Loss: 0.000837
Epoch 1000/10000, Training Loss: 0.000837
Epoch 1100/10000, Training Loss: 0.000837
Epoch 1200/10000, Training Loss: 0.000837
Epoch 1300/10000, Training Loss: 0.000837
Epoch 1400/10000, Training Loss: 0.000837
Epoch 1500/10000, Training Loss: 0.000837
Epoch 1600/10000, Training Loss: 0.000837
Epoch 1700/10000, Training Loss: 0.000837
Epoch 1800/10000, Training Loss: 0.000837
Epoch 1900/10000, Training Loss: 0.000837
Epoch 2000/10000, Training Loss: 0.000837
Epoch 2100/10000, Training Loss: 0.000837
Epoch 2200/10000, Training Loss: 0.000837
Epoch 2300/10000, Training Loss: 0.000837
Epoch 2400/10000, Training Loss: 0.000837
Epoch 2500/10000, Training Loss: 0.000837
Epoch 2600/10000, Training Loss: 0.000837
Epoch 2700/10000, Training Loss: 0.000837
Epoch 2800/10000, Training Loss: 0.000837
Epoch 2900/10000, Training Loss: 0.000837
Epoch 3000/10000, Training Loss: 0.000837
Epoch 3100/10000, Training Loss: 0.000837
Epoch 3200/10000, Training Loss: 0.000837
Epoch 3300/10000, Training Loss: 0.000837
Epoch 3400/10000, Training Loss: 0.000837
Epoch 3500/10000, Training Loss: 0.000837
Epoch 3600/10000, Training Loss: 0.000837
Epoch 3700/10000, Training Loss: 0.000837
Epoch 3800/10000, Training Loss: 0.000837
Epoch 3900/10000, Training Loss: 0.000837
Epoch 4000/10000, Training Loss: 0.000837
Epoch 4100/10000, Training Loss: 0.000837
Epoch 4200/10000, Training Loss: 0.000837
Epoch 4300/10000, Training Loss: 0.000837
Epoch 4400/10000, Training Loss: 0.000837
Epoch 4500/10000, Training Loss: 0.000837
Epoch 4600/10000, Training Loss: 0.000837
Epoch 4700/10000, Training Loss: 0.000837
Epoch 4800/10000, Training Loss: 0.000837
Epoch 4900/10000, Training Loss: 0.000837
Epoch 5000/10000, Training Loss: 0.000837
Epoch 5100/10000, Training Loss: 0.000837
Epoch 5200/10000, Training Loss: 0.000837
Epoch 5300/10000, Training Loss: 0.000837
Epoch 5400/10000, Training Loss: 0.000837
Epoch 5500/10000, Training Loss: 0.000837
Epoch 5600/10000, Training Loss: 0.000837
Epoch 5700/10000, Training Loss: 0.000837
Epoch 5800/10000, Training Loss: 0.000837
Epoch 5900/10000, Training Loss: 0.000837
Epoch 6000/10000, Training Loss: 0.000837
Epoch 6100/10000, Training Loss: 0.000837
Epoch 6200/10000, Training Loss: 0.000837
Epoch 6300/10000, Training Loss: 0.000837
Epoch 6400/10000, Training Loss: 0.000837
Epoch 6500/10000, Training Loss: 0.000837
Epoch 6600/10000, Training Loss: 0.000837
Epoch 6700/10000, Training Loss: 0.000837
Epoch 6800/10000, Training Loss: 0.000837
Epoch 6900/10000, Training Loss: 0.000837
Epoch 7000/10000, Training Loss: 0.000837
Epoch 7100/10000, Training Loss: 0.000837
Epoch 7200/10000, Training Loss: 0.000837
Epoch 7300/10000, Training Loss: 0.000837
Epoch 7400/10000, Training Loss: 0.000837
Epoch 7500/10000, Training Loss: 0.000837
Epoch 7600/10000, Training Loss: 0.000837
Epoch 7700/10000, Training Loss: 0.000837
Epoch 7800/10000, Training Loss: 0.000837
Epoch 7900/10000, Training Loss: 0.000837
Epoch 8000/10000, Training Loss: 0.000837
Epoch 8100/10000, Training Loss: 0.000837
Epoch 8200/10000, Training Loss: 0.000837
Epoch 8300/10000, Training Loss: 0.000837
Epoch 8400/10000, Training Loss: 0.000837
Epoch 8500/10000, Training Loss: 0.000837
Epoch 8600/10000, Training Loss: 0.000837
Epoch 8700/10000, Training Loss: 0.000837
Epoch 8800/10000, Training Loss: 0.000837
Epoch 8900/10000, Training Loss: 0.000837
Epoch 9000/10000, Training Loss: 0.000837
Epoch 9100/10000, Training Loss: 0.000837
Epoch 9200/10000, Training Loss: 0.000837
Epoch 9300/10000, Training Loss: 0.000837
Epoch 9400/10000, Training Loss: 0.000837
Epoch 9500/10000, Training Loss: 0.000837
Epoch 9600/10000, Training Loss: 0.000837
Epoch 9700/10000, Training Loss: 0.000837
Epoch 9800/10000, Training Loss: 0.000837
Epoch 9900/10000, Training Loss: 0.000837
Epoch 10000/10000, Training Loss: 0.000837

===== 模型评价指标 =====
训练集 MSE: 0.000837
训练集 R²: 0.999979
测试集 MSE: 0.001161
测试集 R²: 0.999969
残差统计: MAE=38.007183074951 MHz, 标准差=46.517955780029 MHz
模型已保存至 ./model/bpnn_model_20250928.pth
[22:08:41 root@Aliyun-ECS ~/nitao/01coding/01python]$python3 bpnn_eval.py 
使用设备: gpu
成功加载模型: ./model/bpnn_model_20250928.pth

测试集 MAE: 0.031204
测试集 MSE: 0.001493
测试集 R²: 0.999971
预测结果已保存至 ./result/test_predictions.csv
残差最大值: 91.049194335938 MHz
残差最小值: 0.476837158203 MHz
残差图已保存至 ./result/residual_plot.png

```



