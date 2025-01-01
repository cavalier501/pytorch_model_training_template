模板还没测试过
下次测试通过后回来修改、记录

目录：
datasets(dataloader)
model
loss
configuration(args)
train
test
utils 其他辅助工具，如结果保存mesh、dict_to_cuda等
    HPWC logger文件中的一些logging方法也放在了utils中

20230330:
修改画图功能：
如果一开始的epoch很大，会影响后面小loss 变化趋势的观察
修改：每个epoch训练结束后，遍历之前的所有epoch，
如果之前epoch的loss大于当前最小loss的100倍，将该epoch对应的loss置零

20230721
修改存储checkpoint过程：
实际情况下每次更关心训练的best 参数和最后一次训练的参数
保存整个过程的参数还需要后续删checkpoint
增加了功能 保存新的最优checkpoint时删除上一次的checkpoint