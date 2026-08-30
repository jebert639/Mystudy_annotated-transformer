"""
demo_train.py - 最小训练脚本示例
================================

以前写训练脚本需要理解 train_runner 的全部流程;
现在只需要一行 (参数想写几个写几个, 没写的用默认值):

    from transformer_api import train
    train(参数名=值, 参数名=值, ...)

直接运行本文件会做一次 2 epoch 的冒烟测试 (tiny 预设, 几分钟内跑完),
验证 环境/数据/建模型/训练/存档/绘图 整条链路都是通的。
"""

from transformer_api import train


if __name__ == "__main__":
    # 冒烟测试: 最小模型跑 2 个 epoch, 确认整条链路可用
    result = train(preset="tiny")

    print("\n冒烟测试完成!")
    print("  输出目录:", result["output_dir"])
    print(f"  最佳验证 loss: {result['best_val_loss']:.4f}")
    print(f"  模型参数量: {result['total_params']:,}")
    print("\n接下来可以推理:")
    print(f'  python src/transformer_api.py infer "I had always thought" --model {result["output_dir"]}/best_model.pt')

    # ---------- 其他常见写法 (取消注释即用) ----------
    #
    # train(name="big512_v2", preset="fast")              # ★给模型起名 -> outputs/big512_v2
    # train(preset="fast")                                # 快速实验配置 (256维/4层/30epoch)
    # train(preset="fast", d_model=384, num_epochs=50)    # 预设基础上覆盖若干参数
    # train(d_model=512, n=6, h=8, d_ff=2048,
    #       warmup_steps=4000, num_epochs=200)            # 论文标准配置
    # train(name="exp_A", data_path="data/combined.txt", seed=0)  # 换语料/名字/种子
    # train(preset="tiny", overwrite=True)                # 覆盖已有输出目录
