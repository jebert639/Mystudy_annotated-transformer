"""
句子补全 - 训练/推理入口脚本

超参数默认值遵循 Transformer 原论文 (Attention Is All You Need):
  d_model=512, N=6, h=8, d_ff=2048, warmup=4000, label_smooth=0.1

用法:
  训练:   python train_sentence_completion.py
  推理:   python train_sentence_completion.py --infer "I had always thought"
"""

import os
import sys
import argparse
from train_runner import train, infer


def main():
    parser = argparse.ArgumentParser(description="Transformer 句子补全训练")

    parser.add_argument("--infer", type=str, default=None, help="推理模式: 输入前缀文本进行补全")
    parser.add_argument("--model_path", type=str, default="output/best_model.pt", help="推理时加载的模型路径")
    parser.add_argument("--output_dir", type=str, default="output", help="输出文件夹路径")
    parser.add_argument("--data_path", type=str, default=None, help="训练数据路径 (默认: the-verdict.txt)")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有输出文件夹")
    parser.add_argument("--num_epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=128, help="批大小")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="验证集比例 (默认: 0.2)")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout 比例 (默认: 0.3)")
    parser.add_argument("--n", type=int, default=6, help="Transformer 层数 (默认: 6)")
    parser.add_argument("--d_model", type=int, default=512, help="模型维度 (默认: 512)")
    parser.add_argument("--d_ff", type=int, default=2048, help="前馈网络维度 (默认: 2048)")
    parser.add_argument("--h", type=int, default=8, help="多头注意力头数 (默认: 8)")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="权重衰减 (默认: 0.0)")

    args = parser.parse_args()

    if args.data_path is None:
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "the-verdict.txt")
    else:
        data_path = args.data_path

    if args.infer is not None:
        result = infer(prompt=args.infer, model_path=args.model_path)
        print(f"输入: {args.infer}")
        print(f"补全: {result}")
        sys.exit(0)

    result = train(
        data_path=data_path,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        dropout=args.dropout,
        n=args.n,
        d_model=args.d_model,
        d_ff=args.d_ff,
        h=args.h,
        weight_decay=args.weight_decay,
        overwrite=args.overwrite,
    )

    print(f"\n推理示例:")
    print(f'  python train_sentence_completion.py --infer "I had always thought" --model_path {result["output_dir"]}/best_model.pt')


if __name__ == "__main__":
    main()