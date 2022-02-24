import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="data/train_dataset")
parser.add_argument("--model_dir", type=str, default="results_use_mecab")
parser.add_argument("-b","--train_batch_size", type=int, default=16)
parser.add_argument("--eval_batch_size", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=5e-5)
parser.add_argument("--adam_epsilon", type=float, default=1e-8)
parser.add_argument("--max_grad_norm", type=float, default=1.0)
parser.add_argument("-e","--num_train_epochs", type=int, default=30)
parser.add_argument("--warmup_ratio", type=int, default=0.1)
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument('--mecab',type=str, default="False")
parser.add_argument(
    "--model_name_or_path",
    type=str,
    default="klue/roberta-base",
)

# Model Specific Argument
parser.add_argument("--hidden_size", type=int, help="GRU의 hidden size", default=768)
parser.add_argument(
    "--vocab_size",
    type=int,
    help="vocab size, subword vocab tokenizer에 의해 특정된다",
    default=None,
)
parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
parser.add_argument("--proj_dim", type=int,
                    help="만약 지정되면 기존의 hidden_size는 embedding dimension으로 취급되고, proj_dim이 GRU의 hidden_size로 사용됨. hidden_size보다 작아야 함.", default=None)
parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5)
args = parser.parse_args()