import argparse
from pathlib import Path


def resolve_dataset_dir(dataset_dir: str, dataset_name: str) -> str:
    if dataset_dir:
        return dataset_dir
    package_root = Path(__file__).resolve().parent
    return str((package_root / "data" / dataset_name).resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="One-click QAER training and evaluation pipeline")
    parser.add_argument("--config", type=str, default="", help="Optional config JSON path")
    parser.add_argument("--dataset-dir", type=str, default="",
                        help="Dataset directory containing train/valid/test/stat")
    parser.add_argument("--dataset-name", type=str, default="", help="Dataset folder name under ./data")
    parser.add_argument("--output-dir", type=str, default="", help="Pipeline output directory")
    parser.add_argument("--skip-train", action="store_true", help="Skip training and only run evaluation")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Existing checkpoint for evaluation when --skip-train is used")
    parser.add_argument("--eval-split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--epochs", type=int, default=0, help="Override training epochs")
    parser.add_argument("--min-epochs", type=int, default=0, help="Override minimum training epochs before early stop")
    parser.add_argument("--early-stopping-patience", type=int, default=-1,
                        help="Override early stopping patience based on validation MRR")
    parser.add_argument("--early-stopping-min-delta", type=float, default=-1.0,
                        help="Override minimum validation MRR improvement for early stopping")
    parser.add_argument("--batch-size", type=int, default=0, help="Override train batch size")
    parser.add_argument("--eval-batch-size", type=int, default=0, help="Override eval batch size")
    parser.add_argument("--num-workers", type=int, default=-1, help="Override DataLoader worker count")
    parser.add_argument("--disable-warm-prior-cache", action="store_true", help="Skip up-front prior cache warmup")
    parser.add_argument("--prior-table-cache-size", type=int, default=-1,
                        help="Bound temporal prior table cache size; 0 means unbounded")
    parser.add_argument("--train-context-batch-limit", type=int, default=-1,
                        help="Override how many samples per train batch use graph/history neural context")
    parser.add_argument("--device", type=str, default="", help="Override device, e.g. cpu or cuda")
    args = parser.parse_args()

    from configs import load_config
    from eval import evaluate_checkpoint
    from train import train_model
    from utils import ensure_dir, save_json

    config = load_config(args.config)
    if args.dataset_dir or args.dataset_name or not args.config:
        config.data.dataset_dir = resolve_dataset_dir(args.dataset_dir, args.dataset_name or "ICEWS18")
    if args.output_dir:
        config.output_dir = args.output_dir
    elif not args.config:
        config.output_dir = str(
            (Path(__file__).resolve().parent / "outputs" / (args.dataset_name or "ICEWS18") / "QAER").resolve())
    if args.dataset_name and args.dataset_name != "ICEWS18" and config.semantic.cache_dir:
        config.semantic.cache_dir = str(Path(config.output_dir) / "semantic_cache")
    if args.epochs > 0:
        config.train.epochs = args.epochs
    if args.min_epochs > 0:
        config.train.min_epochs = args.min_epochs
    if args.early_stopping_patience >= 0:
        config.train.early_stopping_patience = args.early_stopping_patience
    if args.early_stopping_min_delta >= 0:
        config.train.early_stopping_min_delta = args.early_stopping_min_delta
    if args.batch_size > 0:
        config.train.batch_size = args.batch_size
    if args.eval_batch_size > 0:
        config.train.eval_batch_size = args.eval_batch_size
    if args.num_workers >= 0:
        config.train.num_workers = args.num_workers
        config.train.persistent_workers = args.num_workers > 0
    if args.disable_warm_prior_cache:
        config.train.warm_prior_cache = False
    if args.prior_table_cache_size >= 0:
        config.data.prior_table_cache_size = args.prior_table_cache_size
    if args.train_context_batch_limit >= 0:
        config.model.train_context_batch_limit = args.train_context_batch_limit
    if args.device:
        config.train.device = args.device

    output_dir = ensure_dir(config.output_dir)
    predictions_json = output_dir / f"{args.eval_split}_predictions_with_paths.json"
    evidence_csv = output_dir / f"{args.eval_split}_structured_evidence_chains.csv"

    print(
        {
            "method": config.method_name,
            "dataset_dir": config.data.dataset_dir,
            "output_dir": str(output_dir),
            "device": config.train.device,
            "epochs": config.train.epochs,
            "min_epochs": config.train.min_epochs,
            "early_stopping_patience": config.train.early_stopping_patience,
            "early_stopping_min_delta": config.train.early_stopping_min_delta,
            "batch_size": config.train.batch_size,
        },
        flush=True,
    )

    if args.skip_train:
        checkpoint = args.checkpoint or str(output_dir / "best_model.pt")
        print({"stage": "skip_train", "checkpoint": checkpoint}, flush=True)
    else:
        print({"stage": "train_stage_start"}, flush=True)
        train_result = train_model(config)
        checkpoint = train_result["best_checkpoint"] or str(output_dir / "best_model.pt")
        print({"stage": "train_stage_done", "checkpoint": checkpoint}, flush=True)

    print("QAER evaluation stage: evaluating checkpoint on", args.eval_split, flush=True)
    metrics = evaluate_checkpoint(
        checkpoint_path=checkpoint,
        dataset_dir=config.data.dataset_dir,
        split=args.eval_split,
        output_json=str(predictions_json),
        output_csv=str(evidence_csv),
    )
    metrics_payload = {
        "checkpoint": checkpoint,
        "eval_split": args.eval_split,
        "MRR": metrics["MRR"],
        "Hits@1": metrics["Hits@1"],
        "Hits@3": metrics["Hits@3"],
        "Hits@10": metrics["Hits@10"],
        "predictions_json": str(predictions_json),
        "evidence_csv": str(evidence_csv),
    }
    save_json(metrics_payload, str(output_dir / f"{args.eval_split}_metrics.json"))
    print(
        metrics_payload,
        flush=True,
    )


if __name__ == "__main__":
    main()
