# eval_all_t5.py
import os
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Evaluate all T5 checkpoints in outputs/t5/")
    parser.add_argument("--output_dir", type=str, default="outputs/t5")
    parser.add_argument("--checkpoints", nargs="+", type=str,
                        help="Explicit list of checkpoint names (e.g., checkpoint-10000 checkpoint-20000). "
                             "If not provided, auto-detects from output_dir.")
    args = parser.parse_args()

    if args.checkpoints:
        # Use explicit list
        checkpoints = [os.path.join(args.output_dir, ckpt) for ckpt in args.checkpoints]
    else:
        # Auto-detect: find all dirs starting with 'checkpoint-'
        if not os.path.exists(args.output_dir):
            print(f"❌ Output directory {args.output_dir} does not exist!")
            return
        all_items = os.listdir(args.output_dir)
        checkpoints = [
            os.path.join(args.output_dir, item)
            for item in all_items
            if item.startswith("checkpoint-") and os.path.isdir(os.path.join(args.output_dir, item))
        ]
        checkpoints.sort()  # Ensure epoch order

    if not checkpoints:
        print("❌ No checkpoints found!")
        return

    print(f"🔍 Found {len(checkpoints)} checkpoints. Starting evaluation...\n")

    for ckpt in checkpoints:
        print(f"\n{'='*60}")
        print(f" Evaluating: {ckpt}")
        print(f"{'='*60}")
        result = subprocess.run([
            "python", "evaluate.py",
            "--model_type", "t5",
            "--model_path", ckpt
        ])
        if result.returncode != 0:
            print(f"❌ Evaluation failed for {ckpt}")

    print("\n✅ All T5 checkpoint evaluations completed!")

if __name__ == "__main__":
    main()