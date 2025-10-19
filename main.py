import argparse
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train","play"], help="train = headless training, play = interactive")
    parser.add_argument("--episodes", type=int, default=5000)
    args = parser.parse_args()

    if args.mode == "train":
        import train as train
        train.run_training(args)
    else:
        import play as play
        play.run_play(args)