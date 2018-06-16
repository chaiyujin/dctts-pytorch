import argparse
from enum import Enum
from pkg.preprocess import preprocess
from pkg.train import train


class Module(Enum):
    Text2Mel = 0
    SuperRes = 1

    def __str__(self):
        return self.name


class Action(Enum):
    preprocess = 0
    train = 1
    synthesis = 2

    def __str__(self):
        return self.name


def str_to_enum(the_enum, s):
    try:
        return the_enum[s]
    except Exception:
        raise ValueError()


def main():
    parser = argparse.ArgumentParser(description="optional actions")
    parser.add_argument("--action", type=lambda x: str_to_enum(Action, x), choices=list(Action))
    parser.add_argument("--module", type=lambda x: str_to_enum(Module, x), choices=list(Module))
    args = parser.parse_args()
    if args.action is None:
        parser.print_help()
        return
    if args.action == Action.preprocess:
        preprocess()
    elif args.action == Action.train:
        if args.module is None:
            parser.print_help()
        else:
            train(args.module)
    elif args.action == Action.synthesis:
        print(args.action)
        pass
    else:
        raise ValueError()


if __name__ == "__main__":
    main()
