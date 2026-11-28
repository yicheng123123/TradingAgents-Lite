"""Run preparation, offline smoke, bounded Pilot prediction, and isolated scoring."""
import argparse

from .data import build_index, prepare
from .run import execute, freeze
from .smoke import offline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["prepare","index","offline","freeze","smoke","formal","score","report"])
    parser.add_argument("--device",choices=["cuda","cpu"],default="cuda")
    args=parser.parse_args()
    if args.stage=="prepare":
        prepare()
    elif args.stage=="index":
        build_index()
    elif args.stage=="offline":
        offline(device=args.device)
    elif args.stage=="freeze":
        freeze()
    elif args.stage in {"smoke","formal"}:
        execute(args.stage,device=args.device)
    else:
        from .metrics import export
        export(allow_settlement=args.stage=="score")


if __name__=="__main__":
    main()
