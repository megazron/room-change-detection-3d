import argparse
import sys

from .frames import InputError


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m rayvote",
        description="Find moved / added / removed objects between recordings of the same room.")
    ap.add_argument("videos", nargs="+", help="before.mp4 after.mp4 (walkaround mode), or 2+ clips with --static")
    ap.add_argument("--out", default="out", help="output directory (default: out)")
    ap.add_argument("--static", action="store_true",
                    help="clips come from one fixed camera (tripod); compares consecutive clips by default")
    ap.add_argument("--pair", action="append", default=[],
                    help="static mode: extra clip pair to compare, 1-based, e.g. --pair 1,5")
    ap.add_argument("--frames", type=int, default=40, help="keyframes per video (default: 40)")
    ap.add_argument("--voxel", type=float, default=0.05, help="voxel size in metres (default: 0.05)")
    ap.add_argument("--device", choices=["cuda", "cpu"], help="force device (default: cuda if available)")
    ap.add_argument("--vocab", help="text file with one object label per line (replaces the default list)")
    ap.add_argument("--gt-poses", help="synthetic test only: use known camera poses from gt.json, skip SfM")
    a = ap.parse_args(argv)
    try:
        if a.static:
            if len(a.videos) < 2:
                raise InputError("static mode needs at least two clips")
            from .static import run_static
            pairs = [(i, i + 1) for i in range(len(a.videos) - 1)]
            for p in a.pair:
                i, j = (int(x) - 1 for x in p.split(","))
                if not (0 <= i < len(a.videos) and 0 <= j < len(a.videos)):
                    raise InputError(f"--pair {p} out of range")
                pairs.append((i, j))
            run_static(a.videos, a.out, pairs=pairs, device=a.device)
            return
        if len(a.videos) != 2:
            raise InputError("walkaround mode takes exactly two videos: before after")
        vocab = [l.strip() for l in open(a.vocab) if l.strip()] if a.vocab else None
        from .pipeline import run
        run(a.videos[0], a.videos[1], a.out, n_frames=a.frames, voxel=a.voxel, gt_poses=a.gt_poses,
            device=a.device, vocab=vocab)
    except InputError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
