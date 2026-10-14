"""Validate the current-boot GPU receipt without models, devices or mutations."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ragscope.experiments.runtime import validate_gpu_lease


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lease', required=True)
    parser.add_argument('--print-device', action='store_true')
    args = parser.parse_args()
    try:
        lease = validate_gpu_lease(args.lease)
    except (OSError, ValueError) as exc:
        print(f'GPU lease rejected: {exc}', file=sys.stderr)
        return 2
    print(str(lease['gpu_device']) if args.print_device else 'VALID_CURRENT_BOOT')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
