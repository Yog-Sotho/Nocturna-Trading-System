"""
Score a Cosmic Ray session, list the surviving mutants, and fail when too many
of the tested ones survive.

cr-rate counts the mutants the operator filter skipped as killed, which
inflates the score; this counts only the mutants whose tests ran. A mutant
that timed out or broke the test run counts as killed.

    cosmic-ray dump mutation.sqlite | python tools/mutation/gate.py --max-survival 10
"""

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--max-survival", type=float, required=True, help="Highest allowed survival rate, in percent")
    args = parser.parse_args(argv)

    tested = pending = 0
    survivors = []
    for line in sys.stdin:
        if not line.strip():
            continue
        item, result = json.loads(line)
        if result is None:
            pending += 1
            continue
        if result["worker_outcome"] == "skipped":
            continue
        tested += 1
        if result["test_outcome"] == "survived":
            for m in item["mutations"]:
                survivors.append(f"{m['module_path']}:{m['start_pos'][0]} {m['operator_name']} "
                                 f"in {m.get('definition_name') or '?'}")

    if pending:
        print(f"{pending} mutants were not run: the session is incomplete")
        return 1
    if not tested:
        print("No mutant was tested")
        return 1
    rate = 100 * len(survivors) / tested
    print(f"tested {tested}, killed {tested - len(survivors)}, survived {len(survivors)} ({rate:.1f}%)")
    for survivor in sorted(survivors):
        print(f"  survived: {survivor}")
    if rate > args.max_survival:
        print(f"Survival rate {rate:.1f}% is above {args.max_survival}%")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
