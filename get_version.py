#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).parent
DEFAULT_OUTPUT_FILE = "version.txt"
DEFAULT_STABLE_BRANCH = "main"
GIT_DESCRIBE_MATCH = 'gmic-[0-9]*.[0-9]*.[0-9]*'
GIT_TAG_PARSE = r'gmic-(?P<version>\d+.\d+.\d+)'
REG_HASH = re.compile('[0-9a-f]+')
REG_HASHLIST = re.compile('([0-9a-f]+\n)*[0-9a-f]+')
VERBOSE = False

parser = argparse.ArgumentParser(description="Calculate the project version",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-u', '--update', action='store_true',
                    help=f"Writes the result in {DEFAULT_OUTPUT_FILE}, instead of only printing the calculated version")
parser.add_argument('-v', '--verbose', action='store_true', help="Enable debug messages")
parser.add_argument('-r', '--ref', default='HEAD', help="Calculate the version at the given ref")
parser.add_argument('-s', '--stable', nargs="?", default=DEFAULT_STABLE_BRANCH,
                    help="Consider the given ref, or --branch/HEAD without argument, as the stable branch")
parser.add_argument('-n', '--next-stable', action='store_true',
                    help="Calculates the next_stable version, i.e if you merged ref to stable")


def debug(msg: str):
    if VERBOSE:
        print("[DEBUG] " + msg, file=sys.stderr)


def run_git(command: list[str], *, error: bool | str = True, expect: Optional[re.Pattern] = None) -> Optional[str]:
    cmd = "'git {} …'".format(command[0])
    debug("Running command $ {}".format("git " + " ".join(a.replace(' ', '\\ ') for a in command)))
    proc = subprocess.run(['git'] + command, text=True, capture_output=True)
    if proc.returncode != 0:
        if error:
            msg = error if type(error) is str else "{cmd} returned an error"
            if proc.stderr:
                msg += ': ' + proc.stderr
            raise RuntimeError(msg)
        else:
            return None
    else:
        out = proc.stdout.strip()
        if expect is not None and expect.fullmatch(out) is None:
            msg = f"Unexpected output from '{cmd}': "
            if out == '':
                raise RuntimeError(msg + "empty")
            else:
                raise RuntimeError(msg + f"invalid '{out}'")
        return out


if __name__ == '__main__':
    os.chdir(SCRIPT_DIR)
    args = parser.parse_args()
    VERBOSE = args.verbose
    starting_ref = args.ref
    starting_hash = run_git(['rev-parse', '--short', args.ref], error="Couldn't resolve revision {}".format(args.ref),
                            expect=REG_HASH)
    debug(f"Using starting point '{starting_ref}' i.e {starting_hash}")
    stable_ref = starting_ref if args.stable is None else args.stable
    stable_hash = run_git(['rev-parse', '--short', stable_ref], error="Couldn't resolve revision {}".format(stable_ref),
                          expect=REG_HASH) if stable_ref != starting_ref else starting_hash
    debug(f"Using stable ref '{stable_ref}' i.e {stable_hash}")

    tag = run_git(['describe', '--tags', '--abbrev=0', '--match', GIT_DESCRIBE_MATCH, starting_hash],
                  error="Couldn't find a matching version tag")
    version = re.fullmatch(GIT_TAG_PARSE, tag)
    if version is None:
        raise RuntimeError(f"Couldn't parse git describe output '{tag}'")
    version = version.group("version")
    debug(f"Found tag {tag}, parsed version {version}")
    first_parents = run_git(['rev-list', '--first-parent', stable_hash], expect=REG_HASHLIST).split('\n')
    is_stable = starting_hash == stable_hash or any(p.startswith(starting_hash) for p in first_parents)

    if is_stable:
        dev_dist = 0
        stable_dist = int(run_git(["rev-list", "--first-parent", "--count", starting_hash, '--not', tag]))
        if not args.next_stable:
            stable_dist -= 1
        debug(f"Ref is stable, counted {stable_dist} merge commits since tag")
    elif run_git(["merge-base", "--is-ancestor", tag, stable_hash], error=False) is not None:
        if run_git(["merge-base", "--is-ancestor", starting_hash, stable_hash], error=False) is not None:
            first_desc = run_git(
                ['rev-list', '--topo-order', '--merges', '--ancestry-path', f'{starting_hash}..{stable_hash}'],
                expect=REG_HASHLIST).split('\n')[-1]
            debug(f"Found earliest descendant {first_desc}")
            stable_hash = run_git(['rev-parse', '--short', first_desc + '^'],
                                  error="Couldn't resolve parent of {}".format(first_desc),
                                  expect=REG_HASH)
            stable_ref = run_git(
                ["name-rev", "--no-undefined", "--always", "--name-only", "--refs=heads/*", stable_hash])
            debug(f"Taking first parent of common as stable ref: {stable_ref}")
            assert run_git(["merge-base", "--is-ancestor", starting_hash, stable_hash], error=False) is None
        dev_dist = int(run_git(["rev-list", "--count", starting_hash, '--not', stable_hash]))
        stable_dist = int(run_git(["rev-list", "--count", stable_hash, '--not', starting_hash]))
        debug(f"Counted {stable_dist} merge commits on stable since tag and {dev_dist} commits since last merge")
    else:
        stable_dist = 0
        dev_dist = int(run_git(["rev-list", "--count", starting_hash, '--not', tag]))
        debug(f"Tag is not an ancestor of stable, counting {dev_dist} commits since last merge")

    if stable_dist > 0:
        version += f".r{stable_dist}"
    if not is_stable and not args.next_stable:
        version += f".dev{dev_dist}"
    print(version)

    if args.update:
        with open(SCRIPT_DIR / DEFAULT_OUTPUT_FILE, 'w') as f:
            f.write(version)
