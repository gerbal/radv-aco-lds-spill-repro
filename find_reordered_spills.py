#!/usr/bin/env python3
# Copyright 2026 Grant McLendon
# SPDX-License-Identifier: MIT

"""Find LDS spills and reloads of one slot that ACO's scheduler reordered.

usage: find_reordered_spills.py SCHED_DUMP NOSCHED_DUMP

Both are RADV_DEBUG=shaders stderr from the same shader, one with
ACO_DEBUG=nosched. With the scheduler off, the IR "After RA" keeps the
spiller's order. The spill pass's SSA ids are the same in both dumps, so each
ds_write_addtid_b32 (by the value it stores) and ds_read_addtid_b32 (by the
value it defines) can be found in both; any write/read pair of one offset in
one block whose order differs between them is reported. Exit status 1 if any
is found, 2 if the shader doesn't spill to LDS.
"""
import re
import sys

# ACO prints no offset0: for offset 0, and (kill) before a value's last use.
READ = re.compile(r"%(\d+):v\[\d+\] = ds_read_addtid_b32 [^ ]+(?: offset0:(\d+))?")
WRITE = re.compile(r"ds_write_addtid_b32 [^ ]+, (?:\(kill\))?%(\d+):v\[\d+\](?: offset0:(\d+))?")
BLOCK = re.compile(r"^BB(\d+)$")


def after_ra(path):
    """[(block, line_no, kind, ssa_id, offset, text)] in the After RA IR."""
    out, block, on = [], None, False
    for n, line in enumerate(open(path, errors="replace"), 1):
        s = line.strip()
        if s == "After RA:":
            on = True
            continue
        if not on:
            continue
        if s == "disasm:":
            break
        if m := BLOCK.match(s):
            block = int(m.group(1))
        elif m := READ.search(s):
            out.append((block, n, "read", int(m.group(1)), int(m.group(2) or 0), s))
        elif m := WRITE.search(s):
            out.append((block, n, "write", int(m.group(1)), int(m.group(2) or 0), s))
        elif "addtid" in s:
            print(f"warning: {path}:{n}: unparsed: {s}", file=sys.stderr)
    return out


def main():
    sched, nosched = after_ra(sys.argv[1]), after_ra(sys.argv[2])
    if not nosched:
        print("no ds_*_addtid_b32 in the nosched dump: no LDS spilling")
        sys.exit(2)
    # An access is (kind, SSA id, offset): the spilled value for a write, the
    # reloaded one for a read. Per block and offset, the order of the writes
    # and reads must be the same with and without the scheduler.
    def order(dump):
        seqs = {}
        for b, n, kind, ssa, off, text in dump:
            seqs.setdefault((b, off), []).append(((kind, ssa), n, text))
        return seqs
    want, got = order(nosched), order(sched)
    unmatched = {(k, a) for k, seq in got.items() for a, _, _ in seq} ^ \
                {(k, a) for k, seq in want.items() for a, _, _ in seq}
    if unmatched:
        print(f"warning: {len(unmatched)} accesses are not in both dumps; they are not checked",
              file=sys.stderr)
    found = 0
    for key, seq in sorted(got.items(), key=lambda kv: kv[1][0][1]):
        rank = {acc: i for i, (acc, _, _) in enumerate(want.get(key, []))}
        for i, (a, na, ta) in enumerate(seq):
            for b, nb, tb in seq[i + 1:]:
                if a in rank and b in rank and rank[a] > rank[b] and a[0] != b[0]:
                    found += 1
                    what = "reload above its spill" if a[0] == "read" else "spill above the slot's previous reload"
                    print(f"BB{key[0]} offset {key[1]}: {what}")
                    print(f"  {sys.argv[1]}:{na}: {ta}")
                    print(f"  {sys.argv[1]}:{nb}: {tb}")
    reads = sum(1 for e in sched if e[2] == "read")
    print(f"{found} same-slot spill/reload pairs reordered by the scheduler ({reads} LDS reloads)")
    sys.exit(1 if found else 0)


main()
