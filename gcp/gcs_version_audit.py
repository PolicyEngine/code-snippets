#!/usr/bin/env python3
"""
GCS Version Audit Script (READ-ONLY)

Lists all object versions in a GCS bucket and shows what action would be
needed to revert each file to its state as of a target date.

THIS SCRIPT DOES NOT MODIFY ANYTHING. It only reads and reports.

Requirements:
    pip install google-cloud-storage

Usage:
    python gcs_version_audit.py my-bucket --target="2026-01-15T23:59:00"
    python gcs_version_audit.py my-bucket --target="2026-01-15T23:59:00" --tz-offset="-05:00"
    python gcs_version_audit.py my-bucket --target="2026-01-15T23:59:00" --limit=10
    python gcs_version_audit.py my-bucket --target="2026-01-15T23:59:00" --show-all-versions
"""

import argparse
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta

from google.cloud import storage


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit GCS object versions and plan a revert (read-only)."
    )
    parser.add_argument("bucket", help="GCS bucket name (without gs:// prefix)")
    parser.add_argument(
        "--prefix",
        default="",
        help="Only examine objects under this prefix (e.g. 'data/')",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Target datetime in ISO 8601 format (e.g. '2026-01-15T23:59:00')",
    )
    parser.add_argument(
        "--tz-offset",
        default="+00:00",
        help="UTC offset for target datetime, e.g. '-05:00' for EST (default: +00:00 UTC)",
    )
    parser.add_argument(
        "--show-all-versions",
        action="store_true",
        help="Print every version of every object (verbose)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after examining this many objects (0 = no limit, useful for testing)",
    )
    return parser.parse_args()


def build_target_dt(target_str, tz_offset_str):
    naive = datetime.fromisoformat(target_str)
    if naive.tzinfo is not None:
        return naive
    # Parse the offset manually
    sign = 1 if tz_offset_str[0] == "+" else -1
    parts = tz_offset_str[1:].split(":")
    hours, minutes = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    tz = timezone(timedelta(hours=sign * hours, minutes=sign * minutes))
    return naive.replace(tzinfo=tz)


def main():
    args = parse_args()
    target_dt = build_target_dt(args.target, args.tz_offset)

    print("=" * 70)
    print("GCS VERSION AUDIT (READ-ONLY)")
    print("=" * 70)
    print(f"  Bucket:      {args.bucket}")
    print(f"  Prefix:      {args.prefix or '(entire bucket)'}")
    print(f"  Target date: {target_dt.isoformat()}")
    print(f"  Limit:       {args.limit or 'none'}")
    print("=" * 70)
    print()

    client = storage.Client()
    bucket = client.bucket(args.bucket)

    # Check versioning status
    bucket.reload()
    if not bucket.versioning_enabled:
        print("WARNING: Object versioning is NOT currently enabled on this bucket.")
        print("         Noncurrent versions may still exist if versioning was enabled previously.")
        print()

    # Collect all versions grouped by object name
    print("Fetching object versions (this may take a while for large buckets)...")
    versions_by_name = defaultdict(list)
    total_versions = 0

    blobs = bucket.list_blobs(prefix=args.prefix or None, versions=True)
    limit_reached = False
    for blob in blobs:
        if limit_reached and blob.name not in versions_by_name:
            break

        versions_by_name[blob.name].append(
            {
                "generation": blob.generation,
                "updated": blob.updated,
                "size": blob.size,
                "is_live": not blob.time_deleted,  # live versions have no deletion time
                "metageneration": blob.metageneration,
            }
        )
        total_versions += 1

        if total_versions % 1000 == 0:
            print(f"  ...scanned {total_versions} versions so far...")

        if args.limit and len(versions_by_name) >= args.limit:
            limit_reached = True

    print(f"  Done. Found {total_versions} total versions across {len(versions_by_name)} objects.")
    print()

    # Classify each object
    actions = {"restore": [], "already_correct": [], "delete_new": [], "no_version": []}

    for name, vers in sorted(versions_by_name.items()):
        # Sort versions by updated time descending
        vers.sort(key=lambda v: v["updated"], reverse=True)

        # Find the live (current) version
        live = [v for v in vers if v["is_live"]]
        current = live[0] if live else None

        # Find the best version at or before the target date
        best_at_target = None
        for v in vers:
            if v["updated"] <= target_dt:
                best_at_target = v
                break  # first match is the most recent one <= target

        if args.show_all_versions:
            print(f"  {name}")
            for v in vers:
                marker = " <-- LIVE" if v["is_live"] else ""
                target_marker = " <-- TARGET" if v is best_at_target else ""
                print(
                    f"    gen={v['generation']}  updated={v['updated'].isoformat()}"
                    f"  size={v['size']}{marker}{target_marker}"
                )
            print()

        if best_at_target is None:
            # Object didn't exist at target date — it was created after
            actions["delete_new"].append(
                {
                    "name": name,
                    "current": current,
                    "earliest_version": vers[-1] if vers else None,
                }
            )
        elif current and current["generation"] == best_at_target["generation"]:
            # Current version IS the target version — no action needed
            actions["already_correct"].append({"name": name, "version": current})
        elif current:
            # Current version differs from target — would need to restore
            actions["restore"].append(
                {
                    "name": name,
                    "current": current,
                    "target_version": best_at_target,
                }
            )
        else:
            # Object was deleted after target date but existed at target — would need to restore
            actions["restore"].append(
                {
                    "name": name,
                    "current": None,
                    "target_version": best_at_target,
                }
            )

    # Print summary
    print("=" * 70)
    print("REVERT PLAN SUMMARY")
    print("=" * 70)
    restore_bytes = sum(item["target_version"]["size"] for item in actions["restore"])
    restore_gb = restore_bytes / (1024 ** 3)
    coldline_cost = restore_gb * 0.02

    print(f"  Already at correct version:  {len(actions['already_correct']):>6}")
    print(f"  Would need to restore:       {len(actions['restore']):>6}")
    print(f"  Created after target (new):  {len(actions['delete_new']):>6}")
    print(f"  Total objects:               {len(versions_by_name):>6}")
    print()
    print(f"  Total restore size:          {restore_gb:>9.2f} GB")
    print(f"  Est. Coldline retrieval cost: ${coldline_cost:>8.2f}  (at $0.02/GB)")
    print()

    if actions["restore"]:
        print("-" * 70)
        print("FILES THAT WOULD BE RESTORED TO AN OLDER VERSION:")
        print("-" * 70)
        for item in actions["restore"]:
            name = item["name"]
            tv = item["target_version"]
            cur = item["current"]
            if cur:
                print(f"  {name}")
                print(f"    Current:  gen={cur['generation']}  updated={cur['updated'].isoformat()}  size={cur['size']}")
                print(f"    Revert to: gen={tv['generation']}  updated={tv['updated'].isoformat()}  size={tv['size']}")
            else:
                print(f"  {name}  (DELETED — would be restored)")
                print(f"    Revert to: gen={tv['generation']}  updated={tv['updated'].isoformat()}  size={tv['size']}")
        print()

    if actions["delete_new"]:
        print("-" * 70)
        print("FILES CREATED AFTER TARGET DATE (did not exist at target time):")
        print("-" * 70)
        for item in actions["delete_new"]:
            name = item["name"]
            ev = item["earliest_version"]
            print(f"  {name}")
            if ev:
                print(f"    Earliest version: updated={ev['updated'].isoformat()}")
        print()

    if actions["already_correct"]:
        print("-" * 70)
        print(f"FILES ALREADY AT CORRECT VERSION: {len(actions['already_correct'])}")
        print("-" * 70)
        if len(actions["already_correct"]) <= 20:
            for item in actions["already_correct"]:
                print(f"  {item['name']}")
        else:
            for item in actions["already_correct"][:10]:
                print(f"  {item['name']}")
            print(f"  ... and {len(actions['already_correct']) - 10} more")
        print()

    print("=" * 70)
    print("NO CHANGES WERE MADE. This is a read-only audit.")
    print("=" * 70)


if __name__ == "__main__":
    main()
