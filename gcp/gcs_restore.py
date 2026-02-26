#!/usr/bin/env python3
"""
GCS Restore Script

Restores all objects in a GCS bucket to their state as of a target date.
By default runs in DRY-RUN mode. Pass --execute to actually perform the restore.

Requires object versioning to be enabled on the bucket.

Requirements:
    pip install google-cloud-storage

Usage:
    # Dry run (no changes)
    python gcs_restore.py my-bucket --target="2026-01-15T23:59:00"

    # Execute restore
    python gcs_restore.py my-bucket --target="2026-01-15T23:59:00" --execute

    # Also delete files created after the target date
    python gcs_restore.py my-bucket --target="2026-01-15T23:59:00" --execute --delete-new
"""

import argparse
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta

from google.cloud import storage


def parse_args():
    parser = argparse.ArgumentParser(
        description="Restore GCS objects to their state at a target date."
    )
    parser.add_argument("bucket", help="GCS bucket name (without gs:// prefix)")
    parser.add_argument(
        "--prefix",
        default="",
        help="Only restore objects under this prefix",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Target datetime in ISO 8601 format (e.g. '2026-01-15T23:59:00')",
    )
    parser.add_argument(
        "--tz-offset",
        default="+00:00",
        help="UTC offset for target datetime (default: +00:00 UTC)",
    )
    parser.add_argument(
        "--delete-new",
        action="store_true",
        help="Delete objects that were created after the target date",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform the restore. Without this flag, only a dry run is shown.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of parallel workers for restore operations (default: 10)",
    )
    return parser.parse_args()


def build_target_dt(target_str, tz_offset_str):
    naive = datetime.fromisoformat(target_str)
    if naive.tzinfo is not None:
        return naive
    sign = 1 if tz_offset_str[0] == "+" else -1
    parts = tz_offset_str[1:].split(":")
    hours, minutes = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    tz = timezone(timedelta(hours=sign * hours, minutes=sign * minutes))
    return naive.replace(tzinfo=tz)


def restore_version(client, bucket_name, object_name, generation):
    """Copy a specific generation of an object over the current live version."""
    bucket = client.bucket(bucket_name)
    source_blob = bucket.blob(object_name, generation=generation)
    dest_blob = bucket.blob(object_name)
    # Rewrite (copy) the old generation to become the new live version
    dest_blob.rewrite(source_blob)
    return object_name, generation, True


def delete_object(client, bucket_name, object_name):
    """Delete the current live version of an object."""
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    blob.delete()
    return object_name, None, True


def main():
    args = parse_args()
    target_dt = build_target_dt(args.target, args.tz_offset)

    mode = "EXECUTE" if args.execute else "DRY RUN"

    print("=" * 70)
    print(f"GCS RESTORE ({mode})")
    print("=" * 70)
    print(f"  Bucket:      {args.bucket}")
    print(f"  Prefix:      {args.prefix or '(entire bucket)'}")
    print(f"  Target date: {target_dt.isoformat()}")
    print(f"  Delete new:  {args.delete_new}")
    print(f"  Workers:     {args.workers}")
    print("=" * 70)
    print()

    client = storage.Client()
    bucket = client.bucket(args.bucket)
    bucket.reload()

    if not bucket.versioning_enabled:
        print("ERROR: Object versioning is not enabled on this bucket.")
        print("       Cannot restore without versioned objects.")
        sys.exit(1)

    # Collect all versions
    print("Scanning object versions...")
    versions_by_name = defaultdict(list)
    total_versions = 0

    blobs = bucket.list_blobs(prefix=args.prefix or None, versions=True)
    for blob in blobs:
        versions_by_name[blob.name].append(
            {
                "generation": blob.generation,
                "updated": blob.updated,
                "size": blob.size,
                "is_live": not blob.time_deleted,
            }
        )
        total_versions += 1
        if total_versions % 1000 == 0:
            print(f"  ...scanned {total_versions} versions...")

    print(f"  Done. {total_versions} versions across {len(versions_by_name)} objects.")
    print()

    # Build action plan
    to_restore = []  # (object_name, target_generation, size)
    to_delete = []   # (object_name,)
    already_ok = 0

    for name, vers in sorted(versions_by_name.items()):
        vers.sort(key=lambda v: v["updated"], reverse=True)

        live = [v for v in vers if v["is_live"]]
        current = live[0] if live else None

        best_at_target = None
        for v in vers:
            if v["updated"] <= target_dt:
                best_at_target = v
                break

        if best_at_target is None:
            # Object created after target date
            if args.delete_new:
                to_delete.append(name)
        elif current and current["generation"] == best_at_target["generation"]:
            already_ok += 1
        else:
            to_restore.append((name, best_at_target["generation"], best_at_target["size"]))

    # Summary
    restore_gb = sum(s for _, _, s in to_restore) / (1024 ** 3)
    print(f"  Restore: {len(to_restore)} objects ({restore_gb:.2f} GB)")
    print(f"  Delete:  {len(to_delete)} objects")
    print(f"  No-op:   {already_ok} objects already at correct version")
    print()

    if not to_restore and not to_delete:
        print("Nothing to do!")
        return

    if not args.execute:
        print("=" * 70)
        print("DRY RUN — no changes made. Pass --execute to perform the restore.")
        print("=" * 70)
        return

    # Confirm
    print("!" * 70)
    print(f"ABOUT TO RESTORE {len(to_restore)} objects and DELETE {len(to_delete)} objects.")
    response = input("Type 'yes' to proceed: ")
    if response.strip().lower() != "yes":
        print("Aborted.")
        sys.exit(0)
    print()

    # Execute restores
    succeeded = 0
    failed = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}

        for name, generation, size in to_restore:
            f = executor.submit(restore_version, client, args.bucket, name, generation)
            futures[f] = ("restore", name)

        for name in to_delete:
            f = executor.submit(delete_object, client, args.bucket, name)
            futures[f] = ("delete", name)

        total = len(futures)
        for i, future in enumerate(as_completed(futures), 1):
            action, name = futures[future]
            try:
                future.result()
                succeeded += 1
            except Exception as e:
                failed += 1
                print(f"  FAILED ({action}) {name}: {e}")

            if i % 50 == 0 or i == total:
                elapsed = time.time() - start_time
                print(f"  Progress: {i}/{total} ({elapsed:.0f}s elapsed)")

    elapsed = time.time() - start_time
    print()
    print("=" * 70)
    print(f"RESTORE COMPLETE in {elapsed:.1f}s")
    print(f"  Succeeded: {succeeded}")
    print(f"  Failed:    {failed}")
    print("=" * 70)


if __name__ == "__main__":
    main()
