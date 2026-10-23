#!/usr/bin/env python3
"""
nerdsight - deploy_space.py
Publishes web/dist to a free Hugging Face Static Space.

    hf auth login                          # once; or export HF_TOKEN=...
    python deploy_space.py                 # export, then upload to <your-user>/nerdsight
    python deploy_space.py --repo org/name --dry-run

The token comes from HF_TOKEN or the `hf auth login` cache and is never written
anywhere by this script.
"""

import argparse
import os
import sys

import export_web

# Remote paths this script owns. Anything else in the Space repo (for example
# .gitattributes) is left alone; stale files under these patterns are removed.
MANAGED = ["*.html", "*.css", "README.md", "js/*", "ui/*", "data/*", "fonts/*"]


def main():
    parser = argparse.ArgumentParser(description="Deploy the static front end to a Hugging Face Space.")
    parser.add_argument("--repo", default=os.environ.get("NERDSIGHT_SPACE"),
                        help="Space id, e.g. user/nerdsight (default: $NERDSIGHT_SPACE or <your-user>/nerdsight)")
    parser.add_argument("--skip-export", action="store_true", help="Upload web/dist as it is")
    parser.add_argument("--dry-run", action="store_true", help="Export and list the files, but do not upload")
    args = parser.parse_args()

    if not args.skip_export:
        export_web.main([])
    if not os.path.exists(os.path.join(export_web.DIST_DIR, "index.html")):
        print("web/dist is empty. Run without --skip-export.", file=sys.stderr)
        sys.exit(1)

    files = sorted(
        os.path.relpath(os.path.join(root, f), export_web.DIST_DIR)
        for root, _, names in os.walk(export_web.DIST_DIR) for f in names
    )
    total = sum(os.path.getsize(os.path.join(export_web.DIST_DIR, f)) for f in files)
    print(f"{len(files)} files, {total / 1e6:.1f} MB")
    if args.dry_run:
        print("\n".join(f"  {f}" for f in files))
        return

    from huggingface_hub import HfApi
    from huggingface_hub.errors import LocalTokenNotFoundError

    api = HfApi()
    try:
        user = api.whoami()["name"]
    except LocalTokenNotFoundError:
        print("Not logged in. Run `hf auth login` or set HF_TOKEN (write access).", file=sys.stderr)
        sys.exit(1)
    repo = args.repo or f"{user}/nerdsight"

    url = api.create_repo(repo, repo_type="space", space_sdk="static", exist_ok=True)
    print(f"Uploading to {url} ...")
    commit = api.upload_folder(
        repo_id=repo,
        repo_type="space",
        folder_path=export_web.DIST_DIR,
        delete_patterns=MANAGED,
        commit_message=f"Deploy nerdsight ({len(files)} files)",
    )
    subdomain = repo.replace("/", "-").replace("_", "-").replace(".", "-").lower()
    print(f"Done: {commit.commit_url}")
    print(f"Live: https://huggingface.co/spaces/{repo}  (direct: https://{subdomain}.static.hf.space)")


if __name__ == "__main__":
    main()
