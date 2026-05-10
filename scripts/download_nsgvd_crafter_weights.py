#!/usr/bin/env python3
from __future__ import annotations

import os
import time
from pathlib import Path

import paramiko


HOST = "10.120.16.228"
USER = "shiyao"
PASSWORD = "shiyao"

REMOTE_FILES = [
    "/ssd/shiyao/NSG-VD/results/ckpts/baselines/standard-Crafter-npr/best_acc_ckpt.pth",
    "/ssd/shiyao/NSG-VD/results/ckpts/baselines/standard-Crafter-npr/best_auroc_ckpt.pth",
    "/ssd/shiyao/NSG-VD/results/ckpts/baselines/standard-Crafter-npr/final_ckpt.pth",
    "/ssd/shiyao/NSG-VD/results/ckpts/baselines/standard-Crafter-tall/best_acc_ckpt.pth",
    "/ssd/shiyao/NSG-VD/results/ckpts/baselines/standard-Crafter-tall/best_auroc_ckpt.pth",
    "/ssd/shiyao/NSG-VD/results/ckpts/baselines/standard-Crafter-tall/final_ckpt.pth",
    "/ssd/shiyao/NSG-VD/results/ckpts/baselines/standard-Crafter-stil/best_acc_ckpt.pth",
    "/ssd/shiyao/NSG-VD/results/ckpts/baselines/standard-Crafter-stil/best_auroc_ckpt.pth",
    "/ssd/shiyao/NSG-VD/results/ckpts/baselines/standard-Crafter-stil/final_ckpt.pth",
    "/ssd/shiyao/NSG-VD/results/ckpts/baselines/standard-Crafter-demamba/best_acc_ckpt.pth",
    "/ssd/shiyao/NSG-VD/results/ckpts/baselines/standard-Crafter-demamba/best_auroc_ckpt.pth",
    "/ssd/shiyao/NSG-VD/results/ckpts/baselines/standard-Crafter-demamba/final_ckpt.pth",
]


def connect_with_retry(max_attempts: int = 5, delay_sec: int = 5) -> paramiko.SSHClient:
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(
                hostname=HOST,
                username=USER,
                password=PASSWORD,
                timeout=30,
                banner_timeout=30,
                auth_timeout=30,
            )
            return client
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            print(f"RETRY_CONNECT\t{attempt}/{max_attempts}\t{exc}", flush=True)
            client.close()
            if attempt < max_attempts:
                time.sleep(delay_sec)
    assert last_exc is not None
    raise last_exc


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    dest_root = repo_root / "downloads" / "nsgvd-crafter-weights"
    dest_root.mkdir(parents=True, exist_ok=True)

    client = connect_with_retry()
    sftp = client.open_sftp()
    try:
        for remote_path in REMOTE_FILES:
            rel = remote_path.replace("/ssd/shiyao/NSG-VD/", "")
            local_path = dest_root / rel.replace("/", os.sep)
            local_path.parent.mkdir(parents=True, exist_ok=True)

            for attempt in range(1, 6):
                try:
                    attrs = sftp.stat(remote_path)
                    size = attrs.st_size
                    if local_path.exists() and local_path.stat().st_size == size:
                        print(f"SKIP\t{size}\t{remote_path}\t{local_path}", flush=True)
                        break
                    print(f"DOWNLOAD\t{size}\t{remote_path}\t{local_path}", flush=True)
                    sftp.get(remote_path, str(local_path))
                    break
                except Exception as exc:  # noqa: BLE001
                    print(f"RETRY_FILE\t{attempt}/5\t{remote_path}\t{exc}", flush=True)
                    try:
                        sftp.close()
                    except Exception:  # noqa: BLE001
                        pass
                    client.close()
                    if attempt >= 5:
                        raise
                    time.sleep(5)
                    client = connect_with_retry()
                    sftp = client.open_sftp()
        print(f"DONE\t{dest_root}", flush=True)
        return 0
    finally:
        try:
            sftp.close()
        except Exception:  # noqa: BLE001
            pass
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
