"""
Checks outbound network access to everything this pipeline needs:
  - S3 (audio downloads via ffmpeg)
  - Hugging Face Hub (CLAP + sentence-transformers model weights)
  - Postgres RDS (only needed if you run recording_recommender.py here too)
  - A generic external host, to distinguish "no outbound at all" from
    "outbound exists but these specific hosts are blocked"

IMPORTANT: run this via your job scheduler (srun/sbatch) so it executes on
an actual COMPUTE NODE, not just typed into your login-node SSH session --
login nodes almost always have internet even when compute nodes don't, so
testing there would give you a false positive.

USAGE:
  srun --pty python3 check_network_access.py
  # or inside an sbatch script:
  python3 check_network_access.py
"""

import os
import socket
import sys
import time
import urllib.error
import urllib.request

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


TCP_TARGETS = [
    ("S3 (audio storage)", "says-api-streamable-audio-dev.s3.amazonaws.com", 443),
    ("Hugging Face Hub", "huggingface.co", 443),
    ("Generic external host", "api.github.com", 443),
]

HTTP_TARGETS = [
    ("S3 (audio storage)", "https://says-api-streamable-audio-dev.s3.amazonaws.com"),
    ("Hugging Face Hub", "https://huggingface.co"),
]

DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
if DB_HOST:
    TCP_TARGETS.append(("Postgres RDS", DB_HOST, DB_PORT))


def check_tcp(name: str, host: str, port: int, timeout: float = 5.0) -> bool:
    try:
        start = time.time()
        with socket.create_connection((host, port), timeout=timeout):
            elapsed_ms = (time.time() - start) * 1000
            print(f"[OK]   {name:24s} {host}:{port}  ({elapsed_ms:.0f}ms)")
            return True
    except Exception as error:
        print(f"[FAIL] {name:24s} {host}:{port}  -> {error}")
        return False


def check_http(name: str, url: str, timeout: float = 5.0) -> bool:
    """
    Any HTTP status code back (even 403/404) means the network path is open
    and a real server answered -- only a connection-level failure (DNS,
    timeout, refused connection) means outbound access is actually blocked.
    """
    request = urllib.request.Request(url, method="HEAD")

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            print(f"[OK]   {name:24s} {url}  -> HTTP {response.status}")
            return True
    except urllib.error.HTTPError as error:
        print(f"[OK]   {name:24s} {url}  -> HTTP {error.code} (reachable, not a network failure)")
        return True
    except Exception as error:
        print(f"[FAIL] {name:24s} {url}  -> {error}")
        return False


def main() -> None:
    print("=== DNS + TCP connectivity ===")
    tcp_results = [check_tcp(name, host, port) for name, host, port in TCP_TARGETS]

    print("\n=== HTTPS reachability ===")
    http_results = [check_http(name, url) for name, url in HTTP_TARGETS]

    print("\n=== Summary ===")
    if all(tcp_results) and all(http_results):
        print("All checks passed -- outbound access looks available from here.")
        sys.exit(0)

    print("Some checks failed. If EVERYTHING failed, you likely have no")
    print("outbound access at all from this node. If only some failed, those")
    print("specific hosts/ports are probably blocked by a firewall/proxy policy.")
    print("\nMake sure this ran on a COMPUTE NODE (via srun/sbatch), not the")
    print("login node -- login nodes often have internet even when compute")
    print("nodes don't, which would make this check falsely pass.")
    sys.exit(1)


if __name__ == "__main__":
    main()
