"""Download one pinned 24.4 MB DINOv2-small CPU artifact; verify its SHA-256."""
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

REVISION = "8b1f705a3a7f6f062f6bdd21986c1583d3ef105d"
SHA256 = "c179f8f7f592449c4c1bca4cd124a7538021428c5ffb89afde9503935b197efb"
URL = f"https://huggingface.co/onnx-community/dinov2-small/resolve/{REVISION}/onnx/model_quantized.onnx"


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / "weights"
    folder.mkdir(exist_ok=True)
    target = folder / "dinov2-small-int8.onnx"
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == SHA256:
        print("Verified existing DINOv2-small INT8 artifact")
        return
    if shutil.disk_usage(folder).free < 700_000_000:
        raise RuntimeError("Less than 700 MB free; download cancelled")
    temp = target.with_suffix(".partial")
    try:
        with urllib.request.urlopen(URL, timeout=60) as response, temp.open("wb") as output:
            count = 0
            while chunk := response.read(1024*1024):
                count += len(chunk)
                if count > 26_000_000:
                    raise RuntimeError("Unexpected model size")
                output.write(chunk)
        if hashlib.sha256(temp.read_bytes()).hexdigest() != SHA256:
            raise RuntimeError("Model checksum mismatch")
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)
    (folder / "provenance.json").write_text(json.dumps({"url": URL, "sha256": SHA256,
        "bytes": target.stat().st_size, "base": "facebook/dinov2-small", "format": "community INT8 ONNX export"}, indent=2))
    print(f"Downloaded and verified {target.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
