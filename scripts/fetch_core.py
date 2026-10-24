"""Скачивает ядро DOSBox Pure (libretro) в cores/."""

import io
import platform
import sys
import urllib.request
import zipfile

from neuronoid.config import ROOT, ConfigError, core_filename

BUILDBOT = "https://buildbot.libretro.com/nightly"


def core_url() -> str:
    machine = platform.machine().lower()
    if sys.platform == "win32" and machine in ("amd64", "x86_64"):
        return f"{BUILDBOT}/windows/x86_64/latest/dosbox_pure_libretro.dll.zip"
    if sys.platform == "darwin" and machine == "arm64":
        return f"{BUILDBOT}/apple/osx/arm64/latest/dosbox_pure_libretro.dylib.zip"
    raise ConfigError(f"Нет готового ядра для {sys.platform}/{machine}")


def main() -> None:
    url = core_url()
    dest = ROOT / "cores"
    dest.mkdir(exist_ok=True)
    print(f"Скачиваю {url}")
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extract(core_filename(), dest)
    print(f"Готово: {dest / core_filename()}")


if __name__ == "__main__":
    try:
        main()
    except ConfigError as e:
        sys.exit(str(e))
