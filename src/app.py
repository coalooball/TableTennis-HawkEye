from __future__ import annotations

from pathlib import Path
import threading

from pytauri_wheel.lib import builder_factory, context_factory
from pytauri_plugins import dialog
import uvicorn

from api import app as fastapi_app


ROOT_DIR = Path(__file__).resolve().parent.parent
TAURI_DIR = ROOT_DIR / "src" / "src-tauri"
APP_ICON_PATH = ROOT_DIR / "src" / "assets" / "app.png"


def start_api_server() -> None:
    config = uvicorn.Config(fastapi_app, host="127.0.0.1", port=8765, log_level="warning")
    server = uvicorn.Server(config)
    server.run()


def set_macos_app_icon() -> None:
    try:
        from AppKit import NSApplication, NSImage
    except Exception:
        return

    image = NSImage.alloc().initWithContentsOfFile_(str(APP_ICON_PATH))
    if image is not None:
        NSApplication.sharedApplication().setApplicationIconImage_(image)


def main() -> None:
    set_macos_app_icon()
    threading.Thread(target=start_api_server, daemon=True).start()
    context = context_factory(TAURI_DIR)
    app = builder_factory().build(context, invoke_handler=None, plugins=[dialog.init()])
    app.run()


if __name__ == "__main__":
    main()
