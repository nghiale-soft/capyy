import logging
import sys
import threading
from pathlib import Path

import uvicorn

from gateway.core.config import load_settings

logger = logging.getLogger("ai-gateway.main")

# Ensure the project root is on sys.path: `gateway/app.py` imports top-level
# modules like `registry`/`router`; when run via the console script
# (`uv run capyy`) or in Docker, CWD is not automatically added.
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _run_dashboard(settings) -> None:
    """Run the dashboard web app on its own port (default 2222)."""
    try:
        uvicorn.run(
            "gateway.webapp:app",
            host=settings.dashboard_host,
            port=settings.dashboard_port,
            reload=False,
            log_level=settings.log_level.lower(),
        )
    except Exception as error:
        logger.error(
            "dashboard web failed to start on %s:%s: %s",
            settings.dashboard_host,
            settings.dashboard_port,
            error,
        )


def main() -> None:
    settings = load_settings()

    # Dashboard runs on its own port; no paths on the API port.
    if settings.dashboard_enabled and settings.dashboard_port != settings.port:
        thread = threading.Thread(
            target=_run_dashboard,
            args=(settings,),
            name="dashboard-web",
            daemon=True,
        )
        thread.start()

    uvicorn.run(
        "gateway.app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
