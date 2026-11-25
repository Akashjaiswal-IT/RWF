"""Universal production & local entrypoint respecting cloud $PORT."""

import os
import uvicorn


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    reload_flag = os.environ.get("RWF_RELOAD", "false").lower() == "true"
    uvicorn.run("backend.main:app", host=host, port=port, reload=reload_flag)
