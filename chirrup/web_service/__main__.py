from chirrup.web_service.app import app
from chirrup.web_service.config import get_config
import uvicorn

if __name__ == "__main__":
    config = get_config()
    uvicorn.run(app, host=config.host, port=config.port, reload=False, log_level="info")
