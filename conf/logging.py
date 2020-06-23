from .base import BASE_PATH, register_file, register_directory

LOG_PATH = register_file(BASE_PATH, 'data/log/app.log')
MASK_IMG_LOG_PATH = register_directory(BASE_PATH, 'data/log/img')

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {
            "format": "[%(asctime)s] %(levelname)-8s %(name)s %(funcName)s:%(lineno)d - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "simple",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "simple",
            "filename": LOG_PATH,
            "maxBytes": 10485760,
            "backupCount": 20,
            "encoding": "utf8",
        }
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console", "file"]
    }
}
