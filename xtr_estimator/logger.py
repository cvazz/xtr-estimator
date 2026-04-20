import logging


class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    multiline_indent = "  "
    # datefmt = "%Y-%m-%d %H:%M:%S"
    datefmt = "%H:%M:%S"
    format = "%(levelname)s %(asctime)s - %(message)s (%(filename)s:%(lineno)d)"
    format = (
        "%(asctime)s: %(name)s: %(levelname)s: %(message)s (%(filename)s:%(lineno)d)"
    )
    format = (
        "%(asctime)s: %(name)s: %(levelname)s: %(message)s (%(filename)s:%(lineno)d)"
    )

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        # format = "%(asctime)s: %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
        # datefmt = "%H:%M:%S"
        log_fmt = self.FORMATS.get(record.levelno)
        # Keep multiline records close to the left edge so table-like output is readable.
        indent = self.multiline_indent
        if record.msg and isinstance(record.msg, str):
            record.msg = record.msg.replace("\n", "\n" + indent)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def setup_logger(log_level=logging.DEBUG):
    """
    Set up the logger with a custom formatter.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(CustomFormatter())
    root_logger.handlers = []  # Remove any default handlers
    root_logger.addHandler(ch)
    logging.getLogger("generate_objects").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    return logger
