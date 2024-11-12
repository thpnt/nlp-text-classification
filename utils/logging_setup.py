import time, logging

def logging_setup():
    # Logging configuration
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    # Setup run logger
    run_logger = logging.getLogger("preprocessing")
    run_logger.setLevel(logging.INFO)
    run_handler = logging.FileHandler(f"logs/preprocessing/{timestamp}run.log", mode='a')
    run_handler.setLevel(logging.INFO)

    # Setup error logger
    error_logger = logging.getLogger("error_logger")
    error_logger.setLevel(logging.ERROR)
    error_handler = logging.FileHandler(f"logs/preprocessing/{timestamp}text_processing_errors.log", mode='a')
    error_handler.setLevel(logging.ERROR)

    # Set log format
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    run_handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)

    # Add handlers to loggers
    run_logger.addHandler(run_handler)
    error_logger.addHandler(error_handler)

    # Add console handler to run logger
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    run_logger.addHandler(console_handler)
    
    return run_logger, error_logger