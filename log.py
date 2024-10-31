import logging
import datetime
import os

def setup_logger(output_path:str, log_to_file = True, name = "", add_date = True):
    if add_date:
        name = name + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    os.makedirs(output_path, exist_ok=True)
    _format = "%(asctime)s|%(name)s|%(levelname)s||%(message)s"
    if log_to_file:
        logging.basicConfig(
            level=logging.INFO,
            format=_format,
            handlers=[
                logging.FileHandler(f"{output_path}/{name}.log"),
                logging.StreamHandler(),
            ],
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format=_format,
            handlers=[logging.StreamHandler()],
        )
    logger = logger = logging.getLogger()
    logger.info()
    return logger
