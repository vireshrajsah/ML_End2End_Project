import logging
import os
from datetime import datetime

LOGFILE_NAME= f"log - {datetime.now().strftime('%m_%d_%Y__%H_%M_%S')}.log"
log_path = os.path.join(os.getcwd(),'logs')
#os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(log_path, LOGFILE_NAME)

logging.basicConfig(
    format= "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s: %(message)s",
    filename= LOG_FILE_PATH,
    level=logging.INFO
)