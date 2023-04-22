import logging
import os
from datetime import datetime

LOGFILE= f"log - {datetime.now().strftime('%m_%d_%Y__%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd,'logs',LOGFILE)
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOGFILE)

logging.basicConfig(
    format= "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s: %(message)s",
    filename= LOGFILE,
    level=logging.INFO
)