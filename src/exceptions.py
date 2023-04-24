import sys
from src.logger import logging

def error_message_detail(error, error_detail:sys):
    '''
    Gets error message details from superclass of the Exception class
    '''
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error logged from exceptions: python script [{0}], line number [{1}] Error message[{2}]".format(file_name, exc_tb.tb_lineno, str(error))
    logging.info(error_message)
    return error_message

class CustomException(Exception):
    """
    Custom Exception class, subclass of python Exception class
    """
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message
    

# if __name__ == "__main__":
#     logging.info("Log initiated")
#     try:
#         a=1/0
#     except Exception as e:
#         logging.info(f"test error raised with message - {e}")
#         raise CustomException(e,sys)
