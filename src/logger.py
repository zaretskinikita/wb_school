import logging
import sys
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) 
stdout_handler = logging.StreamHandler(stream=sys.stdout)
format_output = logging.Formatter('%(levelname)s : %(name)s : %(message)s : %(asctime)s')
stdout_handler.setFormatter(format_output)
logger.addHandler(stdout_handler)

def logging(message):
   logger.info(message)