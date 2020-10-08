import logging

formatter = "[%(levelname)s] %(asctime)s %(message)s"

logging.basicConfig(filename="outputs.log",
                    level=logging.DEBUG, format=formatter)

logger = logging.getLogger(__name__)
