import logging

formatter = "[%(levelname)s] %(asctime)s %(message)s"

logging.basicConfig(filename="outputs.log", filemode="w",
                    level=logging.INFO, format=formatter)

logger = logging.getLogger(__name__)
