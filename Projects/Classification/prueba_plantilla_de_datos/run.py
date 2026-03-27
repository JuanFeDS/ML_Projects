""" Run the main function."""
from src.logger import get_logger

def run():
    """Run the main function."""
    logger = get_logger(__name__)
    logger.info('Main Function')


if __name__ == '__main__':
    run()
