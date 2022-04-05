from argparse import ArgumentParser, Namespace
import logging

from movement_params.loop import loop


# Parse arguments
argument_parser = ArgumentParser(
    description='Software for calculating movement parameters of objects on video stream',
    epilog='(c) Alyukov Danila, Gerasimovich Ilya, Goldobin Ilya, Chernov Maxim\n'
           'ITMO University 2022'
)
argument_parser.add_argument('--debug', action='store_true', help='Debug mode')

ARGS: Namespace = argument_parser.parse_args()

# Change config to DEBUG if key founded
if ARGS.debug:
    logging.info('Program runs in DEBUG mode')
else:
    logging.info('Program runs in DEFAULT mode')

# Start program
loop()
