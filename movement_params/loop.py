import logging

from movement_params import CONFIG


def loop():
    try:
        while True:
            frame = CONFIG.input_type.get_frame()
            if frame:
                for processor in CONFIG.processors:
                    frame = processor.process(frame)

                CONFIG.output_type.push(frame)
    except KeyboardInterrupt:
        logging.info('Keyboard interrupt. Program stopped')

    except Exception as e:
        logging.exception(e)
