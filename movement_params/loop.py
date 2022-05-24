import logging
import cv2
from movement_params import CONFIG


def loop():
    try:
        while True:
            frame = CONFIG.input_type.get_frame()
            if frame:
                for processor in CONFIG.processors:
                    frame = processor.process(frame)

                CONFIG.output_type.push(frame)
            # if cv2.getWindowProperty(CONFIG.output_type.window_title, cv2.WND_PROP_VISIBLE) < 1:
            #     print("Window closed. Terminating...")
            #     break
    except KeyboardInterrupt:
        logging.info('Keyboard interrupt. Program stopped')

    except Exception as e:
        logging.exception(e)
