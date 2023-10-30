import os
import sys
import time
import pandas as pd
from prettytable import PrettyTable
from tabulate import tabulate
import textwrap
import re

class CustomLogger:
    def __init__(self, name=None):
        self.name = name
        self.log_file = name + '.log' if name else None
        if self.log_file:
            self.log_fd = open(self.log_file, 'a', encoding='utf-8')  # Set encoding to UTF-8
        else:
            self.log_fd = None

    def __del__(self):
        if self.log_fd:
            self.log_fd.close()

    def _get_timestamp(self):
        return time.strftime('%Y-%m-%d %H:%M:%S')

    def _format_floats_in_string(self, message):
        # Regular expression to find floating-point numbers within the message
        float_pattern = r"\b\d+\.\d+\b"

        def format_float(match):
            return f"{float(match.group()):.4f}"

        return re.sub(float_pattern, format_float, message)
    
    def _format_message(self, level, message):
        timestamp = self._get_timestamp()
        return f"[{timestamp}] {level}: {message}"

    def info(self, message):
        formatted_msg = self._format_message('INFO', self._auto_format_message(message))
        print(formatted_msg)
        if self.log_fd:
            self.log_fd.write(formatted_msg + '\n')
            self.log_fd.flush()  # Flush the buffer to write to the file immediately

    def _get_dataframe_pretty_table(self, df):
        table_str = str('\n' + tabulate(df, headers='keys', tablefmt='fancy_grid', floatfmt='.4g', showindex=False))
        return table_str

    def _auto_format_message(self, message, max_length=80):
        if isinstance(message, pd.DataFrame):
            return self._get_dataframe_pretty_table(message)

        message = self._format_floats_in_string(message)
        # Check if the message exceeds the max_length
        if len(message) > max_length:
            
            max_length=max_length+28
            
            # Wrap the message into multiple lines
            wrapped_lines = textwrap.wrap(message, width=max_length)

            # Add dots to fill each line up to max_length
            wrapped_message = '\n'+'\n'.join(line + '.' * (max_length - len(line)) for line in wrapped_lines)
            return wrapped_message

        # Calculate the number of dots needed for right-fill
        dots_needed = max_length - len(message)
        if dots_needed < 0:
            dots_needed = 0

        # Right-fill the message with dots
        dots = '.' * dots_needed
        return f"{message}{dots}"

    def print(self, *args, sep=' ', end='\n', file=None):
        message = sep.join(map(str, args)) + end
        formatted_msg = self._format_message('PRINT', self._auto_format_message(message))
        print(formatted_msg)
        if self.log_fd:
            self.log_fd.write(formatted_msg + '\n')
            self.log_fd.flush()  # Flush the buffer to write to the file immediately

    def set_log_file(self, new_filename):
        self.info('Logfile saved to: {}.log'.format(new_filename))
        if self.log_fd:
            self.log_fd.close()

        if new_filename:
            new_log_file = new_filename + '.log'
            self.log_fd = open(new_log_file, 'a', encoding='utf-8')  # Set encoding to UTF-8
            self.log_file = new_log_file
        else:
            self.log_file = None
            self.log_fd = None


# Example Usage
if __name__ == "__main__":
    # Initiate the logger without a name
    logger = CustomLogger()
    logger.info("This is an information message.")
    logger.print("This is a print message.")

    # Change log file name and continue logging
    new_filename = "new_log"
    logger.set_log_file(new_filename)
    logger.info("Now logging to a new log file.")
    logger.print("This message will be in the new log file.")

    # Disable log file saving
    logger.set_log_file(None)
    logger.info("This message will not be saved to any log file.")
    logger.print("This message will only be printed to the console.")
