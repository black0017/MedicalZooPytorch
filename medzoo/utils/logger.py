#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Input/output functions."""

# Standard library
import logging
import os
import sys

from medzoo.utils import make_dirs_if_not_present
from medzoo.utils.timer import Timer


class Logger:
    def __init__(self, log_lovel=None, name=None):
        self.logger = None
        self.timer = Timer()

        self.log_filename = "train_"
        self.log_filename += self.timer.get_time()
        self.log_filename += ".log"

        self.log_folder = '../logs/'
        make_dirs_if_not_present(self.log_folder)

        self.log_filename = os.path.join(self.log_folder, self.log_filename)

        logging.captureWarnings(True)

        if not name:
            name = __name__

        self.logger = logging.getLogger(name)

        # Set level
        if log_lovel is None:
            level = 'INFO'
        else:
            level = log_lovel
        self.logger.setLevel(getattr(logging, level.upper()))

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d-%H:%M:%S",
        )

        # Add handlers
        file_hdl = logging.FileHandler(self.log_filename)
        file_hdl.setFormatter(formatter)
        self.logger.addHandler(file_hdl)
        # logging.getLogger('py.warnings').addHandler(file_hdl)
        cons_hdl = logging.StreamHandler(sys.stdout)
        cons_hdl.setFormatter(formatter)
        self.logger.addHandler(cons_hdl)

    def get_logger(self):
        return self.logger

