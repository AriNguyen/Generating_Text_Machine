"""===============================================================================
  *
  * Author: Arya Nguyen
  * Component: pipelining
  * Description: Abstract base class that defines the pipeline step
  *
=================================================================================="""

import os
import codecs

from abc import ABCMeta, abstractmethod
from .pipeline_data import PData


class PipelineStep(object):
    def __init__(self, data=None):
        __metaclass__ = ABCMeta
        if data is not None:
            if isinstance(data, PData):
                self.data = data
            else:
                self.set_data(data)

    @abstractmethod
    def execute(self, data=None):
        if data is not None:
            if isinstance(data, PData):
                self.data = data
            else:
                self.set_data(data)
        return self

    def set_data(self, data):
        if os.path.isfile(data):
            with codecs.open(data, "r", encoding='utf-8-sig') as f:
                self.data = PData(f.read())
        elif os.path.isdir(data):
            raise ValueError("TypeError: Expected file path")
        elif isinstance(data, str):
            self.data = PData(data)
