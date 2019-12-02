"""===============================================================================
  *
  * Author: Arya Nguyen
  * Component: pipelining
  * Description: generic pipeline
  *
=================================================================================="""

from .pipeline_step import PipelineStep


class Pipeline(PipelineStep):
    def __init__(self, data=None, steps=None, **kwargs):
        """

        :param data: must be path of text file or string
        :param steps: must be iterable form
        """
        super(Pipeline, self).__init__(data=data, **kwargs)
        self._steps = steps

    def execute(self, **kwargs):
        super(Pipeline, self).__init__(**kwargs)
        for step_name, step_obj in self._steps:
            self.data = step_obj.execute(data=self.data, **kwargs).data
        return self


