import abc
import json


#taken from: https://gitlab.datadrivendiscovery.org/dhartnett/psl/blob/master/sri/pipelines/base.py
class BasePipeline(object):
    def __init__(self, datasets):
        self._datasets = datasets
        self._pipeline = self._gen_pipeline()

    @abc.abstractmethod
    def _gen_pipeline(self):
        '''
        Create a D3M pipeline for this class.
        '''
        pass

    @abc.abstractmethod
    def assert_result(self, tester, results, dataset):
        '''
        Make sure that the results from an invocation of this pipeline are valid.
        '''
        pass

    def get_id(self):
        return self._pipeline.id

    def get_datasets(self):
        '''
        Get the name of datasets compatibile with this pipeline.
        '''
        return self._datasets

    def get_json(self):
        # Make it pretty.
        return json.dumps(json.loads(self._pipeline.to_json()), indent = 4)