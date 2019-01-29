from d3m.metadata import pipeline as meta_pipeline
from d3m.metadata.base import Context, ArgumentType

from jhu_primitives.pipelines.base import BasePipeline
from jhu_primitives.sgm import SeededGraphMatching
from jhu_primitives.sgm import Hyperparams


DATASETS = {
    '49_facebook'
}

class sgm_pipeline_10(BasePipeline):
    def __init__(self):
        super().__init__(DATASETS)

    def _gen_pipeline(self):
        pipeline = meta_pipeline.Pipeline(context = Context.TESTING)
        pipeline.add_input(name = 'inputs')

        step_0 = meta_pipeline.PrimitiveStep(primitive_description = SeededGraphMatching.metadata.query())
        step_0.add_argument(
                name = 'inputs',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'inputs.0'
        )
        step_0.add_hyperparameter(
                name = 'reps',
                argument_type = ArgumentType.VALUE,
                data = 10
        )
        step_0.add_hyperparameter(
                name = 'threshold',
                argument_type = ArgumentType.VALUE,
                data = 0.1
        )
        step_0.add_output('produce')
        pipeline.add_step(step_0)

        # Adding output step to the pipeline
        pipeline.add_output(name = 'Predictions', data_reference = 'steps.0.produce')

        return pipeline

    def assert_result(self, tester, results, dataset):
        tester.assertEquals(len(results), 1)
        tester.assertEquals(len(results[0]), 1208)
