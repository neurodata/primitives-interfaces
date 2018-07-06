from d3m.metadata import pipeline as meta_pipeline

from jhu_primitives.pipelines.base import BasePipeline
from jhu_primitives.lse import LaplacianSpectralEmbedding
from jhu_primitives.gclass import GaussianClassification
from jhu_primitives.lcc import LargestConnectedComponent

DATASETS = {
    'DS18076'
}


class GCLASSoLSE_pipeline(BasePipeline):
    def __init__(self):
        super().__init__(DATASETS)

    def _gen_pipeline(self):
        pipeline = meta_pipeline.Pipeline(context=meta_pipeline.PipelineContext.TESTING)
        pipeline.add_input(name='inputs')

        step_0 = meta_pipeline.PrimitiveStep(primitive_description=LargestConnectedComponent.metadata.query())
        step_0.add_argument(
            name='inputs',
            argument_type=meta_pipeline.ArgumentType.CONTAINER,
            data_reference='inputs.0'
        )

        step_0.add_output('produce')
        pipeline.add_step(step_0)

        step_1 = meta_pipeline.PrimitiveStep(primitive_description=LaplacianSpectralEmbedding.metadata.query())
        step_1.add_argument(
            name='inputs',
            argument_type=meta_pipeline.ArgumentType.CONTAINER,
            data_reference='steps.0.produce'
        )

        step_1.add_output('produce')
        pipeline.add_step(step_1)

        step_2 = meta_pipeline.PrimitiveStep(primitive_description=GaussianClassification.metadata.query())
        step_2.add_argument(
            name='inputs',
            argument_type=meta_pipeline.ArgumentType.CONTAINER,
            data_reference='steps.1.produce'
        )

        step_2.add_output('produce')
        pipeline.add_step(step_2)

        # Adding output step to the pipeline
        pipeline.add_output(name='Predictions', data_reference='steps.1.produce')

        return pipeline

    def assert_result(self, tester, results, dataset):
        tester.assertEquals(len(results), 1)
        # tester.assertEquals(len(results[0]), 1208)
