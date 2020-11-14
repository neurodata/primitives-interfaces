from d3m import index
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
# from d3m.primitives.feature_construction.deep_feature_synthesis import SingleTableFeaturization
# from d3m.primitives.data_transformation import column_parser


from d3m.metadata import pipeline as meta_pipeline
# from d3m.metadata.base import Context

DATASETS = {
    'LL1_2734_CLIR'
}

class gclass_ase_pipeline(BasePipeline):
    def __init__(self):
        super().__init__(DATASETS)

    def _gen_pipeline(self):
        pipeline = meta_pipeline.Pipeline()
        pipeline.add_input(name='inputs')

        # Step 0: dataset_to_dataframe
        step_0 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.dataset_to_dataframe.Common'))
        step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.learningData')
        step_0.add_output('produce')
        pipeline.add_step(step_0)

        # Step 1: dataset_to_dataframe
        step_1 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.dataset_to_dataframe.Common'))
        step_1.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.1')
        step_1.add_output('produce')
        pipeline.add_step(step_1)

        # Step 2: dataset_to_dataframe
        step_2 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.dataset_to_dataframe.Common'))
        step_2.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.2')
        step_2.add_output('produce')
        pipeline.add_step(step_2)

        return pipeline

    def assert_result(self, tester, results, dataset):
        tester.assertEquals(len(results), 1)
        # tester.assertEquals(len(results[0]), 1208)
