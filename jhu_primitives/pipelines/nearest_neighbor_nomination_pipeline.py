from d3m import index
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import PrimitiveStep
from d3m.metadata.pipeline import Pipeline
from d3m.metadata import pipeline as meta_pipeline

from jhu_primitives.pipelines.base import BasePipeline
from jhu_primitives.nearest_neighbor_nomination import NearestNeighborNomination
from jhu_primitives.partial_procrustes import PartialProcrustes

DATASETS = {
    'LL1_2734_CLIR'
}

class nearest_neighbor_nomination_pipeline(BasePipeline):
    def __init__(self):
        super().__init__(DATASETS)

    def _gen_pipeline(self):
        pipeline = meta_pipeline.Pipeline()
        pipeline.add_input(name='inputs')

        # Step 0: dataset_to_dataframe
        step_0 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.dataset_to_dataframe.Common'))
        step_0.add_argument(name='inputs',
                            argument_type=ArgumentType.CONTAINER,
                            data_reference='inputs.0')
        step_0.add_output('produce')
        pipeline.add_step(step_0)

        # Step 1: dataset_to_dataframe
        step_1 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.dataset_to_dataframe.Common'))
        step_1.add_argument(name='inputs',
                            argument_type=ArgumentType.CONTAINER,
                            data_reference='inputs.0')
        step_1.add_hyperparameter(
                name='dataframe_resource',
                argument_type = ArgumentType.VALUE,
                data='1'
        )
        step_1.add_output('produce')
        pipeline.add_step(step_1)

        # Step 2: dataset_to_dataframe
        step_2 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.dataset_to_dataframe.Common'))
        step_2.add_argument(name='inputs',
                            argument_type=ArgumentType.CONTAINER,
                            data_reference='inputs.0')
        step_2.add_hyperparameter(
                name='dataframe_resource',
                argument_type=ArgumentType.VALUE,
                data='2'
        )
        step_2.add_output('produce')
        pipeline.add_step(step_2)

        # Step 3
        step_3 = meta_pipeline.PrimitiveStep(
            primitive_description=PartialProcrustes.metadata.query())
        step_3.add_argument(
            name='inputs_1',
            argument_type=ArgumentType.CONTAINER,
            data_reference='steps.1.produce'
        )
        step_3.add_argument(
            name='inputs_2',
            argument_type=ArgumentType.CONTAINER,
            data_reference='steps.2.produce'
        )
        step_3.add_argument(
            name='reference',
            argument_type=ArgumentType.CONTAINER,
            data_reference='steps.0.produce'
        )

        step_3.add_output('produce')
        pipeline.add_step(step_3)

        # Step 4
        step_4 = meta_pipeline.PrimitiveStep(
            primitive_description=NearestNeighborNomination.metadata.query())
        step_4.add_argument(
            name='inputs_1',
            argument_type=ArgumentType.CONTAINER,
            data_reference='steps.1.produce'
        )
        step_4.add_argument(
            name='inputs_2',
            argument_type=ArgumentType.CONTAINER,
            data_reference='steps.3.produce'
        )
        step_4.add_argument(
            name='reference',
            argument_type=ArgumentType.CONTAINER,
            data_reference='steps.0.produce'
        )

        step_4.add_output('produce')
        pipeline.add_step(step_4)

        # Adding output step to the pipeline
        pipeline.add_output(name='Predictions', data_reference='steps.4.produce')

        return pipeline

    def assert_result(self, tester, results, dataset):
        tester.assertEquals(len(results), 1)
        # tester.assertEquals(len(results[0]), 1208)
