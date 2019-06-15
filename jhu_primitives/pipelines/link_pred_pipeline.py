from d3m.metadata import pipeline as meta_pipeline
from d3m.metadata.base import Context, ArgumentType

from jhu_primitives.pipelines.base import BasePipeline
from jhu_primitives.link_pred_graph_reader import LinkPredictionGraphReader as DataConversion
from jhu_primitives.ase import AdjacencySpectralEmbedding
from jhu_primitives.link_pred_rc import LinkPredictionRankClassifier as RankClassification


DATASETS = {
    '59_umls'
}


class link_pred_pipeline(BasePipeline):
    def __init__(self):
        super().__init__(DATASETS)

    def _gen_pipeline(self):
        pipeline = meta_pipeline.Pipeline()
        pipeline.add_input(name = 'inputs')

        step_0 = meta_pipeline.PrimitiveStep(primitive_description = DataConversion.metadata.query())
        step_0.add_argument(
            name = 'inputs',
            argument_type=ArgumentType.CONTAINER,
            data_reference='inputs.0'
        )

        step_0.add_output('produce')
        pipeline.add_step(step_0)

        step_1 = meta_pipeline.PrimitiveStep(primitive_description = AdjacencySpectralEmbedding.metadata.query())
        step_1.add_argument(
                name = 'inputs',
                argument_type = ArgumentType.CONTAINER,
                data_reference = 'steps.0.produce'
        )
        step_1.add_hyperparameter(
                name = 'which_elbow',
                argument_type = ArgumentType.VALUE,
                data = 1
        )
        step_1.add_hyperparameter(
                name = 'max_dimension',
                argument_type = ArgumentType.VALUE,
                data = 2
        )
        step_1.add_hyperparameter(
                name = 'use_attributes',
                argument_type = ArgumentType.VALUE,
                data = False
        )

        step_1.add_output('produce')
        pipeline.add_step(step_1)


        step_2 = meta_pipeline.PrimitiveStep(primitive_description = RankClassification.metadata.query())
        step_2.add_argument(
            name = 'inputs',
            argument_type= ArgumentType.CONTAINER,
            data_reference=  'steps.1.produce'
        )

        step_2.add_output('produce')
        pipeline.add_step(step_2)

        # Adding output step to the pipeline
        pipeline.add_output(name = 'Predictions', data_reference = 'steps.2.produce')

        return pipeline

    def assert_result(self, tester, results, dataset):
        tester.assertEquals(len(results), 1)
        #tester.assertEquals(len(results[0]), 1208)
