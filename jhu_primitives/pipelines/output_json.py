# Output pipelines in JSON.

import argparse
import os
import jhu_primitives
#from seeded_graph_matching_pipeline import SeededGraphMatchingPipeline
#from gmm_ase_pipeline import GMMoASE_pipeline
#from sgc_pipeline import SGC_pipeline
#from gmm_lse_pipeline import GMMoLSE_pipeline
#from gclass_ase_pipeline import GCLASSoASE_pipeline
from gclassolse_pipeline import GCLASSoLSE_pipeline

def load_args():
    parser = argparse.ArgumentParser(description = "Output a pipeline's JSON")

    parser.add_argument(
        'pipeline', action = 'store', metavar = 'PIPELINE',
        help = "the name of the pipeline to generate",
    )

    arguments = parser.parse_args()

    return arguments.pipeline

def main():
    pipeline_name = load_args()

    pipeline = GMMoLSE_pipeline()
    #for pipeline_class in sri.pipelines.all.get_pipelines():
    #    if (pipeline_class.__name__ == pipeline_name):
    #        pipeline = pipeline_class()
    #        break

    if (pipeline is None):
        raise ValueError("Could not find pipeline with name: %s." % (pipeline_name))

    print(pipeline.get_json())
#
if __name__ == '__main__':
    main()
