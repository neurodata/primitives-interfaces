# Output pipelines in JSON.

import argparse
import os
import jhu_primitives
from sgc_pipeline import SGC_pipeline

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

    pipeline = SGC_pipeline()
    #for pipeline_class in sri.pipelines.all.get_pipelines():
    #    if (pipeline_class.__name__ == pipeline_name):
    #        pipeline = pipeline_class()
    #        break

    if (pipeline is None):
        raise ValueError("Could not find pipeline with name: %s." % (pipeline_name))

    print(pipeline.get_json())

if __name__ == '__main__':
    main()
