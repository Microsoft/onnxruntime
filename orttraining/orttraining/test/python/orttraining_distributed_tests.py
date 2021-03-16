#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import sys
import argparse

from _test_commons import run_subprocess

import logging

logging.basicConfig(
    format="%(asctime)s %(name)s [%(levelname)s] - %(message)s",
    level=logging.DEBUG)
log = logging.getLogger("DistributedTests")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cwd", help="Path to the current working directory")
    return parser.parse_args()


def run_checkpoint_tests(cwd, log):
    log.debug('Running: Checkpoint tests')

    command = [sys.executable, 'orttraining_test_checkpoint.py']

    run_subprocess(command, cwd=cwd, log=log).check_returncode()

def run_distributed_allreduce_tests(cwd, log):
    log.debug('Running: distributed allreduce tests')

    command = [sys.executable, 'orttraining_test_allreduce.py']

    run_subprocess(command, cwd=cwd, log=log).check_returncode()

@unittest.skip('Skipped because pipeline parallel is disabled.')
def run_pipeline_parallel_tests(cwd, log):
    log.debug('Running: pipeline parallel tests')

    command = [sys.executable, 'orttraining_test_dhp_parallel_tests.py']

    run_subprocess(command, cwd=cwd, log=log).check_returncode()

def main():
    import torch
    ngpus = torch.cuda.device_count()

    if ngpus < 2:
        raise RuntimeError("Cannot run distributed tests with less than 2 gpus.")

    args = parse_arguments()
    cwd = args.cwd

    log.info("Running distributed tests pipeline")

    run_checkpoint_tests(cwd, log)

    run_distributed_allreduce_tests(cwd, log)

    run_pipeline_parallel_tests(cwd, log)

    return 0


if __name__ == "__main__":
    sys.exit(main())
