#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pipeline to reproduce fixed models and evaluation protocols.

This is the main pipeline for the reasoning step in the paper:
Are Disentangled Representations Helpful for Abstract Visual Reasoning?
Sjoerd van Steenkiste, Francesco Locatello, Juergen Schmidhuber, Olivier Bachem.
NeurIPS, 2019.
https://arxiv.org/abs/1905.12506
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from absl import app
from absl import flags
from absl import logging
from disentanglement_lib.config import reproduce
from disentanglement_lib.evaluation import evaluate
from disentanglement_lib.methods.unsupervised import train
from disentanglement_lib.postprocessing import postprocess
from disentanglement_lib.visualize import visualize_model
import numpy as np

FLAGS = flags.FLAGS
flags.DEFINE_string("study", "unsupervised_study_v1",
                    "Name of the study.")
flags.DEFINE_string("output_directory", None,
                    "Output directory of experiments ('{model_num}' will be"
                    " replaced with the model index  and '{study}' will be"
                    " replaced with the study name if present).")
# Model flags. If the model_dir flag is set, then that directory is used and
# training is skipped.
flags.DEFINE_string("model_dir", None, "Directory to take trained model from.")
# Otherwise, the model is trained using the 'model_num'-th config in the study.
flags.DEFINE_integer("model_num", 0,
                     "Integer with model number to train.")
flags.DEFINE_boolean("only_print", False,
                     "Whether to only print the hyperparameter settings.")
flags.DEFINE_boolean("overwrite", False,
                     "Whether to overwrite output directory.")

def get_model_configs(beta):
    bindings = []
    bindings.append("dataset.name = \'cars3d\'")
    bindings.append('encoder.encoder_fn = @conv_encoder')
    bindings.append('decoder.decoder_fn = @deconv_decoder')
    bindings.append('model.name = \'beta_vae\'')
    bindings.append('vae.beta = {}'.format(beta))
    bindings.append('model.model = @vae()')
    bindings.append('model.random_seed = 0')
    config_file = '.\\disentanglement_lib\\config\\unsupervised_study_v1\\model_configs\\shared.gin'
    return bindings, config_file

def main(unused_argv):
  # Obtain the study to reproduce.
  study = reproduce.STUDIES[FLAGS.study]
  for beta in [1e-3, 1e-2, 0.1, 1, 10, 100, 1000]:
      # Set correct output directory.
      if FLAGS.output_directory is None:
          output_directory = os.path.join("output", "{study}", "{beta}")
      else:
        output_directory = FLAGS.output_directory

      # Insert model number and study name into path if necessary.
      output_directory = output_directory.format(beta=str(beta),
                                                 study="benchmark-experiment-6.1")

      # Model training (if model directory is not provided).

      model_bindings, model_config_file = get_model_configs(beta)
      logging.info("Training model...")
      model_dir = os.path.join(output_directory, "model")
      model_bindings = [
          "model.name = '{}'".format(os.path.basename(model_config_file)).replace(".gin", ""),  #,
          #"model.model_num = {}".format(FLAGS.model_num),
      ] + model_bindings
      train.train_with_gin(model_dir, FLAGS.overwrite, [model_config_file],
                           model_bindings)


      # We visualize reconstructions, samples and latent space traversals.
      visualize_dir = os.path.join(output_directory, "visualizations")
      visualize_model.visualize(model_dir, visualize_dir, FLAGS.overwrite)

      # We fix the random seed for the postprocessing and evaluation steps (each
      # config gets a different but reproducible seed derived from a master seed of
      # 0). The model seed was set via the gin bindings and configs of the study.
      random_state = np.random.RandomState(0)

      # We extract the different representations and save them to disk.
      postprocess_config_files = sorted(study.get_postprocess_config_files())
      for config in postprocess_config_files:
        post_name = os.path.basename(config).replace(".gin", "")
        logging.info("Extracting representation %s...", post_name)
        post_dir = os.path.join(output_directory, "postprocessed", post_name)
        postprocess_bindings = [
            "postprocess.random_seed = {}".format(random_state.randint(2**16)),
            "postprocess.name = '{}'".format(post_name)
        ]
        postprocess.postprocess_with_gin(model_dir, post_dir, FLAGS.overwrite,
                                         [config], postprocess_bindings)

      # Iterate through the disentanglement metrics.
      eval_configs = sorted(study.get_eval_config_files())
      for config in postprocess_config_files:
        post_name = os.path.basename(config).replace(".gin", "")
        post_dir = os.path.join(output_directory, "postprocessed",
                                post_name)
        # Now, we compute all the specified scores.
        for gin_eval_config in eval_configs:
          metric_name = os.path.basename(gin_eval_config).replace(".gin", "")
          logging.info("Computing metric '%s' on '%s'...", metric_name, post_name)
          metric_dir = os.path.join(output_directory, "metrics", post_name,
                                    metric_name)
          eval_bindings = [
              "evaluation.random_seed = {}".format(random_state.randint(2**16)),
              "evaluation.name = '{}'".format(metric_name)
          ]
          evaluate.evaluate_with_gin(post_dir, metric_dir, FLAGS.overwrite,
                                     [gin_eval_config], eval_bindings)


if __name__ == "__main__":
  app.run(main)
