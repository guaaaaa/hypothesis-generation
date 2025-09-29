import argparse
import logging
import re
import time
import pickle
import sys
import os
import math
import json

import random
from typing import Callable, Tuple, Union
import torch
import numpy as np

from hypogenic.extract_label import extract_label_register

from hypogenic.tasks import BaseTask
from hypogenic.prompt import BasePrompt
from hypogenic.utils import set_seed

from hypogenic.algorithm.summary_information import SummaryInformation
from hypogenic.algorithm.generation import DefaultGeneration
from hypogenic.algorithm.inference import (
    DefaultInference,
    OneStepAdaptiveInference,
    FilterAndWeightInference,
    TwoStepAdaptiveInference,
    UpperboundInference,
)
from hypogenic.algorithm.replace import DefaultReplace
from hypogenic.algorithm.update import SamplingUpdate, DefaultUpdate
from hypogenic.logger_config import LoggerConfig

# Import our custom Gemini wrapper
from gemini_wrapper import GeminiWrapper

LoggerConfig.setup_logger(level=logging.INFO)

logger = LoggerConfig.get_logger("HypoGenic")


def load_dict(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

def main():
    # set up tools
    start_time = time.time()
    
    print(f"\n{'='*80}")
    print(f"🚀 HYPOTHESIS GENERATION WITH GEMINI")
    print(f"{'='*80}")

    # For detailed argument descriptions, please run `hypogenic_generation --help` or see `hypogenic_cmd/generation.py`
    task_config_path = "./data/retweet/config.yaml"
    
    # Gemini model configuration
    model_name = "gemini-2.5-flash"  # or "gemini-2.5-pro"
    model_path = None
    model_type = "gemini"
    
    max_num_hypotheses = 3 
    output_folder = f"./outputs/retweet/{model_name}/hyp_{max_num_hypotheses}/"
    old_hypothesis_file = None
    num_init = 3
    num_train = 10 
    num_test = 25
    num_val = 25
    k = 1
    alpha = 5e-1
    update_batch_size = 1
    num_hypotheses_to_update = 1
    save_every_10_examples = 10
    init_batch_size = 3
    init_hypotheses_per_batch = 3 
    cache_seed = None
    temperature = 1e-5
    max_tokens = 100000
    seeds = [42]

    print(f"📋 CONFIGURATION:")
    print(f"   Model: {model_name}")
    print(f"   Task config: {task_config_path}")
    print(f"   Output folder: {output_folder}")
    print(f"   Max hypotheses: {max_num_hypotheses}")
    print(f"   Training samples: {num_train}")
    print(f"   Test samples: {num_test}")
    print(f"   Validation samples: {num_val}")
    print(f"   Seeds: {seeds}")
    print(f"   Temperature: {temperature}")
    print(f"   Max tokens: {max_tokens}")
    print(f"   Async processing: ✅ Enabled")
    print(f"{'='*80}\n")

    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize Gemini API wrapper
    print(f"🔧 INITIALIZING GEMINI API...")
    # Note: You need to set your API key as an environment variable or pass it directly
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY environment variable not set. Please set it to use Gemini API.")
        logger.warning("You can set it with: export GEMINI_API_KEY='your_api_key_here'")
        # You can also set it directly here for testing:
        # api_key = "your_api_key_here"
    
    api = GeminiWrapper(model=model_name, api_key=api_key)
    print(f"✅ Gemini API initialized with model: {model_name}")

    # If implementing a new task, you need to create a new extract_label function and pass in the Task constructor.
    # For existing tasks (shoe, hotel_reviews, retweet, headline_binary), you can use the extract_label_register.
    print(f"📊 LOADING TASK CONFIGURATION...")
    task = BaseTask(
        task_config_path, extract_label=None, from_register=extract_label_register
    )
    print(f"✅ Task configuration loaded from: {task_config_path}")

    for seed_idx, seed in enumerate(seeds):
        print(f"\n🌱 PROCESSING SEED {seed_idx + 1}/{len(seeds)}: {seed}")
        print(f"{'='*60}")
        
        set_seed(seed)
        print(f"📊 Loading data for seed {seed}...")
        train_data, _, _ = task.get_data(num_train, num_test, num_val, seed)
        print(f"✅ Data loaded: {len(train_data)} training samples")
        
        print(f"🔧 Setting up components...")
        prompt_class = BasePrompt(task)
        inference_class = DefaultInference(api, prompt_class, train_data, task)
        generation_class = DefaultGeneration(api, prompt_class, inference_class, task)

        update_class = DefaultUpdate(
            generation_class=generation_class,
            inference_class=inference_class,
            replace_class=DefaultReplace(max_num_hypotheses),
            save_path=output_folder,
            num_init=num_init,
            k=k,
            alpha=alpha,
            update_batch_size=update_batch_size,
            num_hypotheses_to_update=num_hypotheses_to_update,
            save_every_n_examples=save_every_10_examples,
        )
        print(f"✅ Components initialized")

        hypotheses_bank = {}
        if old_hypothesis_file is None:
            print(f"\n🎯 INITIALIZING HYPOTHESES...")
            print(f"   Initial hypotheses: {num_init}")
            print(f"   Batch size: {init_batch_size}")
            print(f"   Hypotheses per batch: {init_hypotheses_per_batch}")
            print(f"   Temperature: {temperature}")
            print(f"   Max tokens: {max_tokens}")
            print(f"   Async processing: ✅ Enabled (concurrent API calls)")
            
            hypotheses_bank = update_class.batched_initialize_hypotheses(
                num_init,
                init_batch_size=init_batch_size,
                init_hypotheses_per_batch=init_hypotheses_per_batch,
                cache_seed=cache_seed,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            print(f"✅ Generated {len(hypotheses_bank)} initial hypotheses")
            
            print(f"💾 Saving initial hypotheses...")
            update_class.save_to_json(
                hypotheses_bank,
                sample=num_init,
                seed=seed,
                epoch=0,
            )
            print(f"✅ Initial hypotheses saved")
        else:
            print(f"📂 Loading existing hypotheses from: {old_hypothesis_file}")
            dict = load_dict(old_hypothesis_file)
            for hypothesis in dict:
                hypotheses_bank[hypothesis] = SummaryInformation.from_dict(
                    dict[hypothesis]
                )
            print(f"✅ Loaded {len(hypotheses_bank)} existing hypotheses")
            
        """for epoch in range(1):
            print(f"\n🔄 UPDATING HYPOTHESES (Epoch {epoch + 1})...")
            print(f"   Current hypotheses: {len(hypotheses_bank)}")
            print(f"   Update batch size: {update_batch_size}")
            print(f"   Hypotheses to update: {num_hypotheses_to_update}")
            
            hypotheses_bank = update_class.update(
                current_epoch=epoch,
                hypotheses_bank=hypotheses_bank,
                current_seed=seed,
                cache_seed=cache_seed,
            )
            print(f"✅ Updated hypotheses: {len(hypotheses_bank)} total")
            
            print(f"💾 Saving final results...")
            update_class.save_to_json(
                hypotheses_bank,
                sample="final",
                seed=seed,
                epoch=epoch,
            )
            print(f"✅ Final results saved")
            
        print(f"🎉 COMPLETED SEED {seed_idx + 1}/{len(seeds)}: {seed}")
        print(f"{'='*60}")

    # print experiment info
    total_time = time.time() - start_time
    print(f"\n🎉 EXPERIMENT COMPLETED!")
    print(f"{'='*80}")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    print(f"📁 Output folder: {output_folder}")
    print(f"🌱 Seeds processed: {len(seeds)}")
    print(f"🤖 Model used: {model_name}")
    print(f"📊 Training samples: {num_train}")
    print(f"🧪 Test samples: {num_test}")
    print(f"✅ Validation samples: {num_val}")
    print(f"{'='*80}")
    logger.info(f"Total time: {total_time:.2f} seconds")
    # TODO: No Implementation for session_total_cost
    # if api.model in GPT_MODELS:
    #     logger.info(f'Estimated cost: {api.api.session_total_cost()}')"""


if __name__ == "__main__":
    main()
