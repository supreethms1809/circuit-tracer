#!/usr/bin/env python3
"""Generate a Spline-CLT benchmark suite config from the RAVEL dataset.

This script formats RAVEL templates and entities, maps the target and foil tokens
correctly (including tokenizer prepended spaces), and outputs a config JSON.
"""

import argparse
import json
import os
import sys
from datasets import load_dataset
from transformers import AutoTokenizer

def clean_token(token: str, tokenizer) -> str:
    """Ensure the token starts with a space if required by the tokenizer.
    
    Hugging Face tokenizers treat leading spaces as part of the token.
    """
    if not token.startswith(" "):
        token_with_space = " " + token
    else:
        token_with_space = token
        
    # Check if tokenizing with space results in a single/better token
    tids = tokenizer.encode(token_with_space, add_special_tokens=False)
    if len(tids) > 0:
        return token_with_space
    return token

def main():
    parser = argparse.ArgumentParser(description="Generate RAVEL benchmark config")
    parser.add_argument("--model", type=str, default="gpt2", help="Tokenizer model name")
    parser.add_argument("--entity-config", type=str, default="city_entity", help="RAVEL entity configuration")
    parser.add_argument("--prompt-config", type=str, default="city_prompt", help="RAVEL prompt configuration")
    parser.add_argument("--limit-entities", type=int, default=10, help="Limit number of entities")
    parser.add_argument("--limit-prompts", type=int, default=2, help="Limit templates per attribute")
    parser.add_argument("--output", type=str, default="experiments/paper_configs/suites/ravel_city_suite.json",
                        help="Path to write the config suite JSON")
    args = parser.parse_args()

    print(f"Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print("Loading RAVEL dataset...")
    entities_ds = load_dataset("hij/ravel", args.entity_config)["train"]
    prompts_ds = load_dataset("hij/ravel", args.prompt_config)["train"]

    # Select distinct entities
    entities = []
    seen_cities = set()
    for row in entities_ds:
        city = row["City"]
        if city not in seen_cities:
            seen_cities.add(city)
            entities.append(row)
        if len(entities) >= args.limit_entities:
            break

    # Group prompt templates by attribute
    attribute_templates = {}
    for row in prompts_ds:
        attr = row["Attribute"]
        if attr not in attribute_templates:
            attribute_templates[attr] = []
        if len(attribute_templates[attr]) < args.limit_prompts:
            if "%s" in row["Template"]:
                attribute_templates[attr].append(row["Template"])

    benchmark_entries = []
    
    # We need to map RAVEL attributes to target and foil tokens
    # Continent foils can be other continents, country foils can be other countries.
    continents = ["Asia", "Europe", "Africa", "North America", "South America", "Oceania"]
    languages = ["English", "Spanish", "French", "German", "Danish", "Italian", "Chinese", "Japanese", "Igbo"]
    
    prompt_counter = 0
    for entity in entities:
        city_name = entity["City"]
        for attr, templates in attribute_templates.items():
            if attr not in entity:
                continue
            
            target_value = entity[attr]
            if not target_value:
                continue
                
            # Determine a foil token based on the attribute family
            foil_value = None
            if attr == "Continent":
                foils = [c for c in continents if c != target_value]
                foil_value = foils[0]
            elif attr == "Language":
                foils = [l for l in languages if l != target_value]
                foil_value = foils[0]
            elif attr == "Country":
                # Fallback to another country
                foil_value = "Germany" if target_value != "Germany" else "France"
            else:
                # Fallback foil
                foil_value = "Unknown"

            for t_idx, template in enumerate(templates):
                prompt_str = template % city_name
                
                # Format target/foil with leading space depending on tokenizer
                target_token = clean_token(target_value, tokenizer)
                foil_token = clean_token(foil_value, tokenizer)
                
                family = "factual"
                if attr == "Continent" or attr == "Country":
                    family = "factual"
                else:
                    family = "ioi" # default to ioi or factual for the runner splits

                entry_id = f"ravel_{city_name.lower().replace(' ', '_')}_{attr.lower()}_{t_idx}"
                
                benchmark_entries.append({
                    "prompt_id": entry_id,
                    "family": family,
                    "prompt": prompt_str,
                    "target_token": target_token,
                    "foil_token": foil_token,
                    "split": "main",
                    "include_macag": True
                })
                prompt_counter += 1

    # Build paper-eval suite config structure
    suite_config = {
        "suite_name": "ravel_eval_suite",
        "description": f"RAVEL benchmark suite for {args.model}",
        "benchmark_manifest_version": "ravel-v1",
        "stages": {
            "collect_dataset": False,
            "train": False,
            "evaluate": True,
            "macag": False,
            "report": True
        },
        "seeds": [101],
        "dataset": {
            "model_name": args.model,
            "dataset_name": "Salesforce/wikitext",
            "dataset_config": "wikitext-2-raw-v1",
            "n_tokens": 100000,
            "seq_len": 128,
            "batch_size": 32,
            "val_fraction": 0.05,
            "device": "cuda",
            "dtype": "bfloat16",
            "val_cache_dir": "/gscratch/ssuresh/shared/activations/paper_v2/gpt2_small/val"
        },
        "model_variants": {
            "spline_feature_match_gpt2_small": {
                "label": "Spline-CLT (feature-match, d_t=4096)",
                "model_name": args.model,
                "scan_id": "spline-fm-gpt2-small",
                "checkpoint_path": "/gscratch/ssuresh/results/paper/paper_v2_gpt2_small_dt4096_2b/runs/spline_feature_match_gpt2_small_pv2/seed_101/checkpoints/spline_dt4096_gpt2_small_pv2_seed101_best",
                "training": {
                    "n_layers": 12,
                    "d_model": 768,
                    "d_transcoder": 4096,
                    "grid_size": 5,
                    "spline_order": 3,
                    "encoder_type": "kan"
                }
            },
            "linear_feature_match_gpt2_small": {
                "label": "Linear CLT (feature-match, d_t=4096)",
                "model_name": args.model,
                "scan_id": "linear-fm-gpt2-small",
                "checkpoint_path": "/gscratch/ssuresh/results/paper/paper_v2_gpt2_small_dt4096_2b/runs/linear_feature_match_gpt2_small_pv2/seed_101/checkpoints/linear_fm_gpt2_small_pv2_2b_seed101_best",
                "training": {
                    "n_layers": 12,
                    "d_model": 768,
                    "d_transcoder": 4096,
                    "encoder_type": "linear"
                }
            }
        },
        "evaluation": {
            "circuit": {
                "top_k_features": 32,
                "run_shapley": False,
                "shapley_samples": 64,
                "alpha": 0.5
            },
            "graph": {
                "max_features": 7500,
                "max_n_logits": 10,
                "desired_logit_prob": 0.99,
                "node_threshold": 0.8,
                "edge_threshold": 0.98,
                "attribution_batch_size": 256,
                "spline_attribution_method": "jacobian_ablation"
            },
            "monosemanticity": {
                "enabled": False,
                "n_features": 100,
                "n_samples": 1000,
                "top_k_examples": 5
            },
            "splines": {
                "enabled": False,
                "n_features": 10,
                "n_samples": 500,
                "top_dims": 3,
                "no_plot": True
            }
        },
        "reporting": {
            "primary_variant": "spline_feature_match_gpt2_small",
            "baseline_variant": "linear_feature_match_gpt2_small",
            "bootstrap_samples": 100,
            "confidence_level": 0.95,
            "figure_case_study_prompt_id": benchmark_entries[0]["prompt_id"] if benchmark_entries else ""
        },
        "benchmark_entries": benchmark_entries
    }

    # Save to output file
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(suite_config, f, indent=2)

    print(f"Successfully generated RAVEL benchmark suite config with {prompt_counter} prompts.")
    print(f"Saved to: {args.output}")

if __name__ == "__main__":
    main()
