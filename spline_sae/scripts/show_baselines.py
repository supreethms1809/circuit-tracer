#!/usr/bin/env python3
"""Print the Neuronpedia / HF baseline table and next commands."""

from __future__ import annotations

BASELINES = [
    {
        "model": "google/gemma-2-2b",
        "sae": "Gemma Scope (JumpReLU)",
        "neuronpedia": "https://www.neuronpedia.org/ (Gemma Scope)",
        "hf": "https://huggingface.co/google/gemma-scope",
        "train": "HF JumpReLU Colab + arxiv:2408.05147",
        "config": "spline_sae/configs/gemma2_2b_res_l12.yaml",
    },
    {
        "model": "meta-llama/Llama-3.1-8B",
        "sae": "Llama Scope (TopK)",
        "neuronpedia": "Neuronpedia Llama Scope release",
        "hf": "https://huggingface.co/OpenMOSS-Team/Llama-Scope",
        "train": "https://github.com/OpenMOSS/Language-Model-SAEs",
        "config": "spline_sae/configs/llama31_8b_res_l16.yaml",
    },
    {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "sae": "Chanin Matryoshka (SAELens)",
        "neuronpedia": "https://www.neuronpedia.org/qwen2.5-7b-it-sae",
        "hf": "https://huggingface.co/chanind/qwen2.5-7B-it-layer-20-saes",
        "train": "https://github.com/decoderesearch/SAELens",
        "config": "spline_sae/configs/qwen25_7b_it_res_l20.yaml",
    },
]


def main() -> None:
    print("Spline-SAE Neuronpedia baselines\n")
    for i, b in enumerate(BASELINES, 1):
        print(f"{i}. {b['model']}")
        print(f"   SAE:        {b['sae']}")
        print(f"   Neuronpedia:{b['neuronpedia']}")
        print(f"   HF:         {b['hf']}")
        print(f"   Train:      {b['train']}")
        print(f"   Config:     {b['config']}")
        print()
    print("Next:")
    print("  conda run -n ct pytest spline_sae/tests -q")
    print("  # then Phase-1: load published SAE + collect 1-layer activations")


if __name__ == "__main__":
    main()
