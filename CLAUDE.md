# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## MaxText Overview

MaxText is a high-performance LLM training and inference framework built on JAX/Flax, optimized for Google Cloud TPUs and GPUs. The codebase focuses on scalability, supporting models from single-host to thousands of chips while maintaining simplicity through JAX's XLA compiler optimizations.

## Key Commands for Development

### Setup and Environment
```bash
# Initial setup (choose MODE: stable, nightly, or libtpu-only)
bash setup.sh MODE=stable DEVICE=tpu

# Install dependencies
pip install -r requirements.txt

# For GPU setups
bash setup.sh MODE=stable DEVICE=gpu
```

### Training Commands
```bash
# Basic training (modular import style - REQUIRED after April 2025)
python3 -m MaxText.train MaxText/configs/base.yml run_name=my_experiment

# Supervised Fine-Tuning (SFT)
python3 -m MaxText.sft_trainer MaxText/configs/sft.yml run_name=my_sft_experiment use_sft=True

# Train with specific model config
python3 -m MaxText.train MaxText/configs/models/llama2-7b.yml run_name=llama2_train

# Resume from checkpoint
python3 -m MaxText.train MaxText/configs/base.yml run_name=existing_experiment
```

### Testing and Linting
```bash
# Run all unit tests
python3 -m pytest --pyargs MaxText.tests

# Run specific test
python3 -m pytest MaxText.tests.data_loader_test

# Lint the codebase
python3 -m pylint $(git ls-files '*.py')

# Combined test and lint
bash unit_test_and_lint.sh
```

### Inference/Decoding
```bash
# Basic decoding
python3 -m MaxText.decode MaxText/configs/inference.yml run_name=decode_test

# Load quantized checkpoint for inference
python3 -m MaxText.decode MaxText/configs/inference.yml load_parameters_path=gs://path/to/checkpoint
```

## Architecture for Custom Data Loading and Model Modifications

### Data Loading Pipeline Structure

The data loading system is modular and located in `MaxText/input_pipeline/`:

**Core Components:**
- `input_pipeline_interface.py`: Main interface for creating data iterators
- `_hf_data_processing.py`: HuggingFace datasets integration (best for custom data)
- `_grain_data_processing.py`: Grain-based data processing
- `_tfds_data_processing.py`: TensorFlow datasets support
- `synthetic_data_processing.py`: Synthetic data for testing

**Key Data Flow:**
1. Config specifies `dataset_type` (hf, grain, tfds, synthetic)
2. `DataLoader` class (MaxText/data_loader.py) wraps the iterator
3. Data is sharded across devices using `maxtext_utils.get_input_data_sharding()`
4. Batches are reshaped and distributed via JAX sharding

**Customization Points for Data Loading:**
- Modify `_hf_data_processing.py` for custom HuggingFace dataset preprocessing
- Add custom tokenization in `_input_pipeline_utils.py`
- Implement new data sources by creating a new `_*_data_processing.py` module
- Configure data parameters in YAML configs under `dataset_*` keys

### Model Architecture and LM Head

**Model Structure (MaxText/layers/):**
- `models.py`: Main `Transformer` class that orchestrates everything
- `decoders.py`: Core decoder implementation with layer stacking
- `embeddings.py`: Token and position embeddings
- Model-specific layers: `llama2.py`, `gemma.py`, `mixtral.py`, etc.

**LM Head Implementation:**
The LM head is integrated as the `shared_embedding` in the Transformer class:
- Located in `models.py:63-71` - uses weight tying by default
- Configured via `emb_dim` and `vocab_size` in configs
- For custom heads, modify the logit computation in `Transformer.__call__`
- Multi-token prediction (MTP) support via `multi_token_prediction.py`

**Key Modification Points:**
1. **Custom LM Head**: Modify `Transformer.setup()` to add a separate head module instead of using shared embeddings
2. **Output Processing**: Adjust logit computation in `Transformer.__call__` around line 150-160
3. **Loss Functions**: Modify loss computation in `train.py:train_step()` function

### Configuration System

All configurations use YAML files in `MaxText/configs/`:
- `base.yml`: Default configuration template
- `sft.yml`: Supervised fine-tuning settings
- `models/`: Pre-configured model architectures
- Custom configs override base settings

**Important Config Parameters for Customization:**
```yaml
# Data loading
dataset_type: "hf"  # or "grain", "tfds", "synthetic"
dataset_path: "path/to/dataset"
tokenizer_path: "path/to/tokenizer"
max_target_length: 2048
global_batch_size: 128

# Model architecture
emb_dim: 768  # Hidden dimension
num_heads: 12
num_layers: 12
vocab_size: 32000  # Vocabulary size (affects LM head)

# Training
learning_rate: 1e-4
steps: 100000
use_sft: false  # Set true for fine-tuning
```

## Critical Implementation Details

### Sharding and Parallelism
- Data parallelism: `dcn_data_parallelism`
- Tensor parallelism: `dcn_tensor_parallelism`
- FSDP: `dcn_fsdp_parallelism`
- Pipeline parallelism: `dcn_pipeline_parallelism`
- Configured via mesh dimensions in configs

### Checkpointing
- Checkpoints stored in `base_output_directory/run_name/checkpoints/`
- Automatic resumption if checkpoint exists for `run_name`
- Parameter-only checkpoints: `load_parameters_path`
- Full state checkpoints: `load_full_state_path`

### Key Files for Model Training Flow
1. `train.py`: Main training loop and step functions
2. `sft_trainer.py`: Supervised fine-tuning implementation
3. `train_utils.py`: Training utilities and JIT compilation
4. `data_loader.py`: Data loading orchestration
5. `layers/models.py`: Model definition and forward pass

### Common Patterns for Extensions

**Adding a New Data Source:**
1. Create `MaxText/input_pipeline/_custom_data_processing.py`
2. Implement `make_custom_train_iterator()` and `make_custom_eval_iterator()`
3. Add to `input_pipeline_interface.py` router
4. Set `dataset_type: "custom"` in config

**Modifying the LM Head:**
1. In `layers/models.py`, add a new head module in `Transformer.setup()`
2. Replace shared embedding usage in forward pass
3. Update loss computation if needed
4. Adjust `vocab_size` and `emb_dim` configs

**Custom Loss Functions:**
1. Modify `train.py:train_step()` function
2. Add loss computation after model forward pass
3. Update metrics in `train_loop()`

## Performance Optimization Tips

- Use `reuse_example_batch: true` for debugging
- Enable `async_checkpointing: true` for better throughput
- Adjust `per_device_batch_size` based on memory
- Use `scan_layers: true` for memory efficiency with many layers
- Enable `enable_profiler: true` for performance analysis

## Debugging Workflows

```bash
# Small-scale smoke test
python3 -m MaxText.train MaxText/configs/tpu_smoke_test.yml run_name=debug_test steps=10

# Profile performance
python3 -m MaxText.train MaxText/configs/base.yml run_name=profile_test enable_profiler=true skip_first_n_steps_for_profiler=10 profiler_steps=5

# Test data loading only
python3 -m MaxText.standalone_dataloader MaxText/configs/base.yml
```

## Model-Specific Considerations

- **Llama Models**: Use RMSNorm, RoPE embeddings, SwiGLU activation
- **Gemma Models**: Use GeGLU activation, different normalization
- **Mixtral**: Sparse MoE implementation in `layers/moe.py`
- **DeepSeek**: Multi-head latent attention (MLA) in `layers/deepseek.py`

Each model family has its own layer implementation file with specific architectural details.