"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Input pipeline for Emilia audio dataset with external tokenization."""

import ml_collections
import jax
import datasets
import grain.python as grain
import numpy as np
import json
from typing import Callable, Optional, List, Dict, Any

from MaxText.input_pipeline import _input_pipeline_utils
from MaxText import multihost_dataloading
from MaxText import max_logging


class AudioTokenMapper:
    """Maps audio tokens to vocabulary indices with index 262144 adjustment."""
    
    def __init__(self, mapping_path: str):
        """Load the adjusted audio token mapping."""
        with open(mapping_path, 'r') as f:
            mapping_data = json.load(f)
        
        # Use the audio_to_embedding mapping for converting audio IDs to embedding indices
        self.audio_to_embedding = {
            int(k): v for k, v in mapping_data.get('audio_to_embedding', {}).items()
        }
        
        # Get vocabulary size info
        self.vocab_size = mapping_data.get('stats', {}).get('adjusted_vocab_size', 264191)
        
        max_logging.log(f"Loaded audio token mapping with {len(self.audio_to_embedding)} audio tokens")
        max_logging.log(f"Adjusted vocabulary size: {self.vocab_size}")
    
    def map_audio_tokens(self, audio_ids: List[List[int]]) -> List[List[int]]:
        """
        Map audio token IDs to embedding indices.
        
        Args:
            audio_ids: List of sequences, each containing audio token IDs (0-8191)
        
        Returns:
            List of sequences with mapped embedding indices
        """
        mapped_sequences = []
        
        for sequence in audio_ids:
            mapped_seq = []
            for audio_id in sequence:
                if audio_id in self.audio_to_embedding:
                    mapped_seq.append(self.audio_to_embedding[audio_id])
                else:
                    # If audio ID not found, use a default padding token
                    # This should rarely happen if the mapping is complete
                    max_logging.log(f"Warning: Audio ID {audio_id} not found in mapping")
                    mapped_seq.append(0)  # Default to padding token
            mapped_sequences.append(mapped_seq)
        
        return mapped_sequences


class EmiliaAudioDataSource(grain.RandomAccessDataSource):
    """Data source for Emilia audio dataset with external tokenization."""
    
    def __init__(
        self,
        dataset: datasets.IterableDataset,
        audio_tokenizer: Callable,
        token_mapper: AudioTokenMapper,
        dataloading_host_index: int,
        dataloading_host_count: int,
        num_threads: int,
        max_target_length: int,
        audio_batch_size: int,
        generate_padding_example: bool,
    ):
        self.dataset = dataset
        self.audio_tokenizer = audio_tokenizer
        self.token_mapper = token_mapper
        self.dataloading_host_index = dataloading_host_index
        self.dataloading_host_count = dataloading_host_count
        self.num_threads = num_threads
        self.max_target_length = max_target_length
        self.audio_batch_size = audio_batch_size
        self.generate_padding_example = generate_padding_example
        
        # Setup dataset sharding
        if hasattr(dataset, "n_shards"):
            self.n_shards = dataset.n_shards
        else:
            self.n_shards = 1
        
        self._check_shard_count()
        self.dataset_shards = [dataloading_host_index * self.num_threads + i for i in range(self.num_threads)]
        self.datasets = [
            _input_pipeline_utils.split_dataset_by_node(dataset, world_size=self.n_shards, rank=x) 
            for x in self.dataset_shards
        ]
        self.data_iters = []
        self.out_of_data = False
        self.audio_buffer = []  # Buffer for batch processing
    
    def _check_shard_count(self):
        """Check if shard count is efficient for multihost loading."""
        if self.n_shards < (self.dataloading_host_count * self.num_threads):
            max_logging.log(
                f"WARNING: Inefficient dataloading. Dataset contains {self.n_shards} shards, "
                "smaller than number of hosts loading data."
            )
            self.n_shards = self.dataloading_host_count * self.num_threads
    
    def __len__(self):
        """Return a large number as HuggingFace IterableDataset doesn't have real length."""
        return 10_000_000_000
    
    def _process_audio_batch(self, audio_batch: List[bytes]) -> List[List[int]]:
        """
        Process a batch of audio through external tokenizer and map tokens.
        
        Args:
            audio_batch: List of audio bytes from MP3 files
        
        Returns:
            List of token sequences mapped to vocabulary indices
        """
        # Call external tokenizer (provided by user)
        audio_token_ids = self.audio_tokenizer(audio_batch)
        
        # Map audio tokens to vocabulary indices
        mapped_tokens = self.token_mapper.map_audio_tokens(audio_token_ids)
        
        return mapped_tokens
    
    def __getitem__(self, index):
        """Get next item from the dataset."""
        if not self.data_iters:
            self.data_iters = [iter(x) for x in self.datasets]
        
        idx = 0  # Default to first iterator if not in threaded context
        try:
            from threading import current_thread
            idx = int(current_thread().name.split("_")[1])
        except:
            pass
        
        # Collect audio samples for batch processing
        while len(self.audio_buffer) < self.audio_batch_size:
            try:
                item = next(self.data_iters[idx])
                # Extract audio from the 'mp3' column
                if 'mp3' in item:
                    self.audio_buffer.append(item['mp3'])
            except StopIteration:
                if self.generate_padding_example and not self.audio_buffer:
                    # Generate padding example if needed
                    return {
                        "inputs": np.zeros(self.max_target_length, dtype=np.int32),
                        "targets": np.zeros(self.max_target_length, dtype=np.int32),
                    }
                break
        
        if self.audio_buffer:
            # Process the batch
            batch_to_process = self.audio_buffer[:self.audio_batch_size]
            self.audio_buffer = self.audio_buffer[self.audio_batch_size:]
            
            # Get tokenized and mapped sequences
            token_sequences = self._process_audio_batch(batch_to_process)
            
            # Return the first sequence from the batch
            if token_sequences:
                tokens = token_sequences[0]
                
                # Truncate or pad to max_target_length
                if len(tokens) > self.max_target_length:
                    tokens = tokens[:self.max_target_length]
                elif len(tokens) < self.max_target_length:
                    tokens = tokens + [0] * (self.max_target_length - len(tokens))
                
                tokens_array = np.array(tokens, dtype=np.int32)
                
                # For audio-only training, inputs and targets are the same
                return {
                    "inputs": tokens_array,
                    "targets": tokens_array,
                }
        
        # If no data available, return padding
        return {
            "inputs": np.zeros(self.max_target_length, dtype=np.int32),
            "targets": np.zeros(self.max_target_length, dtype=np.int32),
        }


def preprocessing_pipeline(
    dataloading_host_index,
    dataloading_host_count,
    global_mesh,
    dataset,
    audio_tokenizer,
    token_mapper,
    global_batch_size,
    max_target_length,
    audio_batch_size,
    shuffle,
    data_shuffle_seed,
    num_threads=1,
    drop_remainder=False,
    generate_padding_example=False,
):
    """Pipeline for preprocessing Emilia audio dataset."""
    
    assert global_batch_size % global_mesh.size == 0, "Batch size should be divisible by number of global devices."
    
    if shuffle:
        dataset = dataset.shuffle(seed=data_shuffle_seed)
    
    # Create data source with audio processing
    dataset = EmiliaAudioDataSource(
        dataset=dataset,
        audio_tokenizer=audio_tokenizer,
        token_mapper=token_mapper,
        dataloading_host_index=dataloading_host_index,
        dataloading_host_count=dataloading_host_count,
        num_threads=num_threads,
        max_target_length=max_target_length,
        audio_batch_size=audio_batch_size,
        generate_padding_example=generate_padding_example,
    )
    
    # Set up operations
    operations = []
    
    # Batch the data
    operations.append(
        grain.Batch(batch_size=global_batch_size // jax.process_count(), drop_remainder=drop_remainder)
    )
    
    # Shift for autoregressive training
    operations.append(_input_pipeline_utils.ShiftData(ignored_ids=[0], axis=1))
    
    # Create dummy index sampler (required for grain.DataLoader)
    dummy_index_sampler = grain.IndexSampler(
        num_records=len(dataset),
        num_epochs=1,
        shard_options=grain.ShardOptions(
            shard_index=dataloading_host_index, 
            shard_count=dataloading_host_count, 
            drop_remainder=False
        ),
        shuffle=False,
        seed=0,
    )
    
    # Create dataloader
    dataloader = grain.DataLoader(
        data_source=dataset,
        operations=operations,
        sampler=dummy_index_sampler,
        worker_count=0,  # No additional workers for now
        worker_buffer_size=1,
        read_options=grain.ReadOptions(num_threads=num_threads, prefetch_buffer_size=128),
    )
    
    # Create multihost iterator
    multihost_gen = multihost_dataloading.MultiHostDataLoadIterator(dataloader, global_mesh)
    
    return multihost_gen


def make_emilia_audio_train_iterator(
    config: ml_collections.ConfigDict,
    global_mesh,
    process_indices_train,
):
    """Load and preprocess Emilia audio dataset for training."""
    
    # Load Emilia dataset with streaming
    language = config.get('emilia_language', 'EN')
    if language != 'ALL':
        data_files = {language: f"Emilia/{language}/*.tar"}
        train_ds = datasets.load_dataset(
            "amphion/Emilia-Dataset",
            data_files=data_files,
            split=language,
            streaming=True,
        )
    else:
        train_ds = datasets.load_dataset(
            "amphion/Emilia-Dataset",
            split="train",
            streaming=True,
        )
    
    # Create token mapper
    token_mapper = AudioTokenMapper(config.audio_token_mapping_path)
    
    # Get audio tokenizer function from config
    # This should be provided by the user
    audio_tokenizer = config.get('audio_tokenizer_fn')
    if audio_tokenizer is None:
        raise ValueError("audio_tokenizer_fn must be provided in config for Emilia audio processing")
    
    # Create iterator
    train_iter = preprocessing_pipeline(
        dataloading_host_index=process_indices_train.index(jax.process_index()),
        dataloading_host_count=len(process_indices_train),
        global_mesh=global_mesh,
        dataset=train_ds,
        audio_tokenizer=audio_tokenizer,
        token_mapper=token_mapper,
        global_batch_size=config.global_batch_size_to_load,
        max_target_length=config.max_target_length,
        audio_batch_size=config.get('audio_batch_size', 32),
        shuffle=config.enable_data_shuffling,
        data_shuffle_seed=config.data_shuffle_seed,
        generate_padding_example=False,
    )
    
    return train_iter


def make_emilia_audio_eval_iterator(
    config: ml_collections.ConfigDict,
    global_mesh,
    process_indices_eval,
):
    """Load and preprocess Emilia audio dataset for evaluation."""
    
    # Load Emilia dataset
    language = config.get('emilia_language', 'EN')
    if language != 'ALL':
        data_files = {language: f"Emilia/{language}/*.tar"}
        eval_ds = datasets.load_dataset(
            "amphion/Emilia-Dataset",
            data_files=data_files,
            split=language,
            streaming=True,
        )
    else:
        eval_ds = datasets.load_dataset(
            "amphion/Emilia-Dataset",
            split="train",  # Emilia only has train split
            streaming=True,
        )
    
    # Create token mapper
    token_mapper = AudioTokenMapper(config.audio_token_mapping_path)
    
    # Get audio tokenizer function
    audio_tokenizer = config.get('audio_tokenizer_fn')
    if audio_tokenizer is None:
        raise ValueError("audio_tokenizer_fn must be provided in config for Emilia audio processing")
    
    eval_generate_padding_example = config.eval_steps > 0
    
    # Create iterator
    eval_iter = preprocessing_pipeline(
        dataloading_host_index=process_indices_eval.index(jax.process_index()),
        dataloading_host_count=len(process_indices_eval),
        global_mesh=global_mesh,
        dataset=eval_ds,
        audio_tokenizer=audio_tokenizer,
        token_mapper=token_mapper,
        global_batch_size=config.global_batch_size_to_load_eval,
        max_target_length=config.max_target_length,
        audio_batch_size=config.get('audio_batch_size', 32),
        shuffle=False,  # No shuffling for eval
        data_shuffle_seed=config.data_shuffle_seed,
        generate_padding_example=eval_generate_padding_example,
    )
    
    return eval_iter