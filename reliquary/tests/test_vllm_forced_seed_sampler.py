"""Unit tests for vLLM forced seed logits processor.

Run with: pytest reliquary/tests/test_vllm_forced_seed_sampler.py
"""
import pytest
import torch

from reliquary.constants import T_PROTO, TOP_K_PROTO, TOP_P_PROTO
from reliquary.environment.forced_sampling import pick, u_at, warp
from reliquary.miner.vllm_forced_seed_sampler import (
    ForcedSeedVllmLogitsProcessor,
    forced_seed_vllm_sampling_params,
)

# Mock vLLM types for testing
from unittest.mock import MagicMock
from dataclasses import dataclass
from typing import Optional


@dataclass
class MockSamplingParams:
    """Mock SamplingParams for testing."""
    temperature: float = 1.0
    max_tokens: int = 100
    extra_args: Optional[dict] = None


@dataclass  
class MockBatchUpdate:
    """Mock BatchUpdate for testing."""
    added: list = None
    removed: list = None
    moved: list = None
    batch_size: int = 0
    
    def __post_init__(self):
        if self.added is None:
            self.added = []
        if self.removed is None:
            self.removed = []
        if self.moved is None:
            self.moved = []


class TestForcedSeedVllmLogitsProcessor:
    """Test suite for ForcedSeedVllmLogitsProcessor."""
    
    @pytest.fixture
    def device(self):
        """Provide CUDA device if available, else CPU."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture
    def processor(self, device):
        """Create a processor instance."""
        mock_config = MagicMock()
        return ForcedSeedVllmLogitsProcessor(
            vllm_config=mock_config,
            device=device,
            is_pin_memory=False,
        )
    
    @pytest.fixture
    def sample_config(self):
        """Provide sample forced seed configuration."""
        return {
            "randomness": "test-randomness-123",
            "prompt_idx": 42,
            "checkpoint_hash": "abc123def456",
            "rollout_index": 0,
            "base_offset": 0,
            "temperature": T_PROTO,
            "top_k": TOP_K_PROTO,
            "top_p": TOP_P_PROTO,
        }
    
    def test_validate_params_valid(self, sample_config):
        """Test that valid config passes validation."""
        params = MockSamplingParams(
            extra_args={"forced_seed_config": sample_config}
        )
        # Should not raise
        ForcedSeedVllmLogitsProcessor.validate_params(params)
    
    def test_validate_params_no_config(self):
        """Test that missing config is allowed (disables processor)."""
        params = MockSamplingParams()
        # Should not raise
        ForcedSeedVllmLogitsProcessor.validate_params(params)
    
    def test_validate_params_missing_field(self, sample_config):
        """Test that missing required field raises ValueError."""
        incomplete_config = sample_config.copy()
        del incomplete_config["randomness"]
        
        params = MockSamplingParams(
            extra_args={"forced_seed_config": incomplete_config}
        )
        
        with pytest.raises(ValueError, match="missing required field"):
            ForcedSeedVllmLogitsProcessor.validate_params(params)
    
    def test_validate_params_wrong_type(self, sample_config):
        """Test that wrong field type raises ValueError."""
        bad_config = sample_config.copy()
        bad_config["prompt_idx"] = "not-an-int"
        
        params = MockSamplingParams(
            extra_args={"forced_seed_config": bad_config}
        )
        
        with pytest.raises(ValueError, match="must be int"):
            ForcedSeedVllmLogitsProcessor.validate_params(params)
    
    def test_is_argmax_invariant(self, processor):
        """Test that processor correctly reports non-invariance."""
        assert processor.is_argmax_invariant() is False
    
    def test_update_state_add_request(self, processor, sample_config):
        """Test adding a request with forced seed config."""
        params = MockSamplingParams(
            extra_args={"forced_seed_config": sample_config}
        )
        
        batch_update = MockBatchUpdate(
            added=[(0, params, None, None)],
            batch_size=1,
        )
        
        processor.update_state(batch_update)
        
        assert 0 in processor.req_configs
        assert processor.req_configs[0]["randomness"] == "test-randomness-123"
        assert processor.req_configs[0]["prompt_idx"] == 42
        assert 0 in processor.req_step_counts
        assert processor.req_step_counts[0] == 0
    
    def test_update_state_add_without_config(self, processor):
        """Test adding a request without forced seed config."""
        params = MockSamplingParams()  # No extra_args
        
        batch_update = MockBatchUpdate(
            added=[(0, params, None, None)],
            batch_size=1,
        )
        
        processor.update_state(batch_update)
        
        assert 0 not in processor.req_configs
        assert 0 not in processor.req_step_counts
    
    def test_update_state_remove_request(self, processor, sample_config):
        """Test removing a request."""
        # First add
        params = MockSamplingParams(
            extra_args={"forced_seed_config": sample_config}
        )
        batch_update_add = MockBatchUpdate(
            added=[(0, params, None, None)],
        )
        processor.update_state(batch_update_add)
        
        # Then remove
        batch_update_remove = MockBatchUpdate(
            removed=[0],
        )
        processor.update_state(batch_update_remove)
        
        assert 0 not in processor.req_configs
        assert 0 not in processor.req_step_counts
    
    def test_update_state_unidirectional_move(self, processor, sample_config):
        """Test unidirectional move operation."""
        from reliquary.miner.vllm_forced_seed_sampler import MoveDirectionality
        
        # Add request at index 0
        params = MockSamplingParams(
            extra_args={"forced_seed_config": sample_config}
        )
        batch_update_add = MockBatchUpdate(
            added=[(0, params, None, None)],
        )
        processor.update_state(batch_update_add)
        processor.req_step_counts[0] = 5  # Simulate some steps
        
        # Move from 0 to 2 (unidirectional)
        batch_update_move = MockBatchUpdate(
            moved=[(0, 2, MoveDirectionality.UNIDIRECTIONAL)],
        )
        processor.update_state(batch_update_move)
        
        assert 0 not in processor.req_configs
        assert 2 in processor.req_configs
        assert processor.req_step_counts[2] == 5
    
    def test_update_state_swap_move(self, processor, sample_config):
        """Test swap move operation."""
        from reliquary.miner.vllm_forced_seed_sampler import MoveDirectionality
        
        # Add two requests
        params1 = MockSamplingParams(
            extra_args={"forced_seed_config": sample_config}
        )
        config2 = sample_config.copy()
        config2["prompt_idx"] = 99
        params2 = MockSamplingParams(
            extra_args={"forced_seed_config": config2}
        )
        
        batch_update_add = MockBatchUpdate(
            added=[(0, params1, None, None), (1, params2, None, None)],
        )
        processor.update_state(batch_update_add)
        processor.req_step_counts[0] = 5
        processor.req_step_counts[1] = 10
        
        # Swap 0 and 1
        batch_update_swap = MockBatchUpdate(
            moved=[(0, 1, MoveDirectionality.SWAP)],
        )
        processor.update_state(batch_update_swap)
        
        # After swap: config for prompt_idx=42 should be at index 1
        # and config for prompt_idx=99 should be at index 0
        assert processor.req_configs[1]["prompt_idx"] == 42
        assert processor.req_configs[0]["prompt_idx"] == 99
        assert processor.req_step_counts[1] == 5
        assert processor.req_step_counts[0] == 10
    
    def test_apply_no_configs(self, processor, device):
        """Test apply when no requests have forced seed."""
        batch_size = 4
        vocab_size = 100
        logits = torch.randn(batch_size, vocab_size, device=device)
        logits_copy = logits.clone()
        
        result = processor.apply(logits)
        
        # Should return unchanged
        assert torch.allclose(result, logits_copy)
    
    def test_apply_single_request(self, processor, device, sample_config):
        """Test apply with single configured request."""
        # Setup: add request
        params = MockSamplingParams(
            extra_args={"forced_seed_config": sample_config}
        )
        batch_update = MockBatchUpdate(
            added=[(0, params, None, None)],
        )
        processor.update_state(batch_update)
        
        # Create logits
        batch_size = 1
        vocab_size = 100
        logits = torch.randn(batch_size, vocab_size, device=device)
        
        # Apply processor
        result = processor.apply(logits)
        
        # Verify: exactly one token should have 0.0 logit, rest -inf
        finite_mask = torch.isfinite(result[0])
        assert finite_mask.sum() == 1  # Only one token not -inf
        
        selected_idx = torch.where(finite_mask)[0].item()
        assert result[0, selected_idx] == 0.0
        
        # Verify step counter incremented
        assert processor.req_step_counts[0] == 1
    
    def test_apply_deterministic(self, processor, device, sample_config):
        """Test that same config produces same token selection."""
        # Setup two identical processors
        mock_config = MagicMock()
        processor1 = ForcedSeedVllmLogitsProcessor(
            vllm_config=mock_config,
            device=device,
            is_pin_memory=False,
        )
        processor2 = ForcedSeedVllmLogitsProcessor(
            vllm_config=mock_config,
            device=device,
            is_pin_memory=False,
        )
        
        # Add same config to both
        params = MockSamplingParams(
            extra_args={"forced_seed_config": sample_config}
        )
        batch_update = MockBatchUpdate(
            added=[(0, params, None, None)],
        )
        processor1.update_state(batch_update)
        processor2.update_state(batch_update)
        
        # Create same logits
        torch.manual_seed(42)
        logits1 = torch.randn(1, 100, device=device)
        torch.manual_seed(42)
        logits2 = torch.randn(1, 100, device=device)
        
        # Apply both
        result1 = processor1.apply(logits1)
        result2 = processor2.apply(logits2)
        
        # Should select same token
        token1 = torch.where(torch.isfinite(result1[0]))[0].item()
        token2 = torch.where(torch.isfinite(result2[0]))[0].item()
        assert token1 == token2
    
    def test_apply_multiple_requests(self, processor, device, sample_config):
        """Test apply with multiple configured requests in batch."""
        # Add two requests with different configs
        config1 = sample_config.copy()
        config1["rollout_index"] = 0
        config2 = sample_config.copy()
        config2["rollout_index"] = 1
        
        params1 = MockSamplingParams(
            extra_args={"forced_seed_config": config1}
        )
        params2 = MockSamplingParams(
            extra_args={"forced_seed_config": config2}
        )
        
        batch_update = MockBatchUpdate(
            added=[(0, params1, None, None), (1, params2, None, None)],
        )
        processor.update_state(batch_update)
        
        # Create logits
        batch_size = 2
        vocab_size = 100
        logits = torch.randn(batch_size, vocab_size, device=device)
        
        # Apply processor
        result = processor.apply(logits)
        
        # Verify both requests have forced tokens
        for i in range(2):
            finite_mask = torch.isfinite(result[i])
            assert finite_mask.sum() == 1
            assert processor.req_step_counts[i] == 1
    
    def test_apply_step_increment(self, processor, device, sample_config):
        """Test that step counter increments correctly across multiple applies."""
        # Setup
        params = MockSamplingParams(
            extra_args={"forced_seed_config": sample_config}
        )
        batch_update = MockBatchUpdate(
            added=[(0, params, None, None)],
        )
        processor.update_state(batch_update)
        
        # Apply multiple times
        vocab_size = 100
        for expected_step in range(5):
            assert processor.req_step_counts[0] == expected_step
            
            logits = torch.randn(1, vocab_size, device=device)
            processor.apply(logits)
        
        assert processor.req_step_counts[0] == 5


class TestForcedSeedVllmSamplingParams:
    """Test helper function for creating sampling params."""
    
    def test_basic_creation(self):
        """Test creating sampling params with required fields."""
        params = forced_seed_vllm_sampling_params(
            randomness="test-rand",
            prompt_idx=42,
            checkpoint_hash="abc123",
            rollout_index=0,
        )
        
        assert params.max_tokens == 512  # default
        assert params.extra_args is not None
        config = params.extra_args["forced_seed_config"]
        assert config["randomness"] == "test-rand"
        assert config["prompt_idx"] == 42
        assert config["checkpoint_hash"] == "abc123"
        assert config["rollout_index"] == 0
        assert config["base_offset"] == 0
        assert config["temperature"] == T_PROTO
        assert config["top_k"] == TOP_K_PROTO
        assert config["top_p"] == TOP_P_PROTO
    
    def test_custom_values(self):
        """Test creating sampling params with custom values."""
        params = forced_seed_vllm_sampling_params(
            randomness="test-rand",
            prompt_idx=42,
            checkpoint_hash="abc123",
            rollout_index=3,
            base_offset=100,
            temperature=0.5,
            top_k=10,
            top_p=0.9,
            max_tokens=256,
        )
        
        config = params.extra_args["forced_seed_config"]
        assert config["rollout_index"] == 3
        assert config["base_offset"] == 100
        assert config["temperature"] == 0.5
        assert config["top_k"] == 10
        assert config["top_p"] == 0.9
        assert params.max_tokens == 256
    
    def test_kwargs_override(self):
        """Test that additional kwargs are passed through."""
        params = forced_seed_vllm_sampling_params(
            randomness="test-rand",
            prompt_idx=42,
            checkpoint_hash="abc123",
            rollout_index=0,
            stop_token_ids=[100, 101],
            include_stop_str_in_output=True,
        )
        
        assert params.stop_token_ids == [100, 101]
        assert params.include_stop_str_in_output is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
