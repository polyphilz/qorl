"""Local vLLM server settings; context length comes from the model settings."""

from pydantic import BaseModel, ConfigDict, Field

MAX_PORT = 65_535


class ServingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    host: str = Field(min_length=1)
    port: int = Field(gt=0, le=MAX_PORT)
    dtype: str = Field(min_length=1)
    tool_call_parser: str = Field(min_length=1)
    max_num_seqs: int = Field(gt=0)
    gpu_memory_utilization: float = Field(gt=0, le=1)
    enable_prefix_caching: bool
    use_flashinfer_sampler: bool
    startup_timeout_seconds: int = Field(gt=0)
    request_timeout_seconds: int = Field(gt=0)
