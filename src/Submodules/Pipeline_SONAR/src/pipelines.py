import torch
from typing import Union, List, Dict, Any
from dataclasses import dataclass
import logging
import subprocess
import os
import json
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('sonar_pipeline')

def get_venv_python():
    """Get the path to the Python interpreter in this module's virtual environment"""
    venv_path = os.path.join(os.path.dirname(__file__), "..", ".venv")
    
    if os.name == 'nt':  # Windows
        python_path = os.path.join(venv_path, "Scripts", "python.exe")
    else:  # Unix-like
        python_path = os.path.join(venv_path, "bin", "python")
    
    if not os.path.exists(python_path):
        raise FileNotFoundError(f"Could not find Python interpreter at {python_path}")
    
    return python_path

def run_in_venv(code: str, **kwargs) -> Any:
    """Run code in the virtual environment and return the result"""
    # Create a temporary script
    temp_script = "temp_sonar.py"
    with open(temp_script, "w") as f:
        f.write(f"""
import sys
import os
import json
import torch
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline, EmbeddingToTextModelPipeline

# Get the arguments
kwargs = {repr(kwargs)}

# Run the code
{code}

# Convert tensor to list for JSON serialization
if isinstance(result, torch.Tensor):
    result = result.tolist()

# Print the result as JSON
print(json.dumps({{"result": result}}))
""")
    
    try:
        # Run the temporary script using the module's Python interpreter
        cmd = [get_venv_python(), temp_script]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Error running in venv: {result.stderr}")
        
        # Parse the JSON output
        output = json.loads(result.stdout)
        return output["result"]
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_script):
            os.remove(temp_script)

@dataclass
class PipelineConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    language: str = "eng_Latn"
    verbose: bool = False
    sequential: bool = False

class BasePipeline:
    def __init__(
        self,
        device: str = None,
        dtype: torch.dtype = None,
        language: str = "eng_Latn",
        verbose: bool = False,
        sequential: bool = False,
    ):
        self.config = PipelineConfig(
            device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
            dtype=dtype or torch.float32,
            language=language,
            verbose=verbose,
            sequential=sequential
        )
        self.device = torch.device(self.config.device)
        
        if self.config.verbose:
            logger.info(f"Initializing pipeline with config: {self.config}")

class TextToEmbeddingPipeline(BasePipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.config.verbose:
            logger.info("Initialized TextToEmbeddingPipeline")
    
    def __call__(
        self,
        inputs: Union[str, List[str]],
        **kwargs
    ) -> torch.Tensor:
        if isinstance(inputs, str):
            inputs = [inputs]
        
        if self.config.verbose:
            logger.info(f"Encoding sentences: {inputs}")
        
        # Run the encoding in the virtual environment
        code = f"""
device = torch.device("{self.device}")
pipeline = TextToEmbeddingModelPipeline(
    encoder="text_sonar_basic_encoder",
    tokenizer="text_sonar_basic_encoder",
    device=device,
    dtype=torch.float32
)
# Pass text directly to predict
result = pipeline.predict({repr(inputs)}, source_lang="{self.config.language}")
"""
        embeddings = run_in_venv(code, inputs=inputs)
        
        if self.config.verbose:
            logger.info(f"Generated embeddings with shape: {len(embeddings)}, dtype: {type(embeddings)}")
        
        # Convert list back to tensor and ensure correct dtype
        embeddings_tensor = torch.tensor(embeddings[0] if len(inputs) == 1 else embeddings, dtype=torch.float32)
        return embeddings_tensor

class EmbeddingToTextPipeline(BasePipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.config.verbose:
            logger.info("Initialized EmbeddingToTextPipeline")
    
    def __call__(
        self,
        inputs: torch.Tensor,
        **kwargs
    ) -> Union[str, List[str]]:
        if not isinstance(inputs, torch.Tensor):
            raise ValueError("Inputs must be a tensor")
            
        if len(inputs.shape) == 1:
            inputs = inputs.unsqueeze(0)
        
        # Ensure correct dtype
        inputs = inputs.to(dtype=torch.float32)
        
        if self.config.verbose:
            logger.info(f"Decoding embedding with shape: {inputs.shape}, dtype: {inputs.dtype}")
        
        # Convert tensor to list for JSON serialization
        embedding = inputs.tolist()
        
        # Run the decoding in the virtual environment
        code = f"""
device = torch.device("{self.device}")
pipeline = EmbeddingToTextModelPipeline(
    decoder="text_sonar_basic_decoder",
    tokenizer="text_sonar_basic_encoder",
    device=device,
    dtype=torch.float32
)
# Convert embedding to tensor with correct dtype
embedding = torch.tensor({repr(embedding)}, dtype=torch.float32)
result = pipeline.predict(embedding, target_lang="{self.config.language}")
"""
        decoded = run_in_venv(code, embedding=embedding)
        
        if self.config.verbose:
            logger.info(f"Decoded text: {decoded}")
        
        return decoded[0] if len(decoded) == 1 else decoded 