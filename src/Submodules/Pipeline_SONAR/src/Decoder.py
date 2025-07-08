# SONAR Decoder

from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline
import torch
import argparse

class ConceptDecoder:
    """
    Concept Decoder for decoding embeddings into text. Based on SONAR.
    """
    # Initialize Concept Decoder
    def __init__(self, Language, Verbose, Sequential):
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize SONAR for Japanese sentence processing
        self.SONAR_Vec2Text = EmbeddingToTextModelPipeline(
            decoder="text_sonar_basic_decoder",
            tokenizer="text_sonar_basic_encoder",
            device=self.device,
            dtype=torch.float32)
        
        self.Args = {
            "language": Language,
            "verbose": Verbose,
            "sequential": Sequential
        }
    
    def LoadEmbedding(self, input_file: str) -> torch.Tensor:
        """Load the embedding tensor from a file"""
        embedding = torch.load(input_file, map_location=self.device)
        # Ensure we have a batch dimension
        if len(embedding.shape) == 1:
            embedding = embedding.unsqueeze(0)  # Add batch dimension
        if self.Args["verbose"]:
            print(f"Loaded embedding from {input_file}")
            print(f"Embedding shape: {embedding.shape}, dtype: {embedding.dtype}")
        return embedding
    
    def Decode(self, embedding: torch.Tensor) -> str:
        # Assume sentences is always a list of sentences, even though it might have a length of 1

        if self.Args["verbose"]:
            print(f"Embedding to Decode: {embedding.shape}, dtype: {embedding.dtype}")

        DecodedSentence = self.SONAR_Vec2Text.predict(embedding, target_lang=self.Args["language"])
        if self.Args["verbose"]:
            print(f"Decoded Sentence: {DecodedSentence}")

        return DecodedSentence

# Example usage
if __name__ == "__main__":
    # Get the command line arguments
    parser = argparse.ArgumentParser(description='Translate text using SONAR.')
    parser.add_argument('-i', '--input', type=str, required=True, help='Input embedding file to decode.')
    parser.add_argument('-l', '--lang', type=str, required=True, help='Language (FLORES-200) (e.g., jpn_Jpan, eng_Latn).')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output.')
    parser.add_argument('-q', '--sequential', action='store_true', help='Process sentences sequentially.')
    args = parser.parse_args()

    print(f"Input file: {args.input}")
    
    if args.verbose:
        print("Verbose mode enabled.")
        print(f"Language: {args.lang}")
        print(f"Sequential: {args.sequential}")

    translator = ConceptDecoder(Language=args.lang, Verbose=args.verbose, Sequential=args.sequential)
    embedding = translator.LoadEmbedding(args.input)
    decoded_text = translator.Decode(embedding)

    if args.verbose:
        print(f"Decoded text: {decoded_text}")

    print(f"Language: {args.lang}, Verbose: {args.verbose}, Sequential: {args.sequential}")
