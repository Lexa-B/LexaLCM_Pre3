# SONAR Encoder

from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
import torch
import argparse
import os

class ConceptEncoder:
    """
    Concept Encoder for encoding text into embeddings. Based on SONAR.
    """
    # Initialize Concept Encoder
    def __init__(self, Language, Verbose, Sequential):
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize SONAR for Japanese sentence processing
        self.SONAR_Text2Vec = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder",
            tokenizer="text_sonar_basic_encoder",
            device=self.device,
            dtype=torch.float32)
        
        self.Args = {
            "language": Language,
            "verbose": Verbose,
            "sequential": Sequential
        }
    
    def Encode(self, sentences: list[str]) -> list[torch.Tensor]:
        # Assume sentences is always a list of sentences, even though it might have a length of 1

        if self.Args["verbose"]:
            print(f"Embedding Sentences: {sentences}")

        embeddings = self.SONAR_Text2Vec.predict(sentences, source_lang=self.Args["language"])
        if self.Args["verbose"]:
            print(f"embeddings.shape: {embeddings.shape}, dtype: {embeddings.dtype}")

        return embeddings

    def EncodeSentence(self, sentence: str) -> str:
        # Translate a single sentence
        return self.Encode([sentence])[0]

    def EncodeBatch(self, sentences: list[str]) -> list[str]:
        # Translate a batch of sentences
        return self.Encode(sentences)
    
    def EncodeText(self, text: str) -> list[torch.Tensor]:
        # Encode each sentence
        sentences = [text.strip()]
        
        if self.Args["sequential"]: # Sequential Translation
            EncodedSentences = []
            for Sentence in sentences:
                Sentence = self.EncodeSentence(Sentence).to(dtype=torch.float32)
                EncodedSentences.append(Sentence)
                print(f"Encoded: {Sentence.shape}, dtype: {Sentence.dtype}")
        else: # Batch Translation
            EncodedSentences = self.EncodeBatch(sentences)
            for i, Sentence in enumerate(sentences):
                print(f"Encoded: {sentences[i]} → {EncodedSentences[i].shape}, dtype: {EncodedSentences[i].dtype}")

        return EncodedSentences

    def SaveEmbedding(self, embedding: torch.Tensor, output_file: str):
        """Save the embedding tensor to a file"""
        # Get the directory path
        dir_path = os.path.dirname(output_file)
        # Only create directory if there is a path
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        # Ensure we have a batch dimension
        if len(embedding.shape) == 1:
            embedding = embedding.unsqueeze(0)  # Add batch dimension
        # Save the tensor
        torch.save(embedding, output_file)
        if self.Args["verbose"]:
            print(f"Saved embedding to {output_file}, shape: {embedding.shape}")

# Example usage
if __name__ == "__main__":
    # Get the command line arguments
    parser = argparse.ArgumentParser(description='Translate text using SONAR.')
    parser.add_argument('-i', '--input', type=str, required=True, help='Input text to encode.')
    parser.add_argument('-l', '--lang', type=str, required=True, help='Language (FLORES-200) (e.g., jpn_Jpan, eng_Latn).')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output.')
    parser.add_argument('-q', '--sequential', action='store_true', help='Process sentences sequentially.')
    parser.add_argument('-o', '--output', type=str, default='embeddings.pt', help='Output file for the embedding.')
    args = parser.parse_args()

    print(f"Input: {args.input}")
    
    if args.verbose:
        print("Verbose mode enabled.")
        print(f"Language: {args.lang}")
        print(f"Sequential: {args.sequential}")

    translator = ConceptEncoder(Language=args.lang, Verbose=args.verbose, Sequential=args.sequential)
    embeddings = translator.EncodeText(args.input)

    if args.verbose:
        print(f"Generated {len(embeddings)} embeddings")

    # Save the first embedding (since we're only encoding one sentence)
    translator.SaveEmbedding(embeddings[0], args.output)

    print(f"Language: {args.lang}, Verbose: {args.verbose}, Sequential: {args.sequential}")
