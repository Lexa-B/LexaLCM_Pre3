# LexaLCM-Pre3-x.xB Two-Tower Latent Diffusion Large Concept Model
これは、Meta FAIRのTwo-Tower Diffusion LCMアーキテクチャを主に基にした、xxxxxxxxxxxxxxxxx個のパラメータを持つ事前学習済みのLCMで、Hugging Face Transformersで実装されています。

[[Meta FAIRのLCMの研究論文（英語）]](https://ai.meta.com/research/publications/large-concept-models-language-modeling-in-a-sentence-representation-space/)

It is a pre-trained LCM with xxxxxxxxxxxxxxxxx parameters mostly based on Meta FAIR's Two-Tower Diffusion LCM architecture, but in Hugging FaceTransformers.

[[Meta FAIR's LCM Paper]](https://ai.meta.com/research/publications/large-concept-models-language-modeling-in-a-sentence-representation-space/)

最初のバージョンは、事前にセグメント化およびコンセプト埋め込みが行われた240万件の日本語および英語のWikipedia記事を用いて学習されています。セグメント化は、1セグメントあたり最大250文字に制限されたSaTを使用して行われ、埋め込みにはSONARが使用されました。
[[データセット]](https://huggingface.co/datasets/Lexa-B/LexaLCM_Datasets)

The first version was trained on a dataset of 2.4M Japanese Wikipedia articles that have been pre-segmented and concept-embedded. Segmentation was performed using SaT capped at 250 characters/segment and embedded was performed with SONAR.
[[Dataset]](https://huggingface.co/datasets/Lexa-B/LexaLCM_Datasets)

## インストール手順 ｜ Installation

```bash
uv venv # create a new virtual environment
source .venv/bin/activate # activate the virtual environment
uv pip install -e ".[gpu]" # install the dependencies (gpu)... if you want to install the dependencies (cpu), use ".[cpu]" instead
```

**Note: You'll need to download the model weights and dataset from the Hugging Face Hub.**
The model weights are located at:
[LexaLCM_Pre2](https://huggingface.co/Lexa-B/LexaLCM_Pre2)

The dataset is located at:
[LexaLCM_Datasets](https://huggingface.co/datasets/Lexa-B/LexaLCM_Datasets) (use the Pre2 version if you specifically want to use the same dataset as the model was trained on)

## モデル推論する ｜ Inference
**Note: you'll need to have an NVIDIA GPU with a decent amount of VRAM to run this as is... I haven't implemented quantization at inference features yet, so the LCM still infers in mixed precision fp32 and bf16, coupled with SONAR's 500M fp32 weights, meaning it's a bit of a memory hog, but still runnable on larger consumer GPUs. I've only tested this version on an RTX 4070ti, which has 12GB of VRAM.**
```bash
clear & uv run Tests/TestInference.py
```

Currently, it's not very smart, but it's a good start. Expect to see something like this:

```txt
2025-06-11 10:33:45,024 - sonar_pipeline - INFO - Initializing pipeline with config: PipelineConfig(device='cuda', dtype=torch.float32, language='eng_Latn', verbose=True, sequential=False)
2025-06-11 10:33:45,024 - sonar_pipeline - INFO - Initialized TextToEmbeddingPipeline
2025-06-11 10:33:45,024 - sonar_pipeline - INFO - Initializing pipeline with config: PipelineConfig(device='cuda', dtype=torch.float32, language='eng_Latn', verbose=True, sequential=False)
2025-06-11 10:33:45,024 - sonar_pipeline - INFO - Initialized EmbeddingToTextPipeline
2025-06-11 10:33:45,024 - sonar_pipeline - INFO - Encoding sentences: ['[[Start of Text.]]']
2025-06-11 10:33:49,761 - sonar_pipeline - INFO - Generated embeddings with shape: 1, dtype: <class 'list'>
2025-06-11 10:33:49,762 - sonar_pipeline - INFO - Encoding sentences: ["Japan's long history is divided into many distinct periods, each contributing to the country and culture in their own way."]
2025-06-11 10:33:54,516 - sonar_pipeline - INFO - Generated embeddings with shape: 1, dtype: <class 'list'>
2025-06-11 10:33:54,517 - sonar_pipeline - INFO - Encoding sentences: ['The Sengoku era was a period of great conflict in Japan.']
2025-06-11 10:33:59,197 - sonar_pipeline - INFO - Generated embeddings with shape: 1, dtype: <class 'list'>
2025-06-11 10:33:59,197 - sonar_pipeline - INFO - Encoding sentences: ['Many clans and their samurai from all over Japan fought in that time.']
2025-06-11 10:34:03,938 - sonar_pipeline - INFO - Generated embeddings with shape: 1, dtype: <class 'list'>
2025-06-11 10:34:03,938 - sonar_pipeline - INFO - Encoding sentences: ['The fighting lasted for many decades, but it was ultimately brought to an end in the unification of Japan.']
2025-06-11 10:34:08,648 - sonar_pipeline - INFO - Generated embeddings with shape: 1, dtype: <class 'list'>
2025-06-11 10:34:08,648 - sonar_pipeline - INFO - Encoding sentences: ['It was followed by a period of peace and cultural growth.']
2025-06-11 10:34:13,346 - sonar_pipeline - INFO - Generated embeddings with shape: 1, dtype: <class 'list'>
2025-06-11 10:34:13,346 - sonar_pipeline - INFO - Encoding sentences: ['This period was known as the Edo period.']
2025-06-11 10:34:18,004 - sonar_pipeline - INFO - Generated embeddings with shape: 1, dtype: <class 'list'>
2025-06-11 10:34:18,005 - sonar_pipeline - INFO - Encoding sentences: ['This period brought forward many new forms of art and culture.']
2025-06-11 10:34:22,635 - sonar_pipeline - INFO - Generated embeddings with shape: 1, dtype: <class 'list'>
2025-06-11 10:34:22,635 - sonar_pipeline - INFO - Encoding sentences: ['These include forms such as ukiyo-e woodblock paintings, kabuki theater, and haiku poetry.']
2025-06-11 10:34:27,380 - sonar_pipeline - INFO - Generated embeddings with shape: 1, dtype: <class 'list'>
2025-06-11 10:34:27,380 - sonar_pipeline - INFO - Encoding sentences: ['The impacts of the Edo period are large and still felt in the present day cultural landscape.']
2025-06-11 10:34:32,156 - sonar_pipeline - INFO - Generated embeddings with shape: 1, dtype: <class 'list'>
→ Context shape: torch.Size([1, 10, 1024]), dtype: torch.float32
[DEBUG - model] labels is None, likely being used for inference. Returning predictied embeddings - shape=torch.Size([1, 10, 1024]), dtype=torch.float32
2025-06-11 10:34:32,892 - sonar_pipeline - INFO - Decoding embedding with shape: torch.Size([1, 1024]), dtype: torch.float32
2025-06-11 10:34:38,902 - sonar_pipeline - INFO - Decoded text: ['He is the founder, chairman, and chief executive officer (CEO) of the company. He is also the founder and chairman of the company.']
Step 0 model next-token guess: He is the founder, chairman, and chief executive officer (CEO) of the company. He is also the founder and chairman of the company.
2025-06-11 10:34:38,902 - sonar_pipeline - INFO - Decoding embedding with shape: torch.Size([1, 1024]), dtype: torch.float32
2025-06-11 10:34:44,496 - sonar_pipeline - INFO - Decoded text: ['It\'s called the "City of Dreams".']
Step 1 model next-token guess: It's called the "City of Dreams".
2025-06-11 10:34:44,496 - sonar_pipeline - INFO - Decoding embedding with shape: torch.Size([1, 1024]), dtype: torch.float32
2025-06-11 10:34:50,317 - sonar_pipeline - INFO - Decoded text: ['The history of Japan, the history of China, the history of Japan, the history of Japan, the history of Japan, the history of Japan.']
Step 2 model next-token guess: The history of Japan, the history of China, the history of Japan, the history of Japan, the history of Japan, the history of Japan.
2025-06-11 10:34:50,317 - sonar_pipeline - INFO - Decoding embedding with shape: torch.Size([1, 1024]), dtype: torch.float32
2025-06-11 10:34:56,090 - sonar_pipeline - INFO - Decoded text: ['The history of the Japanese empire can be traced back to the period of the Qing Dynasty.']
Step 3 model next-token guess: The history of the Japanese empire can be traced back to the period of the Qing Dynasty.
2025-06-11 10:34:56,090 - sonar_pipeline - INFO - Decoding embedding with shape: torch.Size([1, 1024]), dtype: torch.float32
2025-06-11 10:35:01,780 - sonar_pipeline - INFO - Decoded text: ['The history of the state of Hokkaido dates back to the first century.']
Step 4 model next-token guess: The history of the state of Hokkaido dates back to the first century.
2025-06-11 10:35:01,780 - sonar_pipeline - INFO - Decoding embedding with shape: torch.Size([1, 1024]), dtype: torch.float32
2025-06-11 10:35:07,581 - sonar_pipeline - INFO - Decoded text: ['The history of Japan, the history of Japan, the history of Japan, the history of Japan.']
Step 5 model next-token guess: The history of Japan, the history of Japan, the history of Japan, the history of Japan.
2025-06-11 10:35:07,581 - sonar_pipeline - INFO - Decoding embedding with shape: torch.Size([1, 1024]), dtype: torch.float32
2025-06-11 10:35:13,248 - sonar_pipeline - INFO - Decoded text: ['It was a time of great turmoil in the history of mankind.']
Step 6 model next-token guess: It was a time of great turmoil in the history of mankind.
2025-06-11 10:35:13,249 - sonar_pipeline - INFO - Decoding embedding with shape: torch.Size([1, 1024]), dtype: torch.float32
2025-06-11 10:35:18,926 - sonar_pipeline - INFO - Decoded text: ['The history of the state of Hokkaido dates back to the first century.']
Step 7 model next-token guess: The history of the state of Hokkaido dates back to the first century.
2025-06-11 10:35:18,926 - sonar_pipeline - INFO - Decoding embedding with shape: torch.Size([1, 1024]), dtype: torch.float32
2025-06-11 10:35:24,766 - sonar_pipeline - INFO - Decoded text: ['The history of the state of Hokkaido dates back to the early 19th century.']
Step 8 model next-token guess: The history of the state of Hokkaido dates back to the early 19th century.
2025-06-11 10:35:24,766 - sonar_pipeline - INFO - Decoding embedding with shape: torch.Size([1, 1024]), dtype: torch.float32
2025-06-11 10:35:30,462 - sonar_pipeline - INFO - Decoded text: ['It was a time of great change in the history of Japan.']
Step 9 model next-token guess: It was a time of great change in the history of Japan.
[1]+  Done                    clear
```

*Ok, so what's going on here? We'll break it down step by step because there's a lot going on and I'm not going to assume that all readers are familiar with the concepts of Attention and Transformer architectures.*

In this example, the model was provided with a prompt of ten sentences and it generated a ten sentence response.

It worked by taking the sentences then passing them trough the SONAR encoder to convert them into ten embeddings of shape [1, 1024] (1024D vectors in batches of 1) each.

Those embeddings are then stacked into a single batch with a sequence length of ten, resulting in a tensor of shape [1, 10, 1024].  The sequence is then passed into the model, where each embedding attends to all previous embeddings in the sequence, and then attempts to predict the next embedding in the sequence. "Attention" is a key component of the Transformer architecture, and it's what allows the model to "attend" to the previous embeddings in the sequence, by being 'aware' of all sentences that came before the current sentence. Attention is what allows the model to 'understand' the context of the previous embeddings in the sequence and how those embeddings modify the meaning of the current embedding. This process is performed to predict the next embedding that it 'guesses' is the most likely to follow *for each embedding in the sequence*.

For example, the first embedding from the prompt (shown in English for understandability, but in actuality it's a 1024D SONAR concept vector), *"[[Start of Text.]]"*, passes through the model and attemts to predict the next embedding in the sequence... it has no previous embeddings to attend to, so it just predicts the next embedding in the sequence, which in this case is *"He is the founder, chairman, and chief executive officer (CEO) of the company. He is also the founder and chairman of the company."*. This is the first embedding in the output sequence, so we've added it to position [0, 0, :] in that output tensor. Based on this 'guess', we can see a few things:
* The model has learned to output data that is similar to the data that it was trained on, which up until now is exclusively Japanese Wikipedia articles.
* The model currently suffers from a common issue seen in smaller "large" models (just like LLMs): repetition.

The next embedding from the prompt, *"Japan's long history is divided into many distinct periods, each contributing to the country and culture in their own way."*, passes through the model and attends to the previous embedding, *"[[Start of Text.]]"*, and then attempts to predict the next embedding in the sequence. This time, it has useful information to attend to, so ideally it can predict the next embedding in the sequence better. Based on these two embeddings, the model can now predict the next embedding in the sequence, which in this case is *"It's called the "City of Dreams"."*. This is the second embedding in the output sequence, so now we've added this one to position [0, 1, :] in that same output tensor.

The process continues for the third embedding, *"The Sengoku era was a period of great conflict in Japan."*, and in this case the model is able to attend to the previous two embeddings, *"[[Start of Text.]]"* and *"Japan's long history is divided into many distinct periods, each contributing to the country and culture in their own way."*... by this point, it has far more information to attend to. Based on these three embeddings, the model can now predict the next embedding in the sequence, which in this case is *"The history of Japan, the history of China, the history of Japan, the history of Japan, the history of Japan, the history of Japan."*. This is the third embedding in the output sequence, which we place at position [0, 2, :] in the output tensor. By this point, we can see: 
* The model has latched on to the topic of the prompt and gives us an output that's no longer unrelated.
* The repetition issue is still a problem

This process continues for all the remaining embeddings, where each embedding attends to the previous embeddings in the sequence and attempts to predict the next embedding in the sequence.

This means that the model also outputs a tensor of shape [1, 10, 1024] (identical to the shape of the input tensor), with embedding_0 being the prediction for the first embedding, embedding_1 being the prediction for the second embedding, and so on... as such, for normal (inference) applications, the final embedding in the sequence is the 'most useful' as it is the model's attempt to predict the next embedding after the ending of all provided sentences; It's the model's prediction of novel information. When actually applying the model to tasks in next-sentence prediction, we'll always only use this last embedding, then continue building a larger oputput by taking each last output sentence, then using it to build up a larger output in an autoregressive manner.

**Note: I explained this process sequentially, but in actuality, the model is performing all of these operations *at the same time* by using a teqhnique called "Masked Attention" to ensure that each embedding only attends to the previous embeddings in the sequence... but as I'm explaining this at a level that targets learners, I'm sacrificing some accuracy to put this in a form that's easier to understand when you're still learning Transformers.**

Although realworld applications will only use the last output, for this inference script, though, all embeddings in the sequence are decoded for better visualization of the model's behavior. So, once this output is generated, all embeddings in the tensor are then passed through the SONAR decoder to convert the embeddings back into text. This is how we see the output that for all input embeddings as follows:

> 0 - [[Start of Text.]] -> He is the founder, chairman, and chief executive officer (CEO) of the company. He is also the founder and chairman of the company.
>
> 1 - Japan's long history is divided into many distinct periods, each contributing to the country and culture in their own way. -> It's called the "City of Dreams".
>
> 2 - The Sengoku era was a period of great conflict in Japan. -> The history of Japan, the history of China, the history of Japan, the history of Japan, the history of Japan, the history of Japan.
> 
> 3 - Many clans and their samurai fought in that time. -> The history of the Japanese empire can be traced back to the period of the Qing Dynasty.
>
> 4 - The fighting lasted for many decades, but it was ultimately brought to an end in the unification of Japan. -> The history of the state of Hokkaido dates back to the first century.
>
> 5 - It was followed by a period of peace and cultural growth. -> The history of Japan, the history of Japan, the history of Japan, the history of Japan.
>
> 6 - This period was known as the Edo period. -> It was a time of great turmoil in the history of mankind.
>
> 7 - This period brought forward many new forms of art and culture. -> The history of the state of Hokkaido dates back to the first century.
>
> 8 - These include forms such as ukiyo-e woodblock paintings, kabuki theater, and haiku poetry. -> The history of the state of Hokkaido dates back to the early 19th century.
>
> 9 - The impacts of the Edo period are large and still felt in the present day cultural landscape. -> It was a time of great change in the history of Japan.

This is a good start, but it's certainly not very smart yet. Although this model is still not outputting any information that's useful for practical applications, it has already had to learn an incredible amount of information about the world to get to this point... for example it has absorbed concepts such as "countries" and "history", that "Japan" is a "Country", that talking about "History" means you often explain it in terms of "Centuries", that there are connections between "Japan", "China", and "Hokkaido", and that concepts like "Change" and "Turmoil" are often associated with "History". 





## AIの事前学習手順 ｜ Training

### 事前テストを実行する ｜ Dry run (sanity check) ## ToDo: fix this
```bash
clear & uv run --extra gpu -m src.LexaLCM.Main --dry-run --verbose
```

### 事前学習手順を始める ｜ Run the training
```bash
clear & uv run --extra gpu -m src.LexaLCM.Main -v
```

## Testing

**Currently, this is not working... I'll patch it in Pre2**

### Test the model
```bash
clear & uv run --extra gpu pytest Tests/TestModel.py
```

### Test the data pipeline
```bash
clear & uv run --extra gpu pytest Tests/TestData.py
```
## Special Concepts
These sentences are the equivalent of special tokens in an LLM. They're a quirk of continuous concept embedding space that the model exists within; because there is no discrete separation of tokens, all special signifiers must coinhabit the same 1024D concept embedding spaces as the normal sentences to be translated. there is no separation.
### Start of Text
日本語：

English:
`[[Start of text.]]`

### End of Text
日本語：

English:
`[[End of text.]]`

### Pad
日本語：

English:

### System
日本語：

English:

### Tool
日本語：

English:

### AI
日本語：

English:

### User
日本語：

English:

## Dataset handling

If you have a dataset in the format of the Meta FAIR "Large Concept Models" paper, you can convert it to the LexaLCM format using the following command:

```bash
clear & uv run --extra data src/Scripts/Data/ConvertMetaParquet.py -i src/_TEMP/DirtyDatasets/ -o src/LexaLCM/Content/Datasets/ -n wikipedia_data_50k
```

where:
- `-i` is the path to the directory with the dataset
- `-o` is the path to the directory to save the converted dataset
- `-n` is the name of the dataset

and in this example, the dataset is called "wikipedia_data_50k" and is located in the directory `src/_TEMP/DirtyDatasets/`. The converted dataset will be saved in the directory `src/LexaLCM/Content/Datasets/` (the default dataset directory for the LexaLCM).












### Verify the embeddings

```bash
uv run --extra data src/Scripts/Data/VerifyEmbeddings.py 
```

where:
- `-d` is the path to the parquet files

For example:
```bash
clear & uv run --extra data src/Scripts/Data/VerifyEmbeddings.py -d src/LexaLCM/Content/Datasets/Wikipedia_Ja
```

### Convert the dataset to the LexaLCM format

```bash
uv run --extra data src/Scripts/Data/ConvertMetaParquet.py
```

where:
- `-d` is the path to the parquet files



### Visualize the dataset

```bash
uv run --extra data src/Scripts/Data/VisualizeDataset.py 
```

Where:
- `-d` is the path to the parquet files
- `-s` is if the dataset is sampled or if all the files are used (sample=True samples 10% of the files)
- `-b` is the batch size for the evaluation process (default is 10)

For example:
```bash
clear & uv run --extra data src/Scripts/Data/VisualizeDataset.py -b 20 -d src/LexaLCM/Content/Datasets/Wikipedia_Ja
```




## Bootstrap the model

```bash
clear & uv run --extra gpu src/LexaLCM/LCM/Utils/BootstrapLCM.py
```



