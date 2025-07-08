# ToDo

## Critical

- [ ] thoroughly check out the whole AdaLN system

## High

- [ ] PreNet: Normalizes and centers the input embeddings
- [ ] PostNet: Denormalizes the output and shifts it back to the SONAR embedding space by the noralizer's original offset.
- [ ] make sure all attention masks are working correctly

## Medium

- [ ] Debug to verify the attention masks are all bool
- [ ] Exponential Moving Average (EMA) for weight stabilization during training or inference
- [ ] Make DataHandler.py able to load multiple datasets at once
- [ ] Remote text and text_sentences from the dataset to speed up loading

## Low

- [ ] Make TestInference.py select model from command line/config file
- [ ] Rewrite the SONAR Encoder/decoder pipeline pipelines.py... it's currently fugly vibe-code. Hopefully FAIR will release their new fairseq2 that supports torch2.7.0-cu128 soon

## Lowest
=