# ToDo

## Critical

- [ ] >1B Params
- [ ] Multi-GPU pretraining
  - [ ] Make it so that the system honors gpus.custom_multi_gpu in Config_Pretrain.yaml
    - [ ] disable the autowrapper
    - [ ] disable the NO-OP to() in LCM_Model.py
    - [ ] Check that there aren't any other places that could cause problems or eat performance
- [ ] Improve the dataset:
  - [ ] 10x size
  - [ ] Diverse selection of data
  - [ ] Cleaning data to reduce "Garbage-in-garbage-out" problems 
  - [ ] Prepping with data augmentation to significantly increase the "bang-for-your-buck" factor, especially with the smaller but critical data sources
- [ ] thoroughly check out the whole AdaLN system
- [ ] Alert if any weights or biases are stuck at 0 or hit NaN
- [ ] 

## High

- [ ] Optimize the training loop for speed
- [ ] Optimize the model's speed
- [ ] Optimize the dataloader's speed
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
