# general hparams

train 200k steps
regularisation = default huggingface, none
LR = {1, 5.62} x 10-6, {1, 5.62} x 10−5, {1, 5.62} x 10−4 
batch size = 3.2M samples
tri-stage LR (10%, 40%, 50%)
for ASR, set bias of FC layer to prior of letters, freeze transformers 3k steps
freeze cnn during train

pick best checkpoint during training based on 1/4 val WER + 1/4 val EER
pick best LR based on 1/4 dev WER + dev EER

report eval metrics:
test set ASR: librispeech test-other, switchboard?
test set SKR: vox2-hard, switchboard?

loss functions:
CTC-loss for ASR
AAM-softmax loss for SKR
MTL: lambda={0.5, 0.9} static weighting between CTC loss and AAM-softmax loss (no dwa)

# Table 1

wav2vec2 BASE, 1 fc for ASR, mean-pooling + 1 FC for SKR

## STL

### ASR librispeech

#### librispeech

```
TODO
```

#### vox2

```
TODO
```

### SKR

#### librispeech

```
TODO
```

#### vox2

```
TODO
```

## MTL

### joint

#### librispeech

```
TODO
```

#### librispeech + voxceleb

```
TODO
```

### disjoint, 3 seconds

#### librispeech

```
TODO
```

#### librispeech + voxceleb

```
TODO
```
### disjoint, 9 seconds

#### librispeech

```
TODO
```

#### librispeech + voxceleb

```
TODO
```

# Table 2



## STL - 3 architectures
data=vox2

```
TODO
```

## MTL disjoint:
data=ls+vox2

```
TODO
```

# Table 3

all model checkpoints result from experiments in Table 1


