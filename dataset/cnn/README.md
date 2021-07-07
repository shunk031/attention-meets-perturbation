# CNN

- [Questions](https://drive.google.com/uc?export=download&id=0BwmD_VLjROrfTTljRDVZMFJnVWM)
- [Stories](https://drive.google.com/uc?export=download&id=0BwmD_VLjROrfTHk4NFg2SndKcjQ)

```shell
$ tar -xvzf cnn.tgz
```
## Build vocabulary

```
$ allennlp make-vocab config/cnn/train.jsonnet -s data/cnn/vocabulary --include-package agt --force
```

## Reference

- [deepmind/rc-data: Question answering dataset featured in "Teaching Machines to Read and Comprehend](https://github.com/deepmind/rc-data/)
