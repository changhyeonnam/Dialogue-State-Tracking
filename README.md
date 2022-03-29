# README

## Abstract

Open Vocabulary DST model 중 하나인 TRADE가 BaseLine model로 주어졌다. Scalable한 Dialogue State model을 개발하기 위해선, predefined ontology를 사용하는 DST모델과 hybrid하는 방향 보다는 Open Vocabulary DST model만을 사용하는 방향으로 개발해야 한다고 생각했다. 해당 report에서는 TRADE의 Encoder에 pretrained model ‘klue/Roberta-base’를 이용해 fine tuning을 통한 성능 향상을 목적으로 개발과 실험을 진행했다. ‘Roberta-base’의 hidden state output에 대해 2개의 LSTM layer, 1개의 Linear Layer를 통과시켜 완성한 Encoder는 public test set 기준으로 기존의 Baseline model보다 Joint accuracy 0.2155 만큼 더 좋은 성능을 보였다. 이후 진행될 Future work 중 하나인 SOM-DST의 Encoder에 대해 해당 report에서 개발한 Encoder를 사용하면 보다 좋은 성능을 낼 수 있을거라 기대할 수 있다.  

## Baseline model

Baseline model로 주어진 TRADE 모델은 Ontology-free(Open vocabulary) DST model로써,  parameter sharing과 Copy mechanism을 통해 Multi domain, Multi-turn mapping problem문제를 해결했다. 해당 task를 진행하기 앞서, 이후 진행될 과정과 비교를 위해 baseline model의 성능을 먼저 확인해 보았다. (Joint_acc, Slot_acc, Slot_f1은 모두 public test set기준에서 평가한 지표들이다.)

|  | Joint_acc | Slot_acc | Slot_f1 | learnig rate | batch_size | epoch |
| --- | --- | --- | --- | --- | --- | --- |
| TRADE | 0.4823 | 0.9800 | 0.9074 | 1e-4 | 16 | 30 |

## Dataset

주어진 train_dials.json은 총 7000개의 대화, 5개의 Domain Class, 26개의 Slot Classt으로 이루어진 데이터 셋이다. eval_dials.json은 총 1000개의 대화, 5개의 Domain Class로 이루어진 데이터 셋이다. (slot은 주어지지 않았다.) 다음과 같이 train, eval의 domain class별 데이터 개수를 볼 수 있다.

```python
# train_dials.json
{'관광': 3732, '식당': 4020, '지하철': 650, '택시': 2374, '숙소': 3802}
# eval_dials.json
{'관광': 4873, '식당': 5327, '지하철': 941, '택시': 2942, '숙소': 5063}
```

위의 class별 개수를 보면, `‘지하철’` domain에 대한 data개수가 상대적으로 적은 class imbalance가 있는 것을 확인할 수 있다. Evaluation 데이터셋에도 똑같은 양상이 나타나있는 것으로 보아,  private testset의 `‘지하철’` domain과 관련된 dataset에서 좋지 못한 성능을 보일 수도 있음을 예상할 수 있다. 하지만 TRADE의 모델에서의 parameter sharing을 통해 domain간 겹치는 slot을 이용하여 이를 어느정도 해결할 수 있다.

모든 slot이 나타나있는 slot_meta.json을 보면 45개의 slot class가 존재하는 것을 확인할 수 있지만, train_dials.json 파일에는 26개의 slot class만이 존재한다. TRADE 모델의 경우 Open Vocab DST model이기 때문에, predefined [domian - slot]외의 unseen domain, slot, value에 대해서 아웃풋을 생성할 수 있다. (이는 State Generator의 Pointer-Generator으로부터 얻는 이점이다.)

### Preprocessing

대부분의 down-stream task를 수행할때, Konlpy package의 Mecab과 같은 한국어 형태소 분석기로 text를 tokenize한 뒤에, pretrained tokenizer로 tokenize하였다. 이에 대한 근거 다음과 같다.  [Dialog-BERT](https://deview.kr/2019/schedule/285?lang=en) 링크의 발표를 보면 pretrained model의 tokenizer만을 사용하는 것 보다 형태소 분석기로 tokenize한 후에, pretrained model의 tokenizer을 사용하는 경우가 더 좋은 성능을 보인다고 한다. 다양한 형태소 분석기가 있었지만, 여러가지 방법을 실험하기 위해서는 가장 빠른 분석기가 필요했다. 그래서 Mecab을 사용하여 각 dialog text을 한국어 형태소로 분석을 하였다. 

### Tokenizer Validation

위에서 언급한 (1) pretranied tokenizer, (2) mecab (3) mecab+pretrained tokenizer 을 사용하였을 때, subword에 대해 tokenize하는 성능을 검증하기 위해 몇개의 testcase로 실험을 해보았다. testcase 중 하나가 다음과 같다.

```python
# train_dials.json파일의 json [10]['dialogue'][0]['text']
'저기 서울 남쪽의 저렴한 숙소를 찾아주세요.'
# only mecab
['저기', '서울', '남쪽', '의', '저렴', '한', '숙소', '를', '찾', '아', '주', '세요', '.']
# only roberta-tokenizer
['저기', '서울', '남쪽', '##의', '저렴', '##한', '숙소', '##를', '찾아', '##주', '##세요', '.']
# mecab + roberta-tokenizer
['저기', '서울', '남쪽', '의', '저렴', '한', '숙소', '를', '찾', '아', '주', '세요', '.']
```

위와 같이 단순히 pretrained tokenizer만 사용했을때 보다 mecab과 함께 사용했을때, ‘찾아’ → ‘찾' + ‘아'와 같이 분절된 것을 확인할 수 있었다. 단순히 몇몇 케이스만으로 어떤 접근법이 좋은 성능인지 검증하는데 불명확할 것이라 판단되어 baseline 모델에 적용하여 이를 확인했다.

|  | Joint_acc | Slot_acc | Slot_f1 | learnig rate | batch_size | epoch |
| --- | --- | --- | --- | --- | --- | --- |
| TRADE(mecab) validation set | 0.2713 | 0.9648 | 0.8056 | 1e-4 | 16 | 30 |
| TRADE(mecab) public testset | 0.2614 | 0.9606 | 0.8142 | 1e-4 | 16 | 30 |

해당 task에서는 [slot-value]의 set이 정확히 일치하는지 평가하는 Joint Goal Accuracy와 같은 경우, public/private test의 true set에도 mecab으로 tokenize되어 있어야 model의 성능 측정이 유의미한다는 것을 뒤늦게 알았다. 그리고, 해당 validation set에서도 위와 같이 좋지 못한 성능을 보였다.  그래서 pretrained tokenize만을 사용하여 dialog를 tokenize하기로 했다.

## Model

주어진 TRADE 모델은 크게 Encoder, State Generator, Slot Gate로 이뤄진다. TRADE paper에 따르면, Encoder에 대해 어떠한 Encoder로도 대체 가능하다고 설명되어 있다. 이에 대해 BERT 계열의 pretrained model을 Encoder로써 사용하고자 했다.  

대부분의 공개된 한국어 pretranied model은 Wikipedia, Wikinews, NSMC와 같은 문어체의 dataset으로 학습되었다. 해당 과제에서 주어진 데이터는 대화체이고, 이 차이로 인해 학습이 잘 되지 않을 것이라고 생각했다. 하지만  [Dialog-BERT](https://deview.kr/2019/schedule/285?lang=en) deview 에서 발표한 내용을 참고해보면, 대화체 데이터로 fine-tuning된 model은 동일한 domain(=대화체)에 대해서 inference시, 성능이 좋은 것을 확인할 수 있었다.

[KLUE-benchmark/KLUE](https://github.com/KLUE-benchmark/KLUE) repository에 있는 Pretrained model 중 대부분의 task에서 성능이 좋은 KLUE-RoBERTa-base 모델과 RoBERTa-large 모델을 Encoder로써 사용하여 실험을 해보았다.

### Encoder

Roberta의 마지막 hidden layer wieght만을 하나의 Linear Layer (`nn.Linear(config.hidden_size, config.hidden_size)`)에 통과시켜 사용하였다. 앞서 말한 pretrained model를 Encoder로 사용한 결과는 다음과 같다.  

|  | Joint_acc | Slot_acc | Slot_f1 | learnig rate | batch_size | epoch |
| --- | --- | --- | --- | --- | --- | --- |
| RobertaTrade(Roberta-large) | 0.5415 | 0.9837 | 0.9266 | 5e-5 | 8 | 30 |
| RobertaTrade(Roberta-base) | 0.6135 | 0.9870 | 0.9414 | 5e-5 | 16 | 30 |

`klue/roberta-large` 학습할 경우,  `roberta-base` 모델보다 더 빠르게 overfitting 되었다.(epoch = 10이 되었을때 부터 학습이 되지 않았다.)  hyper-parameter opitmazation을 통해 이 문제를 해결할 수 있다고 생각했지만, 주어진 기간 동안 가장 좋은 성능을 보여주는 모델을 개발하기 위해 `roberta-base`를 선택하였다. 

길이가 긴 Document에 대한 classification task에서의 finetuning에 대한 [논문](https://arxiv.org/pdf/1910.10781.pdf)에서 제시한 RoBERT(Recuurence over BERT)모델의 아이디어를 사용 하였다. RoBerta의 마지막 hidden state를 input으로 하여,  LSTM(num_layer=2) 에 통과시키고, LSTM의 두개의 hidden state을 concat한후, Fully connected layer에 넣는 모델을 Encoder로써 사용해 보았다.

|  | Joint_acc | Slot_acc | Slot_f1 | learnig rate | batch_size | epoch |
| --- | --- | --- | --- | --- | --- | --- |
| RobertaTrade( roberta- base  -one Linear layer) | 0.6135 | 0.9870 | 0.9411 | 5e-5 | 16 | 30 |
| RobertaTrade( roberta-base - LSTM) | 0.6143 | 0.9870 | 0.9414 | 5e-5 | 16 | 30 |

roberta-base의 경우, epcoh 30을 넘어 epoch 50이상에서도 validation set에 overfitting되지 않는 성능을 보여주었다. 위의 table에서 성능이 더 좋은 roberta-base-LSTM model을 overfitting 될때까지 epoch수를 늘려 학습을 해보았다.

|  | Joint_acc | Slot_acc | Slot_f1 | learnig rate | batch_size | epoch |
| --- | --- | --- | --- | --- | --- | --- |
| RobertaTrade( roberta-base - LSTM) | 0.7048 | 0.9899 | 0.9555 | 5e-5 | 16 | 100 |

## Future Work

Encoder의 finetuing이 끝난 후, 다음 개발하고 있던 모델은  SOM DST (Efficient Dialogue State Tracking by Selectively Overwriting Memory)이다. 기존의 TRADE모델의 경우 모든 slot을 generation을 하여, 이에 대한 비효율성이 생긴다. 이를 해결하고자 개발된 SOM DST모델을 개발하고 있었다. Utterance가 input으로 들어가는 SOM-DST의 State Operation Predictor를 통해, Update가 필요한 경우에만 Value를 생성함으로써 효율적인 Generation이 가능하게 된다. SOMDST, TRADE 모델을 Soft Voting 방법으로 ensemble 하여, 보다 좋은 결과를 기대할 수 있다고 생각한다.

## Directory tree

```python
.
├── README.md
├── data
│   ├── Simple\ EDA.ipynb
│   ├── eval_dataset
│   │   ├── eval_dials.json
│   │   ├── eval_dials.pkl
│   │   └── slot_meta.json
│   └── train_dataset
│       ├── dev_dials.pkl
│       ├── ontology.json
│       ├── slot_meta.json
│       ├── train_dials.json
│       └── train_dials.pkl
├── data_utils.py
├── eval_utils.py
├── evaluation.py
├── inference.py
├── model
│   ├── Roberta_TRADE.py
│   └── TRADE.py
├── parser.py
├── preprocessor.py
├── results
│   ├── exp_config.json
│   ├── predictions.csv
│   └── slot_meta.json
└── train.py

```

GPU sever에서 권한 문제로 인해, mecab package가 설치되지 않아 미리 mecab으로 tokenize한 object을 pkl로 저장하여 사용했습니다.

## Development enviroment

- OS: Max OS X
- IDE: pycharm
- GPU: NVIDIA RTX A6000

## Dependency

```python
Python        >= 3.7
tokenizers    >= 0.9.4
torch         >= 1.10.2
konlpy        >= 0.6.0
pandas        >= 1.3.5
numpy         >= 1.21.5
transformers  >= 4.2.0
```

## Quick start for training

```python
python train.py -b 16 -e 20
```

다음 command를 입력하면 최종모델과 동일한 환경에서 학습을 한다.

## Quick start for inference

```python
python inference.py
```

### Reference

1. [KLUE-benchmark/KLUE](https://github.com/KLUE-benchmark/KLUE) 
2. [HIERARCHICAL TRANSFORMERS FOR LONG DOCUMENT CLASSIFICATION](https://arxiv.org/pdf/1910.10781.pdf)
3. [Transferable Multi-Domain State Generator for Task-Oriented Dialogue Systems](https://arxiv.org/pdf/1905.08743.pdf)
4. [Efficient Dialogue State Tracking by Selectively Overwriting Memory (SOM DST)](https://arxiv.org/pdf/1911.03906.pdf)
