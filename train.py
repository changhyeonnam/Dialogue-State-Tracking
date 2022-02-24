import json
import os
import random
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import (AdamW,
                          BertTokenizer,
                          get_linear_schedule_with_warmup,
                          AutoTokenizer,
                          )


from data_utils import (WOSDataset,
                        get_examples_from_dialogues,
                        load_dataset,
                        set_seed)

from eval_utils import DSTEvaluator
from evaluation import _evaluation
from inference import inference
from model.TRADE import TRADE, masked_cross_entropy_for_value
from model.Roberta_TRADE import RobertaTrade
from preprocessor import TRADEPreprocessor
from parser import args

# print device info
device = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')
print('device:',device)

# print gpu info
if torch.cuda.is_available():
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())

if __name__ == "__main__":

    # random seed 고정
    set_seed(args.random_seed)

    # Data Loading
    train_data_file = f"{args.data_dir}/train_dials.json"
    slot_meta = json.load(open(f"{args.data_dir}/slot_meta.json"))
    train_data, dev_data, dev_labels = load_dataset(train_data_file)

    train_examples = get_examples_from_dialogues(
        train_data, user_first=False, dialogue_level=False
    )
    dev_examples = get_examples_from_dialogues(
        dev_data, user_first=False, dialogue_level=False
    )

    # Define Preprocessor
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    processor = TRADEPreprocessor(slot_meta, tokenizer)
    args.vocab_size = len(tokenizer)
    print(f'tokenizer vocab size :{args.vocab_size}')
    args.n_gate = len(processor.gating2id) # gating 갯수 none, dontcare, ptr

    #Extracting Featrues
    # train_features = processor.convert_examples_to_features(train_examples)
    # dev_features = processor.convert_examples_to_features(dev_examples)
    #
    # with open('data/train_dataset/train_dials.pkl',mode='wb') as f:
    #     pickle.dump(train_features,f)
    # with open('data/train_dataset/dev_dials.pkl',mode='wb') as f:
    #     pickle.dump(dev_features,f)
    # quit()
    if args.mecab == 'True':
        with open('data/train_dataset/train_dials.pkl', mode='rb') as f:
            train_features=pickle.load(f)
        with open('data/train_dataset/dev_dials.pkl', mode='rb') as f:
            dev_features=pickle.load(f)
    else:
        print('Not Using mecab')
        train_features = processor.convert_examples_to_features(train_examples)
        dev_features = processor.convert_examples_to_features(dev_examples)

    tokenized_slot_meta = []
    for slot in slot_meta:
        tokenized_slot_meta.append(
            tokenizer.encode(slot.replace("-", " "), add_special_tokens=False)
        )

    # Model 선언
    model = RobertaTrade(config=args, tokenized_slot_meta=tokenized_slot_meta)
    model.to(device)
    print("Model is Loaded")
    # Slot Meta tokenizing for the decoder initial inputs

    train_data = WOSDataset(train_features)
    train_sampler = RandomSampler(train_data)
    train_loader = DataLoader(
        train_data,
        batch_size=args.train_batch_size,
        sampler=train_sampler,
        collate_fn=processor.collate_fn,
    )
    print("# train:", len(train_data))

    dev_data = WOSDataset(dev_features)
    dev_sampler = SequentialSampler(dev_data)
    dev_loader = DataLoader(
        dev_data,
        batch_size=args.eval_batch_size,
        sampler=dev_sampler,
        collate_fn=processor.collate_fn,
    )
    print("# dev:", len(dev_data))
    
    # Optimizer 및 Scheduler 선언
    n_epochs = args.num_train_epochs
    t_total = len(train_loader) * n_epochs
    warmup_steps = int(t_total * args.warmup_ratio)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

    loss_fnc_1 = masked_cross_entropy_for_value  # generation
    loss_fnc_2 = nn.CrossEntropyLoss()  # gating

    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)

    json.dump(
        vars(args),
        open(f"{args.model_dir}/exp_config.json", "w"),
        indent=2,
        ensure_ascii=False,
    )
    json.dump(
        slot_meta,
        open(f"{args.model_dir}/slot_meta.json", "w"),
        indent=2,
        ensure_ascii=False,
    )

    best_score, best_checkpoint = 0, 0
    for epoch in range(n_epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            input_ids, segment_ids, input_masks, gating_ids, target_ids, guids = [
                b.to(device) if not isinstance(b, list) else b for b in batch
            ]

            # teacher forcing
            if (
                args.teacher_forcing_ratio > 0.0
                and random.random() < args.teacher_forcing_ratio
            ):
                tf = target_ids
            else:
                tf = None


            all_point_outputs, all_gate_outputs = model(
                input_ids, segment_ids, input_masks, target_ids.size(-1), tf
            )

            # generation loss
            loss_1 = loss_fnc_1(
                all_point_outputs.contiguous(),
                target_ids.contiguous().view(-1),
                tokenizer.pad_token_id,
            )

            # gating loss
            loss_2 = loss_fnc_2(
                all_gate_outputs.contiguous().view(-1, args.n_gate),
                gating_ids.contiguous().view(-1),
            )
            loss = loss_1 + loss_2

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if step % 100 == 0:
                print(
                    f"[{epoch}/{n_epochs}] [{step}/{len(train_loader)}] loss: {loss.item()} gen: {loss_1.item()} gate: {loss_2.item()}"
                )

        predictions = inference(model, dev_loader, processor, device)
        eval_result = _evaluation(predictions, dev_labels, slot_meta)
        for k, v in eval_result.items():
            print(f"{k}: {v}")

        if best_score < eval_result['joint_goal_accuracy']:
            print("Update Best checkpoint!")
            best_score = eval_result['joint_goal_accuracy']
            best_checkpoint = epoch

        torch.save(model.state_dict(), f"{args.model_dir}/model-{epoch}.bin")
    print(f"Best checkpoint: {args.model_dir}/model-{best_checkpoint}.bin")
