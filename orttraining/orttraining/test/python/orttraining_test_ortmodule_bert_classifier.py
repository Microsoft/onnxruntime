import logging
import argparse
import torch
import wget
import os
import pandas as pd
import zipfile
from transformers import BertTokenizer, AutoConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import numpy as np
import random
import time
import datetime

import onnxruntime
from onnxruntime.training import ORTModule
# from perf_log.ort_module_perf_test_tools import log_perf_metrics

def train(model, optimizer, scheduler, train_dataloader, epoch, device, args):
    # ========================================
    #               Training
    # ========================================
    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

    # Perform one full pass over the training set.
    print('\n======== Epoch {:} / {:} with batch size {:} ========'.format(epoch + 1, args.epochs, args.batch_size))

    # Measure how long the training epoch takes.
    t0 = time.time()
    start_time = t0

    # Reset the total loss for this epoch.
    total_loss = 0

    # Put the model into training mode. Don't be mislead--the call to
    # `train` just changes the *mode*, it doesn't *perform* the training.
    # `dropout` and `batchnorm` layers behave differently during training
    # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):

        if step == args.train_steps:
            break

        # Unpack this training batch from our dataloader.
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using the
        # `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because
        # accumulating the gradients is "convenient while training RNNs".
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        model.zero_grad()

        # Perform a forward pass (evaluate the model on this training batch).
        # This will return the loss (rather than the model output) because we have provided the `labels`.
        # The documentation for this `model` function is here:
        #   https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification

        outputs = model(b_input_ids,
                        attention_mask=b_input_mask,
                        labels=b_labels)

        # The call to `model` always returns a tuple, so we need to pull the
        # loss value out of the tuple.
        loss = outputs[0]

        # Progress update every 40 batches.
        if step % args.log_interval == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            curr_time = time.time()
            elapsed_time = curr_time - start_time

            # Report progress.
            print(f'Batch {step:4} of {len(train_dataloader):4}. Execution time: {elapsed_time:.4f}. Loss: {loss.item():.4f}')
            start_time = curr_time

        if args.view_graphs:
            import torchviz
            pytorch_backward_graph = torchviz.make_dot(outputs[0], params=dict(list(model.named_parameters())))
            pytorch_backward_graph.view()

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value
        # from the tensor.
        total_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)

    epoch_time = time.time() - t0
    print("\n  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:.4f}s".format(epoch_time))
    return epoch_time

def test(model, validation_dataloader, device, args):
    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.
    print("\nRunning Validation with batch size {:} ...".format(args.test_batch_size))

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    t0 = time.time()

    # Tracking variables
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():

            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have
            # not provided labels.
            # token_type_ids is the same as the "segment ids", which
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification

            # TODO: original sample had the last argument equal to None, but b_labels is because model was
            #       exported using 3 inputs for training, so validation must follow.
            #       Another approach would be checkpoint the trained model, re-export the model for validation with the checkpoint
            outputs = model(b_input_ids,
                            attention_mask=b_input_mask,
                            labels=b_labels)

        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.
        logits = outputs[1]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        # Accumulate the total accuracy.
        eval_accuracy += tmp_eval_accuracy

        # Track the number of batches
        nb_eval_steps += 1

    # Report the final accuracy for this validation run.
    epoch_time = time.time() - t0
    accuracy = eval_accuracy/nb_eval_steps
    print("  Accuracy: {0:.2f}".format(accuracy))
    print("  Validation took: {:.4f}s".format(epoch_time))
    return epoch_time, accuracy

def load_dataset(args):
    # 2. Loading CoLA Dataset

    def _download_dataset(download_dir):
        if not os.path.exists(download_dir):
            # Download the file (if we haven't already)
            print('Downloading dataset...')
            url = 'https://nyu-mll.github.io/CoLA/cola_public_1.1.zip'
            wget.download(url, './cola_public_1.1.zip')
        else:
            print('Reusing cached dataset')

    if not os.path.exists(args.data_dir):
        _download_dataset('./cola_public_1.1.zip')
        # Unzip it
        print('Extracting dataset')
        with zipfile.ZipFile('./cola_public_1.1.zip', 'r') as zip_ref:
            zip_ref.extractall('./')
    else:
        print('Reusing extracted dataset')

    # Load the dataset into a pandas dataframe.
    df = pd.read_csv(os.path.join(args.data_dir, "in_domain_train.tsv"), delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])

    # Get the lists of sentences and their labels.
    sentences = df.sentence.values
    labels = df.label.values

    # 3. Tokenization & Input Formatting

    # Load the BERT tokenizer.
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # Set the max length of encoded sentence.
    # 64 is slightly larger than the maximum training sentence length of 47...
    MAX_LEN = 64

    # Tokenize all of the sentences and map the tokens to their word IDs.
    input_ids = []
    for sent in sentences:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        encoded_sent = tokenizer.encode(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                    )

        # Pad our input tokens with value 0.
        if len(encoded_sent) < MAX_LEN:
            encoded_sent.extend([0]*(MAX_LEN-len(encoded_sent)))

        # Truncate to MAX_LEN
        if len(encoded_sent) > MAX_LEN:
            encoded_sent = encoded_sent[:MAX_LEN]

        # Add the encoded sentence to the list.
        input_ids.append(encoded_sent)

    input_ids = np.array(input_ids, dtype=np.longlong)

    # Create attention masks
    attention_masks = []

    # For each sentence...
    for sent in input_ids:
        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]

        # Store the attention mask for this sentence.
        attention_masks.append(att_mask)

    # Use 90% for training and 10% for validation.
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels,
                                                                random_state=2018, test_size=0.1)
    # Do the same for the masks.
    train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels,
                                                random_state=2018, test_size=0.1)

    # Convert all inputs and labels into torch tensors, the required datatype
    # for our model.
    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)

    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)

    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)

    # Create the DataLoader for our training set.
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

    # Create the DataLoader for our validation set.
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=args.test_batch_size)

    return train_dataloader, validation_dataloader

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    '''Takes a time in seconds and returns a string hh:mm:ss'''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# def bert_classifier_log_perf_metrics(args, train_dataloader, perf_metrics_start_time):
#     total_train_steps = 0
#     if args.train_steps > 0:
#         total_train_steps = args.train_steps * args.epochs
#     else:
#         total_train_steps = len(train_dataloader) * args.epochs

#     perf_metrics = {}
#     perf_metrics['Model'] = 'orttraining_test_ortmodule_bert_classifier'
#     perf_metrics['BatchId'] = 'na'
#     perf_metrics['ModelName'] = 'bert'
#     perf_metrics['DisplayName'] = 'bert'
#     perf_metrics['UseMixedPrecision'] = False
#     perf_metrics['UseAutoCast'] = False
#     perf_metrics['UseDeepSpeed'] = False
#     perf_metrics['Optimizer'] = 'AdamW'
#     perf_metrics['BatchSize'] = args.batch_size
#     perf_metrics['SeqLen'] = 64                 # MAX_LEN = 64
#     perf_metrics['PredictionsPerSeq'] = 0       # NA
#     perf_metrics['NumOfBatches'] = args.epochs * total_train_steps
#     perf_metrics['WeightUpdateSteps'] = args.epochs * total_train_steps
#     perf_metrics['Round'] = 0
#     perf_metrics['GradAccSteps'] = 0

#     perf_metrics_duration = datetime.datetime.now() - perf_metrics_start_time
#     perf_metrics['AvgTimePerBatch'] = \
#         perf_metrics_duration.microseconds / total_train_steps

#     perf_metrics['Throughput'] = \
#         args.batch_size * total_train_steps / perf_metrics_duration.seconds

#     perf_metrics['StabilizedThroughput'] = 0    # TODO
#     perf_metrics['EndToEndThroughput'] = 0      # TODO
#     perf_metrics['TotalTime'] = perf_metrics_duration.seconds

#     perf_metrics['AvgCPU'] = 0                  # TODO
#     perf_metrics['Memory'] = 0                  # TODO
#     perf_metrics['RunConfig'] = 'na'
#     perf_metrics['Time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#     log_perf_metrics(perf_metrics, args.perf_mysql_server_name, args.perf_power_bi_user_name,
#         args.perf_power_bi_password, args.perf_power_bi_database, args.perf_repo_path)

def main():
    # 1. Basic setup
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--pytorch-only', action='store_true', default=False,
                        help='disables ONNX Runtime training')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--view-graphs', action='store_true', default=False,
                        help='views forward and backward graphs')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--epochs', type=int, default=4, metavar='N',
                        help='number of epochs to train (default: 4)')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=40, metavar='N',
                        help='how many batches to wait before logging training status (default: 40)')
    parser.add_argument('--train-steps', type=int, default=-1, metavar='N',
                        help='number of steps to train. Set -1 to run through whole dataset (default: -1)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='WARNING',
                        help='Log level (default: WARNING)')
    parser.add_argument('--num-hidden-layers', type=int, default=1, metavar='H',
                        help='Number of hidden layers for the BERT model. A vanila BERT has 12 hidden layers (default: 1)')
    parser.add_argument('--data-dir', type=str, default='./cola_public/raw',
                        help='Path to the bert data directory')

    # parser.add_argument('--log_perf_metrics', action='store_true', default=False,
    #                     help='whether to log perf data')
    # parser.add_argument('--perf_mysql_server_name', type=str, help='perfmance mysql name')
    # parser.add_argument('--perf_power_bi_user_name', type=str, help='perfmance power BI account user name')
    # parser.add_argument('--perf_power_bi_password', type=str, help='perfmance power BI account password')
    # parser.add_argument('--perf_power_bi_database', type=str, help='perfmance database')
    # parser.add_argument('--perf_repo_path', type=str, help='path of ort source')

    args = parser.parse_args()

    # Device (CPU vs CUDA)
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    # Set log level
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.log_level)
    logging.basicConfig(level=numeric_level)

    # 2. Dataloader
    train_dataloader, validation_dataloader = load_dataset(args)

    # 3. Modeling
    # Load BertForSequenceClassification, the pretrained BERT model with a single
    # linear classification layer on top.
    config = AutoConfig.from_pretrained(
            "bert-base-uncased",
            num_labels=2,
            num_hidden_layers=args.num_hidden_layers,
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
    )
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
        config=config,
    )

    if not args.pytorch_only:
        model = ORTModule(model)

    # TODO: change it to False to stop saving ONNX models
    model._save_onnx = True
    model._save_onnx_prefix = 'BertForSequenceClassification'

    # Tell pytorch to run this model on the GPU.
    if torch.cuda.is_available() and not args.no_cuda:
        model.cuda()

    # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
    optimizer = AdamW(model.parameters(),
                    lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                    eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                    )

    # Authors recommend between 2 and 4 epochs
    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * args.epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)
    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    onnxruntime.set_seed(args.seed)
    if torch.cuda.is_available() and not args.no_cuda:
        torch.cuda.manual_seed_all(args.seed)

    # perf_metrics_start_time = datetime.datetime.now()

    # 4. Train loop (fine-tune)
    total_training_time, total_test_time, epoch_0_training, validation_accuracy = 0, 0, 0, 0
    for epoch_i in range(0, args.epochs):
        total_training_time += train(model, optimizer, scheduler, train_dataloader, epoch_i, device, args)
        if not args.pytorch_only and epoch_i == 0:
            epoch_0_training = total_training_time
        test_time, validation_accuracy = test(model, validation_dataloader, device, args)
        total_test_time += test_time

    assert validation_accuracy > 0.5

    print('\n======== Global stats ========')
    if not args.pytorch_only:
        estimated_export = 0
        if args.epochs > 1:
            estimated_export = epoch_0_training - (total_training_time - epoch_0_training)/(args.epochs-1)
            print("  Estimated ONNX export took:               {:.4f}s".format(estimated_export))
        else:
            print("  Estimated ONNX export took:               Estimate available when epochs > 1 only")
        print("  Accumulated training without export took: {:.4f}s".format(total_training_time - estimated_export))
    print("  Accumulated training took:                {:.4f}s".format(total_training_time))
    print("  Accumulated validation took:              {:.4f}s".format(total_test_time))

    # if args.log_perf_metrics:
    #     bert_classifier_log_perf_metrics(args, train_dataloader, perf_metrics_start_time)

if __name__ == '__main__':
    main()