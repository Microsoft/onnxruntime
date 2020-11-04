import copy
from functools import partial
import inspect
import math
from numpy.testing import assert_allclose
import onnx
import os
import pytest
import torch
import subprocess

import onnxruntime
from onnxruntime.capi.ort_trainer import IODescription as Legacy_IODescription,\
                                         ModelDescription as Legacy_ModelDescription,\
                                         LossScaler as Legacy_LossScaler,\
                                         ORTTrainer as Legacy_ORTTrainer
from onnxruntime.training import _utils, amp, checkpoint, optim, orttrainer, TrainStepInfo,\
                                      model_desc_validation as md_val,\
                                      orttrainer_options as orttrainer_options

import _test_commons, _test_helpers

from orttrainer_bert_toy_onnx import *

###############################################################################
# Testing starts here #########################################################
###############################################################################


@pytest.mark.parametrize("dynamic_shape", [
    (True),
    (False)
])
def testToyBERTModelBasicTraining(dynamic_shape):
    model_desc = bert_model_description(dynamic_shape)
    model = load_bert_onnx_model()

    optim_config = optim.LambConfig()
    opts =  orttrainer.ORTTrainerOptions({})
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)

    for i in range(10):
        sample_input = generate_random_input_from_model_desc(model_desc)
        output = trainer.train_step(*sample_input)
        assert output.shape == torch.Size([])


@pytest.mark.parametrize("expected_losses", [
    ([10.988012313842773, 10.99226188659668, 11.090812683105469, 11.042860984802246, 10.988919258117676,
      11.105875015258789, 10.981894493103027, 11.081543922424316, 10.997451782226562, 11.10739517211914])
])
def testToyBERTDeterministicCheck(expected_losses):
    # Common setup
    train_steps = 10
    device = 'cuda'
    seed = 1
    rtol = 1e-3
    torch.manual_seed(seed)
    onnxruntime.set_seed(seed)

    # Modeling
    model_desc = bert_model_description()
    model = load_bert_onnx_model()
    params = optimizer_parameters(model)
    optim_config = optim.LambConfig()
    opts =  orttrainer.ORTTrainerOptions({
        'debug' : {
            'deterministic_compute': True
        },
        'device': {
            'id': device,
        },
    })
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)

    # Train
    experimental_losses = []
    for i in range(train_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, i)
        experimental_losses.append(trainer.train_step(*sample_input).cpu().item())

    # Check output
    _test_helpers.assert_model_outputs(experimental_losses, expected_losses, rtol=rtol)


@pytest.mark.parametrize("initial_lr, lr_scheduler, expected_learning_rates, expected_losses", [
    (1.0, optim.lr_scheduler.ConstantWarmupLRScheduler,\
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [10.988012313842773, 10.99213981628418, 120.79301452636719, 36.11647033691406, 95.83200073242188,\
         221.2766571044922, 208.40316772460938, 279.5332946777344, 402.46380615234375, 325.79254150390625]),
    (0.5, optim.lr_scheduler.ConstantWarmupLRScheduler,\
        [0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        [10.988012313842773, 10.99213981628418, 52.69743347167969, 19.741533279418945, 83.88340759277344,\
         126.39848327636719, 91.53898620605469, 63.62016296386719, 102.21206665039062, 180.1424560546875]),
    (1.0, optim.lr_scheduler.CosineWarmupLRScheduler,\
        [0.0, 0.9931806517013612, 0.9397368756032445, 0.8386407858128706, 0.7008477123264848, 0.5412896727361662,\
         0.37725725642960045, 0.22652592093878665, 0.10542974530180327, 0.02709137914968268],
        [10.988012313842773, 10.99213981628418, 120.6441650390625, 32.152557373046875, 89.63705444335938,\
         138.8782196044922, 117.57748413085938, 148.01927185058594, 229.60403442382812, 110.2930908203125]),
    (1.0, optim.lr_scheduler.LinearWarmupLRScheduler,\
        [0.0, 0.9473684210526315, 0.8421052631578947, 0.7368421052631579, 0.631578947368421, 0.5263157894736842,\
         0.42105263157894735, 0.3157894736842105, 0.21052631578947367, 0.10526315789473684],
        [10.988012313842773, 10.99213981628418, 112.89633178710938, 31.114538192749023, 80.94029235839844,\
         131.34490966796875, 111.4329605102539, 133.74252319335938, 219.37344360351562, 109.67041015625]),
    (1.0, optim.lr_scheduler.PolyWarmupLRScheduler,\
        [0.0, 0.9473684263157895, 0.8421052789473684, 0.7368421315789474, 0.6315789842105263, 0.5263158368421054,
         0.42105268947368424, 0.31578954210526317, 0.21052639473684212, 0.10526324736842106],
        [10.988012313842773, 10.99213981628418, 112.89633178710938, 31.114538192749023, 80.9402847290039,\
         131.3447265625, 111.43253326416016, 133.7415008544922, 219.37147521972656, 109.66986083984375])
])
def testToyBERTModelLRScheduler(initial_lr, lr_scheduler, expected_learning_rates, expected_losses):
    return # TODO: re-enable after nondeterminism on backend is fixed
    # Common setup
    device = 'cuda'
    total_steps = 10
    seed = 1
    warmup = 0.05
    cycles = 0.5
    power = 1.
    lr_end = 1e-7
    rtol = 1e-3
    torch.manual_seed(seed)
    onnxruntime.set_seed(seed)

    # Setup LR Schedulers
    if lr_scheduler == optim.lr_scheduler.ConstantWarmupLRScheduler or lr_scheduler == optim.lr_scheduler.LinearWarmupLRScheduler:
        lr_scheduler = lr_scheduler(total_steps=total_steps, warmup=warmup)
    elif lr_scheduler == optim.lr_scheduler.CosineWarmupLRScheduler:
        lr_scheduler = lr_scheduler(total_steps=total_steps, warmup=warmup, cycles=cycles)
    elif lr_scheduler == optim.lr_scheduler.PolyWarmupLRScheduler:
        lr_scheduler = lr_scheduler(total_steps=total_steps, warmup=warmup, power=power, lr_end=lr_end)
    else:
        raise RuntimeError("Invalid lr_scheduler")

    # Modeling
    model_desc = bert_model_description()
    model = load_bert_onnx_model()
    optim_config = optim.AdamConfig(lr=initial_lr)
    opts =  orttrainer.ORTTrainerOptions({
        'debug' : {
            'deterministic_compute': True
        },
        'device': {
            'id': device,
        },
        'lr_scheduler' : lr_scheduler
    })
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)

    # Train
    losses = []
    learning_rates = []
    for i in range(total_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, i)
        losses.append(trainer.train_step(*sample_input).cpu().item())
        learning_rates.append(trainer.options.lr_scheduler.get_last_lr()[0])

    # Check output
    _test_helpers.assert_model_outputs(learning_rates, expected_learning_rates, rtol=rtol)
    _test_helpers.assert_model_outputs(losses, expected_losses, rtol=rtol)


@pytest.mark.parametrize("loss_scaler, expected_losses", [
    (None, [10.98803424835205, 10.99240493774414, 11.090575218200684, 11.042827606201172, 10.988829612731934,\
        11.105679512023926, 10.981968879699707, 11.081787109375, 10.997162818908691, 11.107288360595703]),
    (amp.DynamicLossScaler(), [10.98803424835205, 10.99240493774414, 11.090575218200684, 11.042827606201172,\
        10.988829612731934, 11.105679512023926, 10.981969833374023, 11.081744194030762, 10.997139930725098, 11.107272148132324]),
    (CustomLossScaler(), [10.98803424835205, 10.99240493774414, 11.090554237365723, 11.042823791503906, 10.98877239227295,\
        11.105667114257812, 10.981982231140137, 11.081765174865723, 10.997125625610352, 11.107298851013184])
])
def testToyBERTModelMixedPrecisionLossScaler(loss_scaler, expected_losses):
    # Common setup
    total_steps = 10
    device = 'cuda'
    seed = 1
    rtol = 1e-3
    torch.manual_seed(seed)
    onnxruntime.set_seed(seed)

    # Modeling
    model_desc = bert_model_description()
    model = load_bert_onnx_model()
    optim_config = optim.LambConfig()
    opts =  orttrainer.ORTTrainerOptions({
        'debug' : {
            'deterministic_compute': True
        },
        'device': {
            'id': device,
        },
        'mixed_precision': {
            'enabled': True,
            'loss_scaler': loss_scaler
        }
    })
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)

    # Train
    losses = []
    for i in range(total_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, i)
        losses.append(trainer.train_step(*sample_input).cpu().item())

    # Check output
    _test_helpers.assert_model_outputs(losses, expected_losses, rtol=rtol)


@pytest.mark.parametrize("gradient_accumulation_steps, expected_losses", [
    (1, [10.988012313842773, 10.99226188659668, 11.090812683105469, 11.042860984802246, 10.988919258117676,
        11.105875015258789, 10.981894493103027, 11.081543922424316, 10.997451782226562, 11.10739517211914]),
    (4, [10.988012313842773, 10.99213981628418, 11.090258598327637, 11.039335250854492, 10.986993789672852,
        11.110128402709961, 10.989538192749023, 11.072074890136719, 11.001150131225586, 11.100043296813965]),
    (7, [10.988012313842773, 10.99213981628418, 11.090258598327637, 11.039335250854492, 10.993097305297852,
        11.112862586975098, 10.996183395385742, 11.072013854980469, 11.00184154510498, 11.097928047180176])
])
def testToyBERTModelGradientAccumulation(gradient_accumulation_steps, expected_losses):
    # Common setup
    total_steps = 10
    device = "cuda"
    seed = 1
    rtol = 1e-3
    torch.manual_seed(seed)
    onnxruntime.set_seed(seed)

    # Modeling
    model_desc = bert_model_description()
    model = load_bert_onnx_model()
    optim_config = optim.LambConfig()
    opts =  orttrainer.ORTTrainerOptions({
        'debug' : {
            'deterministic_compute': True
        },
        'device': {
            'id': device,
        },
        'batch' : {
            'gradient_accumulation_steps' : gradient_accumulation_steps
        },
    })
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)

    # Train
    losses = []
    for i in range(total_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, i)
        losses.append(trainer.train_step(*sample_input).cpu().item())

    # Check output
    _test_helpers.assert_model_outputs(losses, expected_losses, rtol=rtol)

def testToyBertCheckpointBasic():
    # Common setup
    seed = 1
    torch.manual_seed(seed)
    onnxruntime.set_seed(seed)
    optim_config = optim.LambConfig()
    opts = orttrainer.ORTTrainerOptions({'debug' : {'deterministic_compute': True}})

    # Create ORTTrainer and save initial state in a dict
    model = load_bert_onnx_model()
    model_desc = bert_model_description()
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)
    sd = checkpoint.experimental_state_dict(trainer)

    ## All initializers must be present in the state_dict
    ##  when the specified model for ORTTRainer is an ONNX model
    for param in trainer._onnx_model.graph.initializer:
        assert param.name in sd

    ## Modify one of the state values and load into ORTTrainer
    sd['bert.encoder.layer.0.attention.output.LayerNorm.weight'] += 10
    checkpoint.experimental_load_state_dict(trainer, sd)

    ## Save a checkpoint
    ckpt_dir = _test_helpers._get_name("ort_ckpt")
    checkpoint.experimental_save_checkpoint(trainer, ckpt_dir, 'bert_toy_save_test')
    del trainer
    del model

    # Create a new ORTTrainer and load the checkpoint from previous ORTTrainer
    model2 = load_bert_onnx_model()
    model_desc2 = bert_model_description()
    trainer2 = orttrainer.ORTTrainer(model2, model_desc2, optim_config, options=opts)
    checkpoint.experimental_load_checkpoint(trainer2, ckpt_dir, 'bert_toy_save_test')
    loaded_sd = checkpoint.experimental_state_dict(trainer2)

    # Assert whether original state and the one loaded from checkpoint matches
    for k,v in loaded_sd.items():
        assert torch.all(torch.eq(v, sd[k]))

# Test ZeRO checkpoint loading from a distributed Zero stage 1 mixedprecision run to the
# following configs. This test ensures that the weights as well as the optimizer
# state are aggregated and loaded correctly into fp32 and fp16 single-node 
# trainers respectively, for Adam and Lamb optimizers.
@pytest.mark.parametrize("optimizer, mixedprecision_enabled, expected_eval_loss", [
    (optim.LambConfig(), False, [11.011026]),
    (optim.AdamConfig(), False, [10.998348]),
    (optim.LambConfig(), True, [11.011026]),
    (optim.AdamConfig(), True, [10.998348]),
])
def testToyBertCheckpointLoadZero(optimizer, mixedprecision_enabled, expected_eval_loss):
    # Common setup
    rtol = 1e-03
    device = 'cuda'
    seed = 1
    torch.manual_seed(seed)
    onnxruntime.set_seed(seed)
    optim_config = optimizer
    opts = orttrainer.ORTTrainerOptions({'debug' : {'deterministic_compute': True},
                                         'device' : {'id' : device},
                                         'mixed_precision': {
                                                'enabled': mixedprecision_enabled,
                                            },
                                         'distributed' : {'allreduce_post_accumulation' : True}})

    # Create ORTTrainer and save initial state in a dict
    model = load_bert_onnx_model()
    model_desc = bert_model_description()
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)

    ckpt_dir = _test_helpers._get_name("ort_ckpt")
    ckpt_prefix = _test_helpers._get_bert_ckpt_prefix(optimizer.name)
    checkpoint_files = sorted(checkpoint._list_checkpoint_files(ckpt_dir, ckpt_prefix))

    ########################################
    # Test the aggregation code individually
    ########################################
    ckpt_agg = checkpoint._CombineZeroCheckpoint(checkpoint_files)
    aggregate_state_dict = ckpt_agg.aggregate_checkpoints()

    expected_state_name = 'aggregated_' + ckpt_prefix + '.pt'
    expected_state_dict = torch.load(os.path.join(ckpt_dir, expected_state_name), map_location=torch.device("cpu"))

    assert expected_state_dict.keys() == aggregate_state_dict.keys()

    for k,v in aggregate_state_dict.items():
        assert_allclose(v, expected_state_dict[k])

    ##########################################################
    # Test the experimental_load_checkpoint functionality that 
    # does the aggregation under the hood
    ##########################################################
    checkpoint.experimental_load_checkpoint(trainer, ckpt_dir, ckpt_prefix)

    # input values
    input_ids = torch.tensor([[26598],[21379],[19922],[ 5219],[ 5644],[20559],[23777],[25672],[22969],[16824],[16822],[635],[27399],[20647],[18519],[15546]], device=device)
    segment_ids = torch.tensor([[0],[1],[0],[1],[0],[0],[1],[0],[0],[1],[1],[0],[0],[1],[1],[1]], device=device)
    input_mask = torch.tensor([[0],[0],[0],[0],[1],[1],[1],[0],[1],[1],[0],[0],[0],[1],[0],[0]], device=device)
    masked_lm_labels = torch.tensor([[25496],[16184],[11005],[16228],[14884],[21660],[ 8678],[23083],[ 4027],[ 8397],[11921],[ 1333],[26482],[ 1666],[17925],[27978]], device=device)
    next_sentence_labels = torch.tensor([0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0], device=device)

    # Actual values
    actual_eval_loss = trainer.eval_step(input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels)
    actual_eval_loss = actual_eval_loss.cpu().numpy().item(0)

    checkpoint_files = sorted(checkpoint._list_checkpoint_files(ckpt_dir, ckpt_prefix))
    loaded_state_dict = checkpoint.experimental_state_dict(trainer)

    # check if the loaded state dict is as expected
    if mixedprecision_enabled:
        assert expected_state_dict.keys() == loaded_state_dict.keys()
    for k,v in loaded_state_dict.items():
        assert_allclose(v, expected_state_dict[k])

    # compare loaded state dict to rank state dicts
    loaded_state_dict = checkpoint._split_state_dict(loaded_state_dict)

    for f in checkpoint_files:
        rank_state_dict = torch.load(f, map_location=torch.device("cpu"))
        rank_state_dict = checkpoint._split_state_dict(rank_state_dict['model'])

        for k,v in rank_state_dict['fp16_param'].items():
            fp32_name = k.split('_fp16')[0]
            assert_allclose(v, loaded_state_dict['fp32_param'][fp32_name], rtol=1e-02, atol=1e-02)
            if mixedprecision_enabled:
                assert_allclose(v, loaded_state_dict['fp16_param'][k])
        
        for k,v in rank_state_dict['optimizer'].items():
            if k in loaded_state_dict['optimizer']:
                assert_allclose(v, loaded_state_dict['optimizer'][k])
            else:
                assert '_view_' in k 

    # Check results
    assert_allclose(expected_eval_loss, actual_eval_loss, rtol=rtol)

def testToyBertCheckpointFrozenWeights():
    # Common setup
    seed = 1
    total_steps = 10
    torch.manual_seed(seed)
    onnxruntime.set_seed(seed)
    opts = orttrainer.ORTTrainerOptions({'debug' : {'deterministic_compute': True},
                                         'utils' : {'frozen_weights' : ['bert.encoder.layer.0.attention.self.value.weight']}})

    # Create ORTTrainer and save initial state in a dict
    model = load_bert_onnx_model()
    model_desc = bert_model_description()
    optim_config = optim.LambConfig()
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)

    # Train for a few steps
    for i in range(total_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, seed)
        _ = trainer.train_step(*sample_input)
    sample_input = generate_random_input_from_model_desc(model_desc, seed + total_steps + 1)
    # Evaluate once to get a base loss
    loss = trainer.eval_step(*sample_input)
    # Save checkpoint
    state_dict = checkpoint.experimental_state_dict(trainer)

    # Load previous state into another instance of ORTTrainer
    model2 = load_bert_onnx_model()
    model_desc2 = bert_model_description()
    optim_config2 = optim.LambConfig()
    trainer2 = orttrainer.ORTTrainer(model2, model_desc2, optim_config2, options=opts)
    checkpoint.experimental_load_state_dict(trainer2, state_dict)
    # Evaluate once to get a base loss
    ckpt_loss = trainer2.eval_step(*sample_input)

    # Must match as both trainers have the same dict state
    assert_allclose(loss.cpu(), ckpt_loss.cpu())
    loaded_state_dict = checkpoint.experimental_state_dict(trainer2)
    assert state_dict.keys() == loaded_state_dict.keys()

@pytest.mark.parametrize("model_params", [
    (['bert.embeddings.LayerNorm.bias']),
    (['bert.embeddings.LayerNorm.bias',
      'bert.embeddings.LayerNorm.weight',
      'bert.encoder.layer.0.attention.output.LayerNorm.bias']),
])
def testORTTrainerFrozenWeights(model_params):
    device = 'cuda'
    total_steps = 10
    seed = 1

    # EXPERIMENTAL API
    model_desc = bert_model_description()
    model = load_bert_onnx_model()

    optim_config = optim.LambConfig()
    # Setup ORTTrainer WITHOUT frozen weights
    opts_dict = {
        'debug' : {
            'deterministic_compute': True
        },
        'device': {
            'id': device,
        },
    }
    opts =  orttrainer.ORTTrainerOptions(opts_dict)

    torch.manual_seed(seed)
    onnxruntime.set_seed(seed)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)

    for i in range(total_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, i)
        trainer.train_step(*sample_input)

    # All model_params must be in the session state
    assert trainer._onnx_model is not None
    session_state = trainer._training_session.get_state()
    assert all([param in session_state for param in model_params])

    # Setup ORTTrainer WITH frozen weights
    opts_dict.update({'utils' : {'frozen_weights' : model_params}})
    opts =  orttrainer.ORTTrainerOptions(opts_dict)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)

    for i in range(total_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, i)
        trainer.train_step(*sample_input)

    # All model_params CANNOT be in the session state
    assert trainer._onnx_model is not None
    session_state = trainer._training_session.get_state()
    assert not any([param in session_state for param in model_params])

def testToyBERTSaveAsONNX():
    device = 'cuda'
    onnx_file_name = '_____temp_toy_bert_onnx_model.onnx'
    if os.path.exists(onnx_file_name):
        os.remove(onnx_file_name)
    assert not os.path.exists(onnx_file_name)

    # Load trainer
    model_desc = bert_model_description()
    model = load_bert_onnx_model()

    optim_config = optim.LambConfig()
    opts =  orttrainer.ORTTrainerOptions({
        'debug' : {
            'deterministic_compute': True
        },
        'device': {
            'id': device,
        },
    })

    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)

    trainer.save_as_onnx(onnx_file_name)
    assert os.path.exists(onnx_file_name)

    with open(onnx_file_name, "rb") as f:
        bin_str = f.read()
        reload_onnx_model = onnx.load_model_from_string(bin_str)
    os.remove(onnx_file_name)

    # Create a new trainer from persisted ONNX model and compare with original ONNX model
    trainer_from_onnx = orttrainer.ORTTrainer(reload_onnx_model, model_desc, optim_config, options=opts)
    assert trainer_from_onnx._onnx_model is not None
    assert (id(trainer_from_onnx._onnx_model) != id(trainer._onnx_model))
    for initializer, loaded_initializer in zip(trainer._onnx_model.graph.initializer, trainer_from_onnx._onnx_model.graph.initializer):
        assert initializer.name == loaded_initializer.name
    assert (onnx.helper.printable_graph(trainer_from_onnx._onnx_model.graph) == onnx.helper.printable_graph(trainer._onnx_model.graph))
    _test_helpers.assert_onnx_weights(trainer, trainer_from_onnx)


###############################################################################
# Temporary tests comparing Legacy vs Experimental ORTTrainer APIs ############
###############################################################################
@pytest.mark.parametrize("optimizer_config", [
    (optim.AdamConfig),
#    (optim.LambConfig), # TODO: re-enable after nondeterminism on backend is fixed
    (optim.SGDConfig)
])
def testToyBERTModelLegacyExperimentalBasicTraining(optimizer_config):
    # Common setup
    train_steps = 512

    device = 'cuda'
    seed = 1
    torch.manual_seed(seed)
    onnxruntime.set_seed(seed)

    # EXPERIMENTAL API
    model_desc = bert_model_description()
    model = load_bert_onnx_model()
    opts =  orttrainer.ORTTrainerOptions({
        'debug' : {
            'deterministic_compute': True
        },
        'device': {
            'id': device,
        },
    })
    optim_config = optimizer_config(lr=0.01)
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)
    experimental_losses = []
    for i in range(train_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, i)
        experimental_losses.append(trainer.train_step(*sample_input).cpu().item())

    # LEGACY IMPLEMENTATION
    torch.manual_seed(seed)
    onnxruntime.set_seed(seed)

    if optimizer_config == optim.AdamConfig:
        legacy_optimizer = 'AdamOptimizer'
    elif optimizer_config == optim.LambConfig:
        legacy_optimizer = 'LambOptimizer'
    elif optimizer_config == optim.SGDConfig:
        legacy_optimizer = 'SGDOptimizer'
    else:
        raise RuntimeError("Invalid optimizer_config")

    device = torch.device(device)
    model = load_bert_onnx_model()
    legacy_model_desc, learning_rate_description, learning_rate = legacy_model_params(lr=optim_config.lr)
    legacy_trainer = Legacy_ORTTrainer(model, None, legacy_model_desc, legacy_optimizer,
                       None,
                       learning_rate_description,
                       device,
                       _use_deterministic_compute=True)
    legacy_losses = []
    for i in range(train_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, i)
        leg_loss = legacy_trainer.train_step(*sample_input, learning_rate)
        legacy_losses.append(leg_loss.cpu().item())

    # Check results
    _test_helpers.assert_model_outputs(experimental_losses, legacy_losses, True)


@pytest.mark.parametrize("initial_lr, lr_scheduler, legacy_lr_scheduler", [
    (1.0, optim.lr_scheduler.ConstantWarmupLRScheduler, _test_commons.legacy_constant_lr_scheduler),
    (0.5, optim.lr_scheduler.ConstantWarmupLRScheduler, _test_commons.legacy_constant_lr_scheduler),
    (1.0, optim.lr_scheduler.CosineWarmupLRScheduler, _test_commons.legacy_cosine_lr_scheduler),
    (1.0, optim.lr_scheduler.LinearWarmupLRScheduler, _test_commons.legacy_linear_lr_scheduler),
    (1.0, optim.lr_scheduler.PolyWarmupLRScheduler, _test_commons.legacy_poly_lr_scheduler),
])
def testToyBERTModelLegacyExperimentalLRScheduler(initial_lr, lr_scheduler, legacy_lr_scheduler):
    ############################################################################
    # These tests require hard-coded values for 'total_steps' and 'initial_lr' #
    ############################################################################

    # Common setup
    total_steps = 128
    device = 'cuda'
    seed = 1
    warmup = 0.05
    cycles = 0.5
    power = 1.
    lr_end = 1e-7

    # Setup both Experimental and Legacy LR Schedulers before the experimental loop
    if legacy_lr_scheduler == _test_commons.legacy_constant_lr_scheduler or legacy_lr_scheduler == _test_commons.legacy_linear_lr_scheduler:
        legacy_lr_scheduler = partial(legacy_lr_scheduler, initial_lr=initial_lr, total_steps=total_steps, warmup=warmup)
    elif legacy_lr_scheduler == _test_commons.legacy_cosine_lr_scheduler:
        legacy_lr_scheduler = partial(legacy_lr_scheduler, initial_lr=initial_lr, total_steps=total_steps, warmup=warmup, cycles=cycles)
    elif legacy_lr_scheduler == _test_commons.legacy_poly_lr_scheduler:
        legacy_lr_scheduler = partial(legacy_lr_scheduler, initial_lr=initial_lr, total_steps=total_steps, warmup=warmup, power=power, lr_end=lr_end)
    else:
        raise RuntimeError("Invalid legacy_lr_scheduler")
    if lr_scheduler == optim.lr_scheduler.ConstantWarmupLRScheduler or lr_scheduler == optim.lr_scheduler.LinearWarmupLRScheduler:
        lr_scheduler = lr_scheduler(total_steps=total_steps, warmup=warmup)
    elif lr_scheduler == optim.lr_scheduler.CosineWarmupLRScheduler:
        lr_scheduler = lr_scheduler(total_steps=total_steps, warmup=warmup, cycles=cycles)
    elif lr_scheduler == optim.lr_scheduler.PolyWarmupLRScheduler:
        lr_scheduler = lr_scheduler(total_steps=total_steps, warmup=warmup, power=power, lr_end=lr_end)
    else:
        raise RuntimeError("Invalid lr_scheduler")


    # EXPERIMENTAL API
    model_desc = bert_model_description()
    model = load_bert_onnx_model()
    torch.manual_seed(seed)
    onnxruntime.set_seed(seed)
    optim_config = optim.AdamConfig(lr=initial_lr)
    opts =  orttrainer.ORTTrainerOptions({
        'debug' : {
            'deterministic_compute': True
        },
        'device': {
            'id': device,
        },
        'lr_scheduler' : lr_scheduler
    })
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)
    experimental_losses = []
    for i in range(total_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, i)
        experimental_losses.append(trainer.train_step(*sample_input).cpu().item())
        assert_allclose(trainer.options.lr_scheduler.get_last_lr()[0], legacy_lr_scheduler(i))

    # LEGACY IMPLEMENTATION
    torch.manual_seed(seed)
    onnxruntime.set_seed(seed)
    device = torch.device(device)
    model = load_bert_onnx_model()
    legacy_model_desc, learning_rate_description, learning_rate = legacy_model_params(initial_lr)
    legacy_trainer = Legacy_ORTTrainer(model, None, legacy_model_desc, "AdamOptimizer",
                       None,
                       learning_rate_description,
                       device,
                       _use_deterministic_compute=True,
                       get_lr_this_step=legacy_lr_scheduler)
    legacy_losses = []
    for i in range(total_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, i)
        leg_loss = legacy_trainer.train_step(*sample_input)
        legacy_losses.append(leg_loss.cpu().item())

    # Check results
    _test_helpers.assert_model_outputs(experimental_losses, legacy_losses)


@pytest.mark.parametrize("loss_scaler, legacy_loss_scaler", [
    (None, Legacy_LossScaler("ort_test_input_loss_scaler", True)),
    (amp.DynamicLossScaler(), Legacy_LossScaler("ort_test_input_loss_scaler", True)),
    (CustomLossScaler(), LegacyCustomLossScaler())
])
def testToyBERTModelMixedPrecisionLossScalerLegacyExperimental(loss_scaler, legacy_loss_scaler):
    # Common setup
    total_steps = 128
    device = "cuda"
    seed = 1

    # EXPERIMENTAL IMPLEMENTATION
    torch.manual_seed(seed)
    onnxruntime.set_seed(seed)
    model_desc = bert_model_description()
    model = load_bert_onnx_model()
    optim_config = optim.AdamConfig(lr=0.001)
    opts =  orttrainer.ORTTrainerOptions({
        'debug' : {
            'deterministic_compute': True
        },
        'device': {
            'id': device,
        },
        'mixed_precision': {
            'enabled': True,
            'loss_scaler': loss_scaler
        }
    })
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)
    experimental_losses = []
    for i in range(total_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, i)
        experimental_losses.append(trainer.train_step(*sample_input).cpu().item())

    # LEGACY IMPLEMENTATION
    torch.manual_seed(seed)
    onnxruntime.set_seed(seed)
    device = torch.device(device)
    model = load_bert_onnx_model()
    legacy_model_desc, learning_rate_description, learning_rate = legacy_model_params(optim_config.lr)
    legacy_trainer = Legacy_ORTTrainer(model, None, legacy_model_desc, "AdamOptimizer",
                       None,
                       learning_rate_description,
                       device,
                       _use_deterministic_compute=True,
                       use_mixed_precision=True,
                       loss_scaler = legacy_loss_scaler)
    legacy_losses = []
    for i in range(total_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, i)
        leg_loss = legacy_trainer.train_step(*sample_input, learning_rate)
        legacy_losses.append(leg_loss.cpu().item())

    # Check results
    _test_helpers.assert_model_outputs(experimental_losses, legacy_losses)


@pytest.mark.parametrize("gradient_accumulation_steps", [
    (1),
    (4),
    (7)
])
def testToyBERTModelGradientAccumulationLegacyExperimental(gradient_accumulation_steps):
    # Common setup
    total_steps = 128
    device = "cuda"
    seed = 1

    # EXPERIMENTAL IMPLEMENTATION
    torch.manual_seed(seed)
    onnxruntime.set_seed(seed)
    model_desc = bert_model_description()
    model = load_bert_onnx_model()
    optim_config = optim.AdamConfig()
    opts =  orttrainer.ORTTrainerOptions({
        'debug' : {
            'deterministic_compute': True
        },
        'device': {
            'id': device,
        },
        'batch' : {
            'gradient_accumulation_steps' : gradient_accumulation_steps
        },
    })
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)
    experimental_losses = []
    for i in range(total_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, i)
        loss = trainer.train_step(*sample_input)
        experimental_losses.append(loss.cpu().item())

    # LEGACY IMPLEMENTATION
    torch.manual_seed(seed)
    onnxruntime.set_seed(seed)
    device = torch.device(device)
    model = load_bert_onnx_model()
    legacy_model_desc, learning_rate_description, learning_rate = legacy_model_params(optim_config.lr)
    legacy_trainer = Legacy_ORTTrainer(model, None, legacy_model_desc, "AdamOptimizer",
                       None,
                       learning_rate_description,
                       device,
                       _use_deterministic_compute = True,
                       gradient_accumulation_steps = gradient_accumulation_steps)
    legacy_losses = []
    for i in range(total_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, i)
        leg_loss = legacy_trainer.train_step(*sample_input, learning_rate)
        legacy_losses.append(leg_loss.cpu().item())

    # Check results
    _test_helpers.assert_model_outputs(experimental_losses, legacy_losses)

@pytest.mark.parametrize("params, legacy_optim_map", [
    # Change the hyper parameters for all parameters
    ([], legacy_optim_params_a),
    # Change the hyperparameters for a subset of hardcoded parameters
    ([{'params':['bert.embeddings.LayerNorm.bias', 'bert.embeddings.LayerNorm.weight'], "alpha": 0.9,
        "beta": 0.999, "lambda_coef": 0.0, "epsilon": 1e-6, "do_bias_correction":False}], legacy_optim_params_b),
    # Change the hyperparameters for a generated set of paramers
    (optimizer_parameters(load_bert_onnx_model()), legacy_optim_params_c)
])
def testToyBERTModelLegacyExperimentalCustomOptimParameters(params, legacy_optim_map):
    # Common setup
    total_steps = 128
    device = "cuda"
    seed = 1

    # EXPERIMENTAL API
    torch.manual_seed(seed)
    onnxruntime.set_seed(seed)
    model_desc = bert_model_description()
    model = load_bert_onnx_model()

    optim_config = optim.AdamConfig(params, alpha= 0.9, beta= 0.999, lambda_coef= 0.01, epsilon= 1e-6, do_bias_correction=False)
    opts =  orttrainer.ORTTrainerOptions({
        'debug' : {
            'deterministic_compute': True
        },
        'device': {
            'id': device,
        },
    })
    trainer = orttrainer.ORTTrainer(model, model_desc, optim_config, options=opts)

    experimental_losses = []
    for i in range(total_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, i)
        experimental_losses.append(trainer.train_step(*sample_input).cpu().item())

    # LEGACY IMPLEMENTATION
    torch.manual_seed(seed)
    onnxruntime.set_seed(seed)
    device = torch.device(device)
    model = load_bert_onnx_model()
    legacy_model_desc, learning_rate_description, learning_rate = legacy_model_params(trainer.optim_config.lr)

    legacy_trainer = Legacy_ORTTrainer(model, None, legacy_model_desc, "AdamOptimizer",
                                       legacy_optim_map,
                                       learning_rate_description,
                                       device,
                                       _use_deterministic_compute=True)
    legacy_losses = []
    for i in range(total_steps):
        sample_input = generate_random_input_from_model_desc(model_desc, i)
        legacy_sample_input = [*sample_input, learning_rate]
        legacy_losses.append(legacy_trainer.train_step(legacy_sample_input).cpu().item())

     # Check results
    _test_helpers.assert_model_outputs(experimental_losses, legacy_losses)
