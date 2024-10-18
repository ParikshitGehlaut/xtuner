# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) S2S_transformer. All rights reserved.

"""
Acknowledge: LLaVA: Visual Instruction Tuning
    (https://llava-vl.github.io/)

            LLaST: Speech to Text Model
    (https://github.com/openaudiolab/LLaST)

"""
import math
import os.path as osp
import warnings
from collections import OrderedDict

import torch
from torch import nn
from accelerate import init_empty_weights
from mmengine import print_log
from mmengine.config import Config, ConfigDict
from mmengine.model import BaseModel
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import (AddedToken, AutoConfig, CLIPImageProcessor,
                          CLIPVisionModel, LlamaForCausalLM,
                          LlamaTokenizerFast, LlavaConfig,
                          LlavaForConditionalGeneration, LlavaProcessor)
from transformers import PretrainedConfig, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.integrations import is_deepspeed_zero3_enabled

from xtuner.registry import BUILDER
from xtuner.dataset.s2s_transformer import prepare_inputs_labels_for_S2S_Transformer
from xtuner.utils import DEFAULT_IMAGE_TOKEN
from .modules import ProjectorConfig, ProjectorModel, dispatch_modules
from .modules.dispatch import SUPPORT_FLASH1, SUPPORT_FLASH2
from .utils import (LoadWoInit, find_all_linear_names,
                    get_peft_model_state_dict, guess_load_checkpoint,
                    make_inputs_require_grad,
                    prepare_inputs_labels_for_multimodal, traverse_dict)


class AudioProjectorConfig(PretrainedConfig):
    model_type = 'projector'
    _auto_class = 'AutoConfig'

    def __init__(self, audio_hidden_size=4096, llm_hidden_size=4096, depth=2, hidden_act='gelu', bias=True, **kwargs):
        self.audio_hidden_size = audio_hidden_size # size of the hidden layer in the audio encoder
        self.llm_hidden_size = llm_hidden_size # size of the hidden layer in the llm
        self.depth = depth # number of layers in projector
        self.hidden_act = hidden_act
        self.bias = bias
        super().__init__(**kwargs)

class AudioEncoder(PreTrainedModel):
    _auto_class = 'AutoModel'
    config_class = AudioProjectorConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True

    def __init__(self, config: AudioProjectorConfig) -> None:
        super().__init__(config)
        self.gradient_checkpointing = False
        print('*' * 30)
        print(config.audio_hidden_size, config.llm_hidden_size)
        modules = [nn.Linear(config.audio_hidden_size, config.llm_hidden_size)]
        for _ in range(1, config.depth):
            modules.append(ACT2FN[config.hidden_act])
            modules.append(nn.Linear(config.llm_hidden_size, config.llm_hidden_size, bias=config.bias))
        self.model = nn.Sequential(*modules)

    # This function enables gradient tracking 
    def enable_input_require_grads(self):
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        self.model.register_forward_hook(make_inputs_require_grad)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, AudioProjectorConfig):
            module.gradient_checkpointing = value

    def forward(self, x):
        if self.gradient_checkpointing and self.training:
            layer_outputs = torch.utils.checkpoint.checkpoint(self.model, x)
        else:
            layer_outputs = self.model(x)
        return layer_outputs
    
class S2S_Transformer_Model(BaseModel):
    def __init__(self, llm, speech_encoder, freeze_llm=False, freeze_speech_encoder=False, speech_select_layer=-1,
                 pretrained_pth=None, projector_depth=2, llm_lora=None, speech_encoder_lora=None,
                 use_activation_checkpointing=True):
        # llm --> language model to use
        # speech_encoder --> encoder for audio data
        # freeze_llm & freeze_speech_encoder --> to stop training of llm and speech encoder
        super().__init__()
        self.freeze_llm = freeze_llm
        self.freeze_speech_encoder = freeze_speech_encoder
        # LoadWoInit --> A utility that loads the LLM and speech encoder without initializing them
        with LoadWoInit():
            self.llm = self._build_from_cfg_or_module(llm)
            self.speech_encoder = self._build_from_cfg_or_module(speech_encoder)

        self.llm.config.use_cache = False
        dispatch_modules(self.llm)

        # create config class and intialize AudioEncoder
        projector_config = AudioProjectorConfig(
            audio_hidden_size=self.speech_encoder.config.hidden_size,
            llm_hidden_size=self.llm.config.hidden_size,
            depth=projector_depth)
        
        self.projector = AudioEncoder(projector_config).to(
            self.speech_encoder.dtype)
        

        ##########################  LoRA Fine Tuning ##########################
        
        if self.freeze_llm:
            self.llm.requires_grad_(False)
        if self.freeze_speech_encoder:
            self.speech_encoder.requires_grad_(False)

        if use_activation_checkpointing:
            # For backward compatibility
            if hasattr(self.llm, 'enable_input_require_grads'):
                self.llm.enable_input_require_grads()
            else:
                self.llm.get_input_embeddings().register_forward_hook(
                    make_inputs_require_grad)
            if hasattr(self.speech_encoder, 'enable_input_require_grads'):
                self.speech_encoder.enable_input_require_grads()
            else:
                self.speech_encoder.get_input_embeddings(
                ).register_forward_hook(make_inputs_require_grad)
            self.projector.enable_input_require_grads()

            # enable gradient (activation) checkpointing for memory efficiency
            self.gradient_checkpointing_enable()

        self.use_llm_lora = llm_lora is not None
        self.use_speech_encoder_lora = speech_encoder_lora is not None

        if self.use_llm_lora:
            self._prepare_llm_for_lora(llm_lora, use_activation_checkpointing)
        if self.use_speech_encoder_lora:
            self._prepare_speech_encoder_for_lora(speech_encoder_lora, use_activation_checkpointing)

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)

            out_str = self.load_state_dict(pretrained_state_dict, strict=False)
            assert len(out_str.unexpected_keys) == 0, out_str.unexpected_keys
            print(f'Load pretrained weight from {pretrained_pth}')

        self.speech_select_layer = speech_select_layer

        self._is_init = True

    def _parse_lora_config(self, lora_config):
        if isinstance(lora_config, dict) or isinstance(
                lora_config, Config) or isinstance(lora_config, ConfigDict):
            lora_config = BUILDER.build(lora_config)
        return lora_config

    def gradient_checkpointing_enable(self):
        self.activation_checkpointing_enable()

    def activation_checkpointing_enable(self):
        self.llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False})
        self.speech_encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False})
        self.projector.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False})

    def gradient_checkpointing_disable(self):
        self.activation_checkpointing_disable()

    def activation_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()
        self.speech_encoder.gradient_checkpointing_disable()
        self.projector.gradient_checkpointing_disable()

    def init_weights(self):
        pass

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        to_return = OrderedDict()
        # Step 1. speech_encoder
        if self.use_speech_encoder_lora:
            to_return.update(
                get_peft_model_state_dict(
                    self.speech_encoder, state_dict=state_dict))
        elif not self.freeze_speech_encoder:
            to_return.update({
                k: v
                for k, v in state_dict.items() if 'speech_encoder.' in k
            })
        # Step 2. LLM
        if self.use_llm_lora:
            to_return.update(
                get_peft_model_state_dict(self.llm, state_dict=state_dict))
        elif not self.freeze_llm:
            to_return.update(
                {k: v
                 for k, v in state_dict.items() if 'llm.' in k})
        # Step 3. Projector
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'projector.' in k})
        return to_return

    def _build_from_cfg_or_module(self, cfg_or_mod):
        if isinstance(cfg_or_mod, nn.Module):
            return cfg_or_mod
        elif isinstance(cfg_or_mod, dict):
            traverse_dict(cfg_or_mod)
            return BUILDER.build(cfg_or_mod)
        else:
            raise NotImplementedError

    def _prepare_llm_for_lora(self,
                              lora_config,
                              use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        self.llm = prepare_model_for_kbit_training(
            self.llm, use_activation_checkpointing)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.llm)
            lora_config.target_modules = modules
        self.llm = get_peft_model(self.llm, lora_config)

    def _prepare_speech_encoder_for_lora(self,
                                         lora_config,
                                         use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.speech_encoder)
            lora_config.target_modules = modules
        self.speech_encoder = get_peft_model(self.speech_encoder, lora_config)

    def forward(self, data, data_samples=None, mode='loss'):
        # convert audio tokens into same data type as speech encoder
        if 'audio_tokens' in data:
            data['audio_tokens'] = data['audio_tokens'].to(
                self.speech_encoder.encoder.conv1.weight.dtype)
            batch_size = data['audio_tokens'].shape[0]
            decoder_input_ids = torch.tensor([
                [1] * batch_size
            ]) * self.speech_encoder.config.decoder_start_token_id
            # audio data is passed to speech encoder to generate hidden states 
            audio_outputs = self.speech_encoder(
                data['audio_tokens'],
                decoder_input_ids=decoder_input_ids.to(
                    data['audio_tokens'].device),
                output_hidden_states=True).encoder_last_hidden_state

            audio_outputs = audio_outputs[:, :max(data['audio_lens']), :]
            # projector or speech adapter transforms the encoder’s hidden states into a form usable by the LLM.
            audio_tokens = self.projector(audio_outputs)
            data['audio_tokens'] = audio_tokens
            # processed data is prepared for LLM.
            data = prepare_inputs_labels_for_S2S_Transformer(llm=self.llm, **data)

        if mode == 'loss':
            return self.compute_loss(data, data_samples)
        elif mode == 'predict':
            return self.predict(data, data_samples)
        elif mode == 'tensor':
            return self._forward(data, data_samples)
        else:
            raise NotImplementedError

    def _forward(self, data, data_samples=None):

        outputs = self.llm(**data)

        return outputs

    def predict(self, data, data_samples=None):
        outputs = self.llm(**data)
        logits_dict = [{'logits': logits} for logits in outputs.logits]
        return logits_dict

    def compute_loss(self, data, data_samples=None):
        outputs = self.llm(**data)
        loss_dict = {'loss': outputs.loss}
        return loss_dict

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.llm, name)