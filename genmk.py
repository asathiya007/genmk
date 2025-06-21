from datasets import Dataset, DatasetDict, Features, Image, Value
from diffusers import (
    AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline,
    UNet2DConditionModel)
from diffusers.optimization import get_scheduler
from diffusers.utils import convert_state_dict_to_diffusers
from img_utils import get_formatted_img
import logging
import os
from peft import LoraConfig
from peft.tuners.lora.layer import LoraLayer
from peft.utils import get_peft_model_state_dict
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, \
    GenerationConfig, AutoTokenizer, PretrainedConfig


RESOLUTION = 1024
MAX_CLIP_TOKENS = 77


def _get_img_paths(img_dir):
    img_paths = []
    for root, _, files in os.walk(img_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() == '.png':
                img_paths.append(os.path.join(root, file))
    return img_paths


class GenMK:
    def __init__(self, lora_dir='./genmk_lora'):
        self.base_model_name = 'stabilityai/stable-diffusion-xl-base-1.0'
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.lora_dir = lora_dir

        # get logger
        self.logger = logging.getLogger('GenMK')
        self.logger.setLevel(logging.INFO)

    def train(self, rank, alpha, batch_size, num_epochs,
              train_text_encoders=False, using_ampere_gpu=False,
              text_encoders_rank=None, text_encoders_alpha=None):
        # load multimodal model from Hugging Face
        model_path = 'microsoft/Phi-4-multimodal-instruct'
        self.logger.info(f'Loading {model_path} model...')
        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map=self.device, torch_dtype="auto",
            trust_remote_code=True,
            _attn_implementation='flash_attention_2'
            if (using_ampere_gpu and 'cuda' in str(self.device))
            else 'eager').to(self.device)
        generation_config = GenerationConfig.from_pretrained(model_path)
        self.logger.info('Loaded model')

        # specify tokens used in prompt
        user_prompt = '<|user|>'
        assistant_prompt = '<|assistant|>'
        prompt_suffix = '<|end|>'

        # specify prompt for getting character image decscriptions
        char_img_prompt = f'{user_prompt}<|image_1|>Describe the appearance '\
            + 'of this character in at most 45 words. Do not answer in a '\
            + f'complete sentence.{prompt_suffix}{assistant_prompt}'

        # get image descriptions and format images
        self.logger.info(
            'Generating image descriptions and formatting images...')
        img_paths = _get_img_paths('./mk_char_imgs')
        formatted_imgs = []
        img_descs = []
        img_desc_batch_size = 12
        for i in range(0, len(img_paths), img_desc_batch_size):
            paths = img_paths[i:i+img_desc_batch_size]
            imgs = []
            for path in paths:
                imgs.append(get_formatted_img(path, 224))
            prompts = [char_img_prompt] * len(imgs)

            # pass images to model, get descriptions
            inputs = processor(
                text=prompts, images=imgs, return_tensors='pt').to(self.device)
            generate_ids = model.generate(
                **inputs, max_new_tokens=MAX_CLIP_TOKENS,
                generation_config=generation_config)
            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
            descs = processor.batch_decode(
                generate_ids, skip_special_tokens=True,
                clean_up_tokenization_spaces=False)
            img_descs += descs

            # format images for training
            formatted_imgs += list(map(
                lambda x: get_formatted_img(x, RESOLUTION), paths))
        self.logger.info(
            'Finished generating image descriptions and formatting images')

        # free up memory
        del processor
        del model
        del generation_config
        torch.cuda.empty_cache()

        # prepare training dataset
        data = {'image': formatted_imgs, 'text': img_descs}
        dataset = Dataset.from_dict(
            data, features=Features(
                {'image': Image(), 'text': Value('string')}))
        dataset_dict = DatasetDict({'train': dataset})
        self.logger.info(
            'Prepared dataset for Mortal Kombat character generation')

        '''
        The following code of this method was adapted from this script
        in the Hugging Face diffusers repository.
        https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora_sdxl.py
        '''

        # utility function for getting text encoder class names
        def _import_model_class_from_model_name_or_path(
                subfolder='text_encoder'):
            text_encoder_config = PretrainedConfig.from_pretrained(
                self.base_model_name, subfolder=subfolder)
            model_class = text_encoder_config.architectures[0]

            if model_class == 'CLIPTextModel':
                from transformers import CLIPTextModel

                return CLIPTextModel
            elif model_class == 'CLIPTextModelWithProjection':
                from transformers import CLIPTextModelWithProjection

                return CLIPTextModelWithProjection
            else:
                raise ValueError(f'{model_class} is not supported.')

        # get diffusion model's tokenizers, text encoders, noise scheduler,
        # VAE, and U-Net from Hugging Face
        self.logger.info(f'Loading components of {self.base_model_name}...')
        tokenizer_one = AutoTokenizer.from_pretrained(
            self.base_model_name, subfolder='tokenizer',
            use_fast=False)
        tokenizer_two = AutoTokenizer.from_pretrained(
            self.base_model_name, subfolder='tokenizer_2',
            use_fast=False)
        text_encoder_cls_one = _import_model_class_from_model_name_or_path()
        text_encoder_cls_two = _import_model_class_from_model_name_or_path(
            subfolder='text_encoder_2')
        text_encoder_one = text_encoder_cls_one.from_pretrained(
            self.base_model_name, subfolder='text_encoder')
        text_encoder_two = text_encoder_cls_two.from_pretrained(
            self.base_model_name, subfolder='text_encoder_2')
        noise_scheduler = DDPMScheduler.from_pretrained(
            self.base_model_name, subfolder='scheduler')
        vae = AutoencoderKL.from_pretrained(
            self.base_model_name, subfolder='vae')
        unet = UNet2DConditionModel.from_pretrained(
            self.base_model_name, subfolder='unet')
        self.logger.info(f'Loaded {self.base_model_name} components')

        # disable gradient calculation, since it is only required for LoRA
        # weights
        vae.requires_grad_(False)
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)
        unet.requires_grad_(False)

        # cast to desired data type
        weight_dtype = torch.float32
        unet.to(self.device, dtype=weight_dtype)
        vae.to(self.device, dtype=weight_dtype)
        text_encoder_one.to(self.device, dtype=weight_dtype)
        text_encoder_two.to(self.device, dtype=weight_dtype)

        # add LoRA weights
        unet_lora_config = LoraConfig(
            r=rank, lora_alpha=alpha,
            init_lora_weights='gaussian',
            target_modules=['to_k', 'to_q', 'to_v', 'to_out.0'],
        )
        unet.add_adapter(unet_lora_config)
        if train_text_encoders:
            if text_encoders_rank is None:
                text_encoders_rank = rank
            if text_encoders_alpha is None:
                text_encoders_alpha = alpha
            text_lora_config = LoraConfig(
                r=text_encoders_rank, lora_alpha=text_encoders_alpha,
                init_lora_weights='gaussian',
                target_modules=['q_proj', 'k_proj', 'v_proj', 'out_proj'])
            text_encoder_one.add_adapter(text_lora_config)
            text_encoder_two.add_adapter(text_lora_config)
        self.logger.info('Added LoRA weights')

        # enable TF32 for faster training on Ampere GPUs,
        if torch.cuda.is_available() and using_ampere_gpu:
            torch.backends.cuda.matmul.allow_tf32 = True

        # create optimizer
        lora_params = list(filter(
            lambda p: p.requires_grad, unet.parameters()))
        if train_text_encoders:
            lora_params = (
                lora_params
                + list(filter(
                    lambda p: p.requires_grad, text_encoder_one.parameters()))
                + list(filter(
                    lambda p: p.requires_grad, text_encoder_two.parameters()))
            )
        optimizer = torch.optim.AdamW(lora_params)

        def _tokenize(tokenizer, prompt):
            text_inputs = tokenizer(
                prompt, padding='max_length',
                max_length=tokenizer.model_max_length, truncation=True,
                return_tensors='pt')
            text_input_ids = text_inputs.input_ids
            return text_input_ids

        # function to tokenize image descriptions
        def _tokenize_descs(examples):
            descs = list(examples['text'])
            tokens_one = _tokenize(tokenizer_one, descs)
            tokens_two = _tokenize(tokenizer_two, descs)
            return tokens_one, tokens_two

        # data preprocessing transforms
        train_resize = transforms.Resize(RESOLUTION)
        train_crop = transforms.CenterCrop(RESOLUTION)
        train_transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

        # utility function for preprocessing data
        def _preprocess_train(examples):
            images = [image.convert('RGB') for image in examples['image']]
            original_sizes = []
            all_images = []
            crop_top_lefts = []
            for image in images:
                original_sizes.append((image.height, image.width))
                image = train_resize(image)
                image = train_crop(image)
                crop_top_left = (0, 0)
                crop_top_lefts.append(crop_top_left)
                image = train_transforms(image)
                all_images.append(image)

            examples['original_sizes'] = original_sizes
            examples['crop_top_lefts'] = crop_top_lefts
            examples['pixel_values'] = all_images
            tokens_one, tokens_two = _tokenize_descs(examples)
            examples['input_ids_one'] = tokens_one
            examples['input_ids_two'] = tokens_two
            return examples

        # set the training transforms on the dataset
        train_dataset = dataset_dict['train'].with_transform(
            _preprocess_train, output_all_columns=True)

        # utility function for collating data
        def _collate_fn(examples):
            pixel_values = torch.stack(
                [example['pixel_values'] for example in examples])
            pixel_values = pixel_values.to(
                memory_format=torch.contiguous_format).float()
            original_sizes = [
                example['original_sizes'] for example in examples]
            crop_top_lefts = [
                example['crop_top_lefts'] for example in examples]
            input_ids_one = torch.stack(
                [example['input_ids_one'] for example in examples])
            input_ids_two = torch.stack(
                [example['input_ids_two'] for example in examples])
            result = {
                'pixel_values': pixel_values,
                'input_ids_one': input_ids_one,
                'input_ids_two': input_ids_two,
                'original_sizes': original_sizes,
                'crop_top_lefts': crop_top_lefts
            }
            return result

        # create dataloader
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, shuffle=True, collate_fn=_collate_fn,
            batch_size=batch_size)

        # get learning rate scheduler
        max_train_steps = num_epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(
            'cosine', optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=max_train_steps)

        # Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
        def encode_prompt(text_encoders, tokenizers, prompt,
                          text_input_ids_list=None):
            prompt_embeds_list = []

            for i, text_encoder in enumerate(text_encoders):
                if tokenizers is not None:
                    tokenizer = tokenizers[i]
                    text_input_ids = _tokenize(tokenizer, prompt)
                else:
                    assert text_input_ids_list is not None
                    text_input_ids = text_input_ids_list[i]

                prompt_embeds = text_encoder(
                    text_input_ids.to(text_encoder.device),
                    output_hidden_states=True, return_dict=False
                )

                # we are only ALWAYS interested in the pooled output of the
                # final text encoder
                pooled_prompt_embeds = prompt_embeds[0]
                prompt_embeds = prompt_embeds[-1][-2]
                bs_embed, seq_len, _ = prompt_embeds.shape
                prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
            pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
            return prompt_embeds, pooled_prompt_embeds

        # training
        self.logger.info('Fine-tuning with LoRA...')
        progress_bar = tqdm(range(0, max_train_steps), desc='Steps')
        for epoch in range(num_epochs):
            unet.train()
            if train_text_encoders:
                text_encoder_one.train()
                text_encoder_two.train()
            for step, batch in enumerate(train_dataloader):
                # get latent space embeddings of images
                pixel_values = batch['pixel_values'].to(
                    self.device, dtype=weight_dtype)
                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = model_input * vae.config.scaling_factor

                # add noise to latent space embeddings
                noise = torch.randn_like(model_input)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (model_input.shape[0],), device=model_input.device)
                timesteps = timesteps.long()
                noisy_model_input = noise_scheduler.add_noise(
                    model_input, noise, timesteps)

                # utility function for getting time ids
                def compute_time_ids(original_size, crops_coords_top_left):
                    # Adapted from
                    # pipeline.StableDiffusionXLPipeline._get_add_time_ids
                    target_size = (RESOLUTION, RESOLUTION)
                    add_time_ids = list(
                        original_size + crops_coords_top_left + target_size)
                    add_time_ids = torch.tensor([add_time_ids])
                    add_time_ids = add_time_ids.to(
                        self.device, dtype=weight_dtype)
                    return add_time_ids
                add_time_ids = torch.cat(
                    [compute_time_ids(s, c) for s, c in zip(
                        batch['original_sizes'], batch['crop_top_lefts'])])

                # predict the noise residual
                unet_added_conditions = {'time_ids': add_time_ids}
                prompt_embeds, pooled_prompt_embeds = encode_prompt(
                    text_encoders=[text_encoder_one, text_encoder_two],
                    tokenizers=None,
                    prompt=None,
                    text_input_ids_list=[
                        batch['input_ids_one'], batch['input_ids_two']])
                unet_added_conditions.update(
                    {'text_embeds': pooled_prompt_embeds})
                model_pred = unet(
                    noisy_model_input,
                    timesteps,
                    prompt_embeds,
                    added_cond_kwargs=unet_added_conditions,
                    return_dict=False,
                )[0]

                # calculate loss
                if noise_scheduler.config.prediction_type == 'epsilon':
                    target = noise
                elif noise_scheduler.config.prediction_type == 'v_prediction':
                    target = noise_scheduler.get_velocity(
                        model_input, noise, timesteps)
                else:
                    raise ValueError(
                        'Unknown prediction type '
                        + f'{noise_scheduler.config.prediction_type}')
                loss = F.mse_loss(
                    model_pred.float(), target.float(), reduction='mean')

                # backpropagate
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # update progress bar
                progress_bar.update(1)
        self.logger.info('Fine-tuning with LoRA complete')

        # save LoRA weights
        unet_lora_state_dict = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(unet))
        if train_text_encoders:
            text_encoder_lora_layers = convert_state_dict_to_diffusers(
                get_peft_model_state_dict(text_encoder_one))
            text_encoder_2_lora_layers = convert_state_dict_to_diffusers(
                get_peft_model_state_dict(text_encoder_two))
        else:
            text_encoder_lora_layers = None
            text_encoder_2_lora_layers = None
        StableDiffusionXLPipeline.save_lora_weights(
            save_directory=self.lora_dir,
            unet_lora_layers=unet_lora_state_dict,
            text_encoder_lora_layers=text_encoder_lora_layers,
            text_encoder_2_lora_layers=text_encoder_2_lora_layers,
        )
        self.logger.info(f'Saved LoRA weights to {self.lora_dir}')

        # save LoRA configs
        if train_text_encoders:
            _save_lora_configs(self.lora_dir, unet_lora_config,
                               text_lora_config)
        else:
            _save_lora_configs(self.lora_dir, unet_lora_config)
        self.logger.info(f'Saved LoRA config(s) to {self.lora_dir}')

        # free up memory
        del unet
        del text_encoder_one
        del text_encoder_two
        del text_encoder_lora_layers
        del text_encoder_2_lora_layers
        torch.cuda.empty_cache()

    def generate(self, prompt):
        # load pipeline with LoRA weights
        if not hasattr(self, 'pipeline'):
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                self.base_model_name, torch_dtype=torch.float32)
            self.logger.info(
                f'Loaded Stable Diffusion ({self.base_model_name}) pipeline')
            pipeline.load_lora_weights(self.lora_dir)
            pipeline = pipeline.to(self.device)

            # set LoRA alpha
            _set_pipeline_lora_alpha(pipeline, self.lora_dir)

            self.pipeline = pipeline

        # generate image and resize to original resolution of Mortal Kombat
        # character images in the dataset
        self.logger.info('Generating image of Mortal Kombat character...')
        img = self.pipeline(prompt).images[0].resize((224, 224))
        self.logger.info('Generated image of Mortal Kombat character')
        return img


'''
The below functions are adapted from the following GitHub issue
discussion comment:
https://github.com/huggingface/diffusers/issues/6087#issuecomment-1846485514

This code is a fix for a bug in the Hugging Face diffusers library where the
alpha parameter of the LoRA config is not loaded correctly when loading
saved LoRA weights. Link to the GitHub issue:
https://github.com/huggingface/diffusers/issues/6087
'''

ADAPTER_NAME = 'default_0'


def _save_lora_configs(lora_dir, unet_lora_config,
                       text_lora_config=None):
    unet_lora_config.save_pretrained(os.path.join(lora_dir, 'unet'))
    if text_lora_config is not None:
        text_lora_config.save_pretrained(
            os.path.join(lora_dir, 'text_encoder'))


def _set_pipeline_lora_alpha(
        pipeline, lora_dir):
    # set LoRA alpha for U-Net
    unet_lora_config_path = os.path.join(lora_dir, 'unet')
    unet_lora_config = LoraConfig.from_pretrained(unet_lora_config_path)
    _set_model_lora_alpha(pipeline.unet, unet_lora_config.lora_alpha)

    # set LoRA alpha for text encoders
    text_lora_config_path = os.path.join(lora_dir, 'text_encoder')
    if os.path.isdir(text_lora_config_path):
        text_lora_config = LoraConfig.from_pretrained(
            text_lora_config_path)
        _set_model_lora_alpha(
            pipeline.text_encoder, text_lora_config.lora_alpha)
        _set_model_lora_alpha(
            pipeline.text_encoder_2, text_lora_config.lora_alpha)


def _set_model_lora_alpha(model, lora_alpha):
    # for each LoRA layer in the model, set the alpha value
    for _, module in model.named_modules():
        if isinstance(module, LoraLayer):
            _set_lora_alpha(module, lora_alpha)


def _set_lora_alpha(lora_layer, lora_alpha):
    adapter = ADAPTER_NAME

    # Modified from peft.tuners.lora.layer.LoraLayer.
    if adapter not in lora_layer.active_adapters:
        return

    # if the LoRA layer is active, then set the alpha and scaling
    lora_layer.lora_alpha[adapter] = lora_alpha
    if lora_layer.r[adapter] > 0:
        lora_layer.scaling[adapter] = (
            lora_layer.lora_alpha[adapter] / lora_layer.r[adapter])
