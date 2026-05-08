"""
Shared utilities for discrete flow GRPO training.

Kept lean — only prompts_to_tensor and its helpers.
Dataset classes → data.py, model loading → model_utils.py, rewards → reward_utils.py.
Constants → config.py.
"""

import torch
from flow_matching.path import MixtureDiscreteSoftmaxProbPath


def prompts_to_tensor(
        prompts,  # list of prompts
        vl_chat_processor,
        path: MixtureDiscreteSoftmaxProbPath,
        t,
        g_or_u,  # 'generation' or 'understanding'
        txt_max_length=500,
        IMG_LEN=576,
        answers=None,
        img_in_question=None,
        img_tokens=None,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
    """
    Plug img_tokens or answer into sentences and return x_0, x_t and data_info.
    """
    data_info_text_token_mask = []
    data_info_image_token_mask = []
    data_info_generation_or_understanding_mask = []
    data_info_attention_mask = []
    data_info_sft_format = []
    data_info_understanding_img = []
    data_info_has_understanding_img = []
    input_ids_list = []
    origin_input_ids_list = []
    for i in range(len(prompts)):
        prompt = prompts[i]
        if g_or_u == "generation":
            conversation = [
                {
                    "role": "User",
                    "content": prompt
                },
                {
                    "role": "Assistant",
                    "content": ""
                }
            ]
        else:
            if answers is not None:
                conversation = [
                    {
                        "role": "User",
                        "content": "<image_placeholder>" + prompt
                    },
                    {
                        "role": "Assistant",
                        "content": answers[i]
                    }
                ]
            else:
                conversation = [
                    {
                        "role": "User",
                        "content": "<image_placeholder>" + prompt
                    },
                    {
                        "role": "Assistant",
                        "content": ""
                    }
                ]

        # output: str
        if g_or_u == "generation":
            sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=vl_chat_processor.sft_format,
                system_prompt="",
            )
        else:
            sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=vl_chat_processor.sft_format,
                system_prompt=vl_chat_processor.system_prompt,
            )

        if g_or_u == "generation":
            sft_format = sft_format + vl_chat_processor.image_start_tag
            input_ids = vl_chat_processor.tokenizer.encode(sft_format)
            input_ids = torch.LongTensor(input_ids)
            img_start = input_ids.shape[0]
            input_ids = torch.cat([input_ids, torch.LongTensor([vl_chat_processor.image_id]*IMG_LEN), torch.LongTensor([vl_chat_processor.image_end_id])])
            img_end = input_ids.shape[0] - 1
        else:
            input_ids = vl_chat_processor.tokenizer.encode(sft_format)
            input_ids = torch.LongTensor(input_ids)
            image_token_mask = (input_ids == vl_chat_processor.image_id)
            image_indices = image_token_mask.nonzero()
            input_ids, _ = vl_chat_processor.add_image_token(
                image_indices=image_indices,
                input_ids=input_ids,
            )

        # pad tokens
        original_input_id_len = input_ids.shape[0]

        if original_input_id_len >= txt_max_length + IMG_LEN:
            print(f"Sample {i} too long, skipping...")
            original_input_id_len = txt_max_length + IMG_LEN - 1
            input_ids = input_ids[:original_input_id_len]
            rows_to_pad = txt_max_length + IMG_LEN - input_ids.shape[0]
            input_ids = torch.cat([input_ids, torch.LongTensor([vl_chat_processor.pad_id]).repeat(rows_to_pad)], dim=0)
            attention_mask = torch.zeros((input_ids.shape[0]), dtype=torch.bool)
            if g_or_u == 'generation':
                attention_mask[:original_input_id_len] = True
            else:
                attention_mask[:] = True
        else:
            rows_to_pad = txt_max_length + IMG_LEN - input_ids.shape[0]
            input_ids = torch.cat([input_ids, torch.LongTensor([vl_chat_processor.pad_id]).repeat(rows_to_pad)], dim=0)
            attention_mask = torch.zeros((input_ids.shape[0]), dtype=torch.bool)
            if g_or_u == 'generation':
                attention_mask[:original_input_id_len] = True
            else:
                attention_mask[:] = True

        if g_or_u == "generation":
            image_expanded_token_mask = torch.zeros_like(input_ids)
            image_expanded_token_mask[img_start: img_end] = True
        else:
            if img_in_question[i] is not None:
                image_expanded_token_mask = (input_ids == vl_chat_processor.image_id).to(dtype=int)
                image_expanded_mask_indices = torch.where(image_expanded_token_mask == 1)[0]
                input_ids[image_expanded_mask_indices] = 0
            else:
                image_expanded_token_mask = torch.zeros_like(input_ids)

        # obtain text token mask
        text_expanded_token_mask = torch.zeros_like(image_expanded_token_mask)
        split_token = vl_chat_processor.tokenizer.encode("Assistant:", add_special_tokens=False)
        split_token_length = len(split_token)

        start_index = -1
        for j in range(len(input_ids) - split_token_length + 1):
            if input_ids[j:j + split_token_length].numpy().tolist() == split_token:
                start_index = j
                break
        if start_index != -1:
            if g_or_u == "generation":
                text_expanded_token_mask[1: (start_index + split_token_length)] = 1
            else:
                text_expanded_token_mask[(start_index + split_token_length):] = 1
        else:
            raise ValueError("Split token not found in input_ids")

        data_info_text_token_mask.append(text_expanded_token_mask)
        data_info_image_token_mask.append(image_expanded_token_mask)
        data_info_generation_or_understanding_mask.append(torch.Tensor([1 if g_or_u == 'generation' else 0]).to(dtype=int).to(device))
        data_info_attention_mask.append(attention_mask)
        data_info_sft_format.append(sft_format)

        if g_or_u == 'generation':
            data_info_has_understanding_img.append(torch.Tensor([False]).to(dtype=int).to(device))
            data_info_understanding_img.append(torch.zeros((3, 384, 384)).to(device))
        else:
            if img_in_question[i] is not None:
                data_info_has_understanding_img.append(torch.Tensor([True]).to(dtype=int).to(device))
                data_info_understanding_img.append(img_in_question[i].to(device))
            else:
                data_info_has_understanding_img.append(torch.Tensor([False]).to(dtype=int).to(device))
                data_info_understanding_img.append(torch.zeros((3, 384, 384)).to(device))

        if g_or_u == "generation":
            input_ids = input_ids.to(device)
            image_expanded_token_mask = image_expanded_token_mask.to(device)
            x_0_img = torch.randint(
                0, 16384,
                (1, IMG_LEN),
                dtype=torch.long, device=device
            )
            path_sample = path.sample(x_0_img, img_tokens[i].view(1, -1), t[i].view(1, 1))
            input_ids[image_expanded_token_mask == 1] = path_sample.x_t[0]
            input_ids_list.append(input_ids)
            origin_input_ids = input_ids.clone()
            origin_input_ids[image_expanded_token_mask == 1] = img_tokens[i]
            origin_input_ids_list.append(origin_input_ids)
        else:
            input_ids = input_ids.to(device)
            text_expanded_token_mask = text_expanded_token_mask.to(device)
            origin_input_ids = input_ids.clone()
            origin_input_ids_list.append(origin_input_ids)
            txt_length = (text_expanded_token_mask == 1).sum().item()
            x_0_txt = torch.randint(
                0, 102400,
                (1, txt_length),
                dtype=torch.long, device=device
            )
            if t[i].sum() == 0:
                input_ids[text_expanded_token_mask == 1] = x_0_txt[0]
            else:
                path_sample = path.sample(x_0_txt, input_ids[text_expanded_token_mask == 1].view(1, -1), t[i].view(1, 1))
                input_ids[text_expanded_token_mask == 1] = path_sample.x_t[0]
            input_ids_list.append(input_ids)

    data_info = dict()
    data_info['text_token_mask'] = torch.stack(data_info_text_token_mask, dim=0).to(device)
    data_info['image_token_mask'] = torch.stack(data_info_image_token_mask, dim=0).to(device)
    data_info['generation_or_understanding_mask'] = torch.tensor(data_info_generation_or_understanding_mask, device=device, dtype=torch.int)
    data_info['attention_mask'] = torch.stack(data_info_attention_mask, dim=0).to(device)
    data_info['sft_format'] = data_info_sft_format
    # Cast to bf16 to match the vision tower / aligner / model dtype.
    data_info['understanding_img'] = torch.stack(data_info_understanding_img, dim=0).to(device).to(torch.bfloat16)
    data_info['has_understanding_img'] = torch.stack(data_info_has_understanding_img, dim=0).to(device)
    input_ids_tensor_t = torch.stack(input_ids_list, dim=0).to(device)
    origin_input_ids_tensor = torch.stack(origin_input_ids_list, dim=0).to(device)
    return input_ids_tensor_t, origin_input_ids_tensor, data_info
