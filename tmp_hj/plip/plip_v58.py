"""PLIP wrapper adapted for transformers 5.x (e.g. 5.8.1).

원본 plip.py 는 transformers 4.x 규약(`get_image_features`/`get_text_features`가
projection까지 적용된 임베딩 텐서 반환)에 의존한다. 5.x 에서는 그 반환값이
비전/텍스트 타워 원본 출력(BaseModelOutputWithPooling)으로 바뀌어 깨진다.

이 모듈은 vision_model/text_model 을 직접 돌리고 visual_projection/text_projection
을 수동 적용해 4.x 의 get_image_features/get_text_features 와 **동일한 값**을 낸다
(정규화 전 raw 임베딩까지 동일). 인터페이스(encode_images/encode_text)는 원본과 호환.
datasets 의존성은 제거하고 순수 torch 배칭을 쓴다.
"""
import numpy as np
import torch
import PIL
from typing import List, Union
from transformers import CLIPModel, CLIPProcessor


class PLIP:
    def __init__(self, model_name, auth_token=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        # slow 고정(4.x 저장 방식과 동일 계열). 참고: 4.x↔5.x image processor 구현차로
        # 이미지 임베딩에 max abs ~2.6e-4 잔차가 남지만 cosine=1.0 이라 랭킹엔 무영향.
        # 텍스트는 사실상 동일(~3e-6). 필요시 use_fast 는 여기서 조정.
        self.preprocess = CLIPProcessor.from_pretrained(model_name, use_fast=False)

    def _load_images(self, images):
        out = []
        for im in images:
            if isinstance(im, str):
                out.append(PIL.Image.open(im).convert("RGB"))
            else:
                out.append(im.convert("RGB") if im.mode != "RGB" else im)
        return out

    @torch.no_grad()
    def encode_images(self, images: Union[List[str], List["PIL.Image.Image"]], batch_size: int):
        embs = []
        for i in range(0, len(images), batch_size):
            chunk = self._load_images(images[i:i + batch_size])
            inputs = self.preprocess(images=chunk, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            # 4.x get_image_features 와 동일: vision pooler_output -> visual_projection
            vision_out = self.model.vision_model(pixel_values=pixel_values)
            image_embeds = self.model.visual_projection(vision_out.pooler_output)
            embs.append(image_embeds.detach().cpu().numpy())
        return np.concatenate(embs, axis=0)

    @torch.no_grad()
    def encode_text(self, text: List[str], batch_size: int):
        embs = []
        for i in range(0, len(text), batch_size):
            chunk = text[i:i + batch_size]
            inputs = self.preprocess(text=chunk, return_tensors="pt",
                                     max_length=77, padding="max_length", truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            # 4.x get_text_features 와 동일: text pooler_output -> text_projection
            text_out = self.model.text_model(**inputs)
            text_embeds = self.model.text_projection(text_out.pooler_output)
            embs.append(text_embeds.detach().cpu().numpy())
        return np.concatenate(embs, axis=0)

    def _cosine_similarity(self, key_vectors: np.ndarray, space_vectors: np.ndarray, normalize=True):
        if normalize:
            key_vectors = key_vectors / np.linalg.norm(key_vectors, ord=2, axis=-1, keepdims=True)
        return np.matmul(key_vectors, space_vectors.T)
