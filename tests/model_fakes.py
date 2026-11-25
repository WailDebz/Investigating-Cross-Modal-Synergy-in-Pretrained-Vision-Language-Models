import torch
from torch import nn


class Holder(nn.Module):
    pass


class FakeCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = Holder()
        self.vision_model.encoder = Holder()
        self.vision_model.encoder.layers = nn.ModuleList([nn.Linear(3, 3)])
        self.text_model = Holder()
        self.text_model.encoder = Holder()
        self.text_model.encoder.layers = nn.ModuleList([nn.Linear(3, 3)])
        self.visual_projection = nn.Linear(3, 4)
        self.text_projection = nn.Linear(3, 4)
        self.token_embedding = nn.Embedding(16, 3)
        self.logit_scale = nn.Parameter(torch.tensor(0.0))

    def get_image_features(self, pixel_values):
        pooled = pixel_values.mean((2, 3))
        encoded = self.vision_model.encoder.layers[-1](pooled)
        return self.visual_projection(encoded)

    def get_text_features(self, input_ids, attention_mask=None):
        pooled = self.token_embedding(input_ids).mean(1)
        encoded = self.text_model.encoder.layers[-1](pooled)
        return self.text_projection(encoded)


class FakeProcessor:
    def __call__(self, images, text, return_tensors, padding):
        return {
            "pixel_values": images,
            "input_ids": torch.tensor([[len(value) % 16] for value in text]),
            "attention_mask": torch.ones(len(text), 1, dtype=torch.long),
        }
