import torch.nn as nn
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

from transformers.utils import ModelOutput
from transformers import PreTrainedModel, PretrainedConfig, AutoModel, AutoTokenizer, AutoProcessor

@dataclass
class SiglipOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits_per_text: Optional[torch.FloatTensor] = None
    logits_per_image: Optional[torch.FloatTensor] = None
    text_embeds: Optional[torch.FloatTensor] = None
    image_embeds: Optional[torch.FloatTensor] = None

class SiglipConfig(PretrainedConfig):
    model_type = "siglip"

    def __init__(self,
                 text_model_name: str = "./pretrained_models/bert-base-chinese",
                 image_model_name: str = "./pretrained_models//vit-base-patch16-224",
                 **kwargs):
        super().__init__(**kwargs)
        self.text_model_name = text_model_name
        self.image_model_name = image_model_name

class SiglipModel(PreTrainedModel):
    config_class = SiglipConfig

    def __init__(self, config: SiglipConfig):
        super().__init__(config)
        self.vision_model = AutoModel.from_pretrained(config.image_model_name)
        self.preprocessor = AutoProcessor.from_pretrained(config.image_model_name)
        self.text_model = AutoModel.from_pretrained(config.text_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_model_name)
        self.t = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))

    def forward(
        self,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
    ):
        text_embeddings = self.text_model(input_ids=input_ids, attention_mask=attention_mask)["pooler_output"]
        image_embeddings = self.vision_model(pixel_values=pixel_values)["pooler_output"]

        # NOTE: L2 Normalization
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        
        # NOTE: logits computation
        logits_per_text = torch.matmul(text_embeddings, image_embeddings.t()) * torch.exp(self.t) + self.b
        logits_per_image = logits_per_text.t()

        batch_size = input_ids.size(0)
        eye = torch.eye(batch_size).to(logits_per_text.device)

        # NOTE: Z_ij是标签, 通常定义为:对角线为1, 非对角线为-1
        labels = 2 * eye - torch.ones_like(logits_per_text, device=logits_per_text.device)

        # NOTE: sigmoid替换之前的softmax, 每个元素独立计算，不依赖其他元素。
        loglik = F.logsigmoid(labels * logits_per_text)
        nll = -torch.sum(loglik, dim=-1)
        loss = nll.mean()
        
        return SiglipOutput(
            loss=loss,
            logits_per_text=logits_per_text,
            logits_per_image=logits_per_image,
            text_embeds=text_embeddings,
            image_embeds=image_embeddings,
        )

if __name__ == "__main__":
    siglip = SiglipModel(SiglipConfig())
    input_ids = torch.randint(0, 1000, (2, 16))
    attention_mask = torch.ones((2, 16))
    pixel_values = torch.randn(2, 3, 224, 224)
    outputs = siglip(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
