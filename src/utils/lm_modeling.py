import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "sentence-transformers/all-roberta-large-v1"

class SentenceTransformersWrapper(torch.nn.Module):
    def __init__(self, pretrained_repo:str = MODEL_NAME):
        super().__init__()
        self.bert_model = AutoModel.from_pretrained(pretrained_repo)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] # First element has all token_embeddings [B, L, 1024]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return (sum_embeddings / sum_mask)

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.mean_pooling(bert_out, attention_mask)
        return (F.normalize(pooled, p=2, dim=1))

def load_sbert(device:str = "cuda"):
    model = SentenceTransformersWrapper(MODEL_NAME).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return (model, tokenizer)

#change the batch_size to 256 or 512 in a cluster

def encode_texts(model, tokenizer, texts:list[str], batch_size: int = 64, device:str = "cuda") -> torch.Tensor:
        """
        Encodes a list of strings into L2-normalized 1024-d embeddings.
        Uses dynamic per-batch padding and moves results to CPU to preserve VRAM.
        Raises ValueError on empty input list.
        """

        # 1. Guard Against empty list 
        if len(texts) == 0:
            raise ValueError("Cannot encode an empty list of text")

        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[ i: i + batch_size]
                # Dynamic padding: padding=True pads ONLY to the longest text in batch_texts!
                encoded = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)

                embeddings = model(input_ids, attention_mask)
                all_embeddings.append(embeddings.cpu())
        return (torch.cat(all_embeddings, dim=0))
