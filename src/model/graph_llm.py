import contextlib
import torch
import torch.nn as nn
from torch_scatter import scatter
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from src.model.gnn import load_gnn_model

BOS = "<s>[INST]"
EOS_USER = "[/INST]"
EOS = "</s>"
IGNORE_INDEX = -100

class GraphLLM(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.max_txt_len = args.max_txt_len
        self.max_new_tokens = args.max_new_tokens

        print(f"Loading LLM from {args.llm_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, use_fast=False)
        self.tokenizer.pad_token_id=0
        self.tokenizer.padding_side="left"

        load_in_4bit = getattr(args, "load_in_4bit", str(args.llm_model_name) == "7b")
        model_kwargs = {
            "low_cpu_mem_usage": True,
            "device_map": {"": 0} if torch.cuda.is_available() else "cpu",
        }
        if load_in_4bit and torch.cuda.is_available():
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True)
        else:
            model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(args.llm_model_path, **model_kwargs)

        if str(args.llm_frozen) == "True":
            print("Freezing LLM backbone parameters!")
            for _, param in model.named_parameters():
                param.requires_grad=False
        else:
            print("Enabling LoRA fine-tuning on LLM!")
            model = prepare_model_for_kbit_training(model)
            lora_config= LoraConfig(
                r = 8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                )
            model = get_peft_model(model, lora_config)

        self.model = model
        self.word_embedding = self.model.get_input_embeddings()
        llm_hidden_dim = self.model.config.hidden_size
        target_device = self.word_embedding.weight.device

        self.graph_encoder = load_gnn_model[args.gnn_model_name](
            in_channels=args.gnn_in_dim,
            out_channels=args.gnn_hidden_dim,
            hidden_channels=args.gnn_hidden_dim,
            num_layers=args.gnn_num_layers,
            dropout=args.gnn_dropout,
            mlp_layers=args.alignment_mlp_layers,
            num_heads=args.gnn_num_heads,
            operator=args.distance_operator,
            edge_dim=getattr(args,"gnn_edge_dim", None),
        ).to(target_device)

        self.projector = nn.Sequential(
            nn.Linear(args.gnn_hidden_dim, 2048),
            nn.Sigmoid(),
            nn.Linear(2048, llm_hidden_dim),
            ).to(target_device)

    @property
    def device(self):
        return (self.word_embedding.weight.device)

    def maybe_autocast(self, dtype=torch.float16):
        if self.device.type != "cpu":
            return torch.cuda.amp.autocast(dtype=dtype)
        return contextlib.nullcontext()

    def encode_graphs(self, samples):
        graphs = samples["graph"].to(self.device)
        n_embeds, _=self.graph_encoder(
               graphs.x,
               graphs.edge_index.long(),
               graphs.question_node,
               graphs.edge_attr,
               graphs.question_edge,
           )
        g_embeds = scatter(
            n_embeds,
            graphs.batch,
            dim=0,
            dim_size=len(samples["id"]),
            reduce="mean",
           )
        return g_embeds

    def forward(self, samples):
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        descriptions = self.tokenizer (samples["desc"], add_special_tokens=False)
        labels = self.tokenizer(samples["label"], add_special_tokens=False)

        eos_tokens = self.tokenizer(EOS, add_special_tokens=False)
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_ids = self.tokenizer(BOS, add_special_tokens=False, return_tensors="pt").input_ids[0].to(self.device)
        bos_embeds = self.word_embedding(bos_ids)
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id, device = self.device)).unsqueeze(0)

        graph_embeds = self.projector(self.encode_graphs(samples)).to(bos_embeds.dtype)

        batch_size = len(samples["id"])
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []

        for i in range(batch_size):
            label_input_ids = labels.input_ids[i][: self.max_new_tokens] + eos_tokens.input_ids
            input_ids = (
                    descriptions.input_ids[i][: self.max_txt_len]
                    + questions.input_ids[i]
                    + eos_user_tokens.input_ids
                    + label_input_ids
                )
            inputs_embeds = self.word_embedding(torch.tensor(input_ids, device=self.device))
            inputs_embeds = torch.cat([bos_embeds, graph_embeds[i].unsqueeze(0), inputs_embeds], dim=0)

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            masked_labels = [IGNORE_INDEX] * (inputs_embeds.shape[0] - len(label_input_ids)) + label_input_ids
            batch_label_input_ids.append(masked_labels)
        max_length = max(x.shape[0] for x in batch_inputs_embeds)
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]
            batch_label_input_ids[i] = [IGNORE_INDEX] * pad_length + batch_label_input_ids[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.device)
        attention_mask = torch.tensor(batch_attention_mask, device=self.device)
        label_input_ids = torch.tensor(batch_label_input_ids, device=self.device)

        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask =attention_mask,
                return_dict=True,
                labels=label_input_ids)
        return (outputs.loss)

    def inference(self, samples):
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)

        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_ids = self.tokenizer(BOS, add_special_tokens=False, return_tensors="pt").input_ids[0].to(self.device)
        bos_embeds = self.word_embedding(bos_ids)
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id, device=self.device)).unsqueeze(0)

        graph_embeds = self.projector(self.encode_graphs(samples)).to(bos_embeds.dtype)
        batch_size = len(samples["id"])
        batch_inputs_embeds = []
        batch_attention_mask = []

        for i in range(batch_size):
            input_ids = (
                descriptions.input_ids[i][: self.max_txt_len]
                + questions.input_ids[i]
                + eos_user_tokens.input_ids
            )
            inputs_embeds = self.word_embedding(torch.tensor(input_ids, device=self.device))
            inputs_embeds = torch.cat([bos_embeds, graph_embeds[i].unsqueeze(0), inputs_embeds], dim=0)
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        max_length = max(x.shape[0] for x in batch_inputs_embeds)
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.device)
        attention_mask = torch.tensor(batch_attention_mask, device=self.device)

        with self.maybe_autocast():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.max_new_tokens,
                attention_mask=attention_mask,
                use_cache=True,
            )
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return {
            "id": samples["id"],
            "pred": pred,
            "label": samples["label"],
            "question": samples["question"],
            "desc": samples["desc"],
            }

    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        return trainable_params, all_param


if __name__ == "__main__":
        import tempfile
        from types import SimpleNamespace
        from transformers import LlamaConfig, LlamaForCausalLM
        from src.dataset.expla_graphs import ExplaGraphsDataset
        from src.utils.collate import collate_fn

        with tempfile.TemporaryDirectory() as tmpdir:
            # Build a fast 2-layer LLaMA proxy (d_LLM = 2048) using cached tokenizer
            tok = AutoTokenizer.from_pretrained("sentence-transformers/all-roberta-large-v1")
            tok.save_pretrained(tmpdir)
            cfg = LlamaConfig(
                vocab_size=len(tok),
                hidden_size=2048,
                intermediate_size=4096,
                num_hidden_layers=2,
                num_attention_heads=16,
                pad_token_id=0,
            )
            LlamaForCausalLM(cfg).half().save_pretrained(tmpdir)

            args = SimpleNamespace(
                llm_model_path=tmpdir,
                llm_model_name="1b_proxy",
                llm_frozen="True",
                load_in_4bit=False,
                max_txt_len=128,
                max_new_tokens=8,
                gnn_model_name="gat",
                gnn_in_dim=1024,
                gnn_hidden_dim=1024,
                gnn_num_layers=4,
                gnn_dropout=0.0,
                alignment_mlp_layers=3,
                gnn_num_heads=4,
                distance_operator="euclidean",
                gnn_edge_dim=1024,
            )

            dataset = ExplaGraphsDataset()
            batch = collate_fn([dataset[0], dataset[1]])

            model = GraphLLM(args)
            trainable, total = model.print_trainable_params()
            loss = model(batch)
            loss.backward()

            proj_grad = model.projector[0].weight.grad.norm().item()
            gat_grad = model.graph_encoder.mlp_x.layers[0].weight.grad.norm().item()
            edge_grad = model.graph_encoder.mlp_edge_attr.layers[0].weight.grad.norm().item()
            llm_grads = [p.grad for p in model.model.parameters() if p.grad is not None]
            vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0

            print("\n=== Phase 5 End-to-End GraphLLM Plumbing Verification ===")
            print(f"Trainable params:    {trainable:,} / {total:,}")
            print(f"Forward Loss:        {loss.item():.4f} (finite: {torch.isfinite(loss).item()})")
            print(f"Projector grad norm: {proj_grad:.6f} (> 0)")
            print(f"GAT mlp_x grad norm: {gat_grad:.6f} (> 0)")
            print(f"GAT edge grad norm:  {edge_grad:.6f} (> 0)")
            print(f"Frozen LLM grads:    {len(llm_grads)} (must be 0)")
            print(f"Peak CUDA VRAM:      {vram_mb:.1f} MiB (< 5500 MiB)")