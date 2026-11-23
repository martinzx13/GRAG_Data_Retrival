import gc 
import os 
import torch 
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm 

from src.dataset.expla_graphs import ExplaGraphsDataset
from src.model.graph_llm import GraphLLM 
from src.utils.ckpt import _reload_best_model, _save_checkpoint
from src.utils.collate import collate_fn
from src.utils.config import parse_args_llama
from src.utils.evaluate import eval_funcs
from src.utils.lr_schedule import adjust_learning_rate
from src.utils.seed import seed_everything

load_dataset = {
        "expla_graphs" : ExplaGraphsDataset,
        }

load_model = {
        "graph_llm" : GraphLLM,
        }

llama_model_path = {
        "1b_proxy": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "7b": "meta-llama/Llama-2-7b-hf",
        "7b_chat": "meta-llama/Llama-2-7b-chat-hf",
        }


def main(args):
    seed_everything(seed=args.seed)
    if (args.use_wandb):
        import wandb 
        wandb.init(
                project=f"{args.project}",
                name=f"{args.dataset}_{args.model_name}_seed{args.seed}",
                config=vars(args),
                )
    print(args)

    # Step 1 & 2 : Load Dataset & Split Subsets
    dataset = load_dataset[args.dataset]()
    idx_split = dataset.get_idx_split()

    train_dataset = [dataset[i] for i in idx_split["train"]]
    val_dataset = [dataset[i] for i in idx_split["val"]]
    test_dataset = [dataset[i] for i in idx_split["test"]]

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        drop_last=True,
        pin_memory=True,
        shuffle=True,
        collate_fn=collate_fn,
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        drop_last=False,
        pin_memory=True,
        shuffle=False,
        collate_fn=collate_fn,
        )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        drop_last=False,
        pin_memory=True,
        shuffle=False,
        collate_fn=collate_fn,
        )

    # Step 3: Build Model 
    if not args.llm_model_path:
        args.llm_model_path = llama_model_path[args.llm_model_name]
    model = load_model[args.model_name](
            graph_type=dataset.graph_type,
            args=args,
            init_prompt=dataset.prompt,
            )

    # Step 4: Configure Optimizer (AdamW over trainable parameters only)
    params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
            [{"params": params, "lr": args.lr, "weight_decay": args.wd}],
            betas=(0.9, 0.95),
            )
    trainable_params, all_param=model.print_trainable_params()
    print(
            f"trainable params: {trainable_params:,} || all params: {all_param:,} || "
            f"trainable%: {100 * trainable_params / all_param:.4f}%"
        )

    # Step 5 : Training & Validation Loop 
    steps_per_epoch = min(len(train_loader), args.max_train_steps) if args.max_train_steps > 0 else len(train_loader)
    progress_bar = tqdm(range(args.num_epochs * steps_per_epoch))
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss, accum_loss = 0.0, 0.0

        for step, batch in enumerate(train_loader):
            if args.max_train_steps > 0 and step >= args.max_train_steps:
                break

            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()

            clip_grad_norm_(optimizer.param_groups[0]["params"], 0.1)

            if (step + 1) % args.grad_steps == 0:
                adjust_learning_rate(
                        optimizer.param_groups[0],
                        args.lr,
                        step / steps_per_epoch + epoch,
                        args,
                        )
            optimizer.step()
            epoch_loss += loss.item()
            accum_loss  += loss.item()

            if (step + 1) % args.grad_steps == 0:
                if (args.use_wandb):
                    import wandb
                    wandb.log({
                            "Lr": optimizer.param_groups[0]["lr"],
                            "Accum Loss": accum_loss / args.grad_steps,
                        })
                accum_loss = 0.0
            progress_bar.update(1)
 
        mean_train_loss = epoch_loss / max(steps_per_epoch, 1)
        print(f"Epoch: {epoch}|{args.num_epochs}: Train Loss (Epoch Mean): {mean_train_loss:.4f}")

        #Validation
        val_loss = 0.0
        val_steps = 0
        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                if args.max_val_steps > 0 and step >= args.max_val_steps:
                    break
                loss = model(batch)
                val_loss += loss.item()
                val_steps += 1

        val_loss = val_loss / max(val_steps, 1)

        print(f"Epoch: {epoch}|{args.num_epochs}: Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(model, optimizer, epoch, args, is_best=True)
            best_epoch = epoch

        print(f"Epoch {epoch} Val Loss {val_loss:.4f} Best Val Loss {best_val_loss:.4f} Best Epoch {best_epoch}")
        if epoch - best_epoch >= args.patience:
            print(f"Early stop at epoch {epoch}")
            break

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()

        # Step 6: Reload Best Checkpoint & Evaluate on Test Split
    model = _reload_best_model(model, args)
    model.eval()
    eval_output = []
    test_steps = min(len(test_loader), args.max_test_steps) if args.max_test_steps > 0 else len(test_loader)
    progress_bar_test = tqdm(range(test_steps))

    for step, batch in enumerate(test_loader):
        if args.max_test_steps > 0 and step >= args.max_test_steps:
            break
        with torch.no_grad():
            output = model.inference(batch)
            eval_output.append(output)
        progress_bar_test.update(1)

    os.makedirs(f"{args.output_dir}/{args.dataset}", exist_ok=True)
    path = (
            f"{args.output_dir}/{args.dataset}/"
            f"model_name_{args.model_name}_llm_model_name_{args.llm_model_name}_"
            f"llm_frozen_{args.llm_frozen}_max_txt_len_{args.max_txt_len}_"
            f"max_new_tokens_{args.max_new_tokens}_gnn_model_name_{args.gnn_model_name}_"
            f"patience_{args.patience}_num_epochs_{args.num_epochs}_seed{args.seed}.csv"
        )
    acc = eval_funcs[args.dataset](eval_output, path)
    print(f"Test Acc: {acc:.4f}")
    return (acc)

if __name__ == "__main__":
    args = parse_args_llama()
    main(args)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
    gc.collect()


