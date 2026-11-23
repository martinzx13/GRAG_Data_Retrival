import os
import torch

def _checkpoint_path(args, tag)->str:
    os.makedirs(f"{args.output_dir}/{args.dataset}", exist_ok=True)
    return (
            f"{args.output_dir}/{args.dataset}/"
            f"model_name_{args.model_name}_llm_model_name_{args.llm_model_name}_"
            f"llm_frozen_{args.llm_frozen}_max_txt_len_{args.max_txt_len}_"
            f"max_new_tokens_{args.max_new_tokens}_gnn_model_name_{args.gnn_model_name}_"
            f"patience_{args.patience}_num_epochs_{args.num_epochs}_"
            f"seed{args.seed}_checkpoint_{tag}.pth" )

def _save_checkpoint(model, optimizer, cur_epoch: int, args, is_best: bool = False) -> str:
    """Save only trainable (requires_grad=True) parameters to keep checkpoints lightweight."""
    param_grad_dic = {k: v.requires_grad for k, v in model.named_parameters()}
    state_dict = {
        k: v.cpu()
        for k, v in model.state_dict().items()
        if param_grad_dic.get(k, False)
    }
    save_obj = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "config": vars(args) if hasattr(args, "__dict__") else args,
        "epoch": cur_epoch,
    }
    tag = "best" if is_best else cur_epoch
    path = _checkpoint_path(args, tag)
    print(f"Saving checkpoint at epoch {cur_epoch} to {path}.")
    torch.save(save_obj, path)
    return (path)

def _reload_best_model(model, args):
    """Load the best trainable checkpoint weights back into the model """
    path = _checkpoint_path(args, "best")
    print(f"Loading best checkpoint weights from {path}")
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    return (model)
