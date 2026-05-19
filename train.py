"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import math
import os
from collections import Counter
from contextlib import nullcontext
from math import ceil
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

from model import Transformer, make_src_mask, make_tgt_mask

wandb = None


def _ensure_wandb():
    global wandb
    if wandb is not None:
        return wandb

    try:
        import wandb as wandb_module
    except ImportError:
        return None

    wandb = wandb_module
    return wandb


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS  
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need"

    Smoothed target distribution:
        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1).")

        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]  (raw model output)
            target : shape [batch * tgt_len]              (gold token indices)

        Returns:
            Scalar loss value.
        """
        if logits.size(-1) != self.vocab_size:
            raise ValueError(
                f"Expected logits with vocab size {self.vocab_size}, got {logits.size(-1)}."
            )

        log_probs = torch.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.full_like(
                log_probs,
                fill_value=self.smoothing / max(self.vocab_size - 2, 1),
            )
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist[:, self.pad_idx] = 0.0

            pad_rows = target == self.pad_idx
            true_dist[pad_rows] = 0.0

        loss = torch.sum(-true_dist * log_probs, dim=1)
        non_pad_mask = target != self.pad_idx

        if non_pad_mask.any():
            return loss[non_pad_mask].mean()
        return loss.mean() * 0.0


# ══════════════════════════════════════════════════════════════════════
#   TRAINING LOOP  
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
    wandb_module=None,
    global_step_start: int = 0,
    grad_log_steps: int = 0,
) -> tuple:
    """
    Run one epoch of training or evaluation.

    Args:
        data_iter  : DataLoader yielding (src, tgt) batches of token indices.
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (None during eval).
        scheduler  : NoamScheduler instance (None during eval).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, perform backward pass and scheduler step.
        device     : 'cpu' or 'cuda'.
        wandb_module : Optional W&B module for step-level logging.
        global_step_start : Number of optimizer updates completed before this epoch.
        grad_log_steps : Log gradient norms for the first N optimizer updates.

    Returns:
        avg_loss : Average loss over the epoch (float).
        avg_confidence : Average softmax prob of correct token (float).
        avg_accuracy : Token accuracy over non-pad targets (float).
        global_step : Updated optimizer step count after the epoch (int).
    """
    if is_train and optimizer is None:
        raise ValueError("An optimizer is required when is_train=True.")

    pad_idx = getattr(loss_fn, "pad_idx", 1)
    model.train(is_train)

    total_loss   = 0.0
    total_tokens = 0
    total_confidence = 0.0
    total_correct = 0
    global_step = global_step_start

    # Collect step-level gradient norms in memory — no network calls during training
    grad_log_buffer = []

    grad_context = nullcontext() if is_train else torch.no_grad()

    pbar = tqdm(data_iter, desc=f"Epoch {epoch_num} [{'Train' if is_train else 'Val'}]", unit="batch")
    for src, tgt in pbar:
        src = src.to(device)
        tgt = tgt.to(device)

        decoder_input = tgt[:, :-1]
        decoder_target = tgt[:, 1:]
        src_mask = make_src_mask(src, pad_idx=pad_idx)
        tgt_mask = make_tgt_mask(decoder_input, pad_idx=pad_idx)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with grad_context:
            logits = model(src, decoder_input, src_mask, tgt_mask)
            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)),
                decoder_target.reshape(-1),
            )

        if is_train:
            loss.backward()

            next_step = global_step + 1
            if next_step <= grad_log_steps:
                q_grad = model.encoder.layers[0].self_attn.W_q.weight.grad
                k_grad = model.encoder.layers[0].self_attn.W_k.weight.grad
                if q_grad is not None and k_grad is not None:
                    grad_log_buffer.append({
                        "train_step": next_step,
                        "grad_norm_Wq": q_grad.norm().item(),
                        "grad_norm_Wk": k_grad.norm().item(),
                    })

            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            global_step = next_step

        non_pad_tokens = int(decoder_target.ne(pad_idx).sum().item())
        total_loss += loss.item() * max(non_pad_tokens, 1)
        total_tokens += max(non_pad_tokens, 1)

        with torch.no_grad():
            probs = torch.softmax(logits.reshape(-1, logits.size(-1)).detach(), dim=-1)
            flat_target = decoder_target.reshape(-1)
            mask = flat_target.ne(pad_idx)
            correct_probs = probs[torch.arange(probs.size(0), device=probs.device), flat_target]
            total_confidence += correct_probs[mask].sum().item()
            predictions = probs.argmax(dim=-1)
            total_correct += predictions[mask].eq(flat_target[mask]).sum().item()

        pbar.set_postfix(loss=loss.item())

    avg_loss       = total_loss / max(total_tokens, 1)
    avg_confidence = total_confidence / max(total_tokens, 1)
    avg_accuracy   = total_correct / max(total_tokens, 1)

    return avg_loss, avg_confidence, avg_accuracy, global_step, grad_log_buffer


# ══════════════════════════════════════════════════════════════════════
#   GREEDY DECODING  
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.
    """
    was_training = model.training
    model.eval()

    src = src.to(device)
    src_mask = src_mask.to(device)
    ys = torch.full((src.size(0), 1), start_symbol, dtype=torch.long, device=device)

    with torch.no_grad():
        memory = model.encode(src, src_mask)

        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys)
            logits = model.decode(memory, src_mask, ys, tgt_mask)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)

            if torch.all(next_token.squeeze(1) == end_symbol):
                break

    model.train(was_training)
    return ys


def _lookup_vocab_index(vocab, token: str) -> int:
    if hasattr(vocab, "stoi"):
        return vocab.stoi[token]
    if hasattr(vocab, "token_to_idx"):
        return vocab.token_to_idx[token]
    raise AttributeError("Vocabulary object must expose stoi or token_to_idx.")


def _lookup_vocab_token(vocab, index: int) -> str:
    if hasattr(vocab, "lookup_token"):
        return vocab.lookup_token(index)
    if hasattr(vocab, "itos"):
        return vocab.itos[index]
    if hasattr(vocab, "idx_to_token"):
        return vocab.idx_to_token[index]
    raise AttributeError("Vocabulary object must expose lookup_token, itos, or idx_to_token.")


def _strip_special_tokens(indices: list[int], vocab) -> list[str]:
    pad_idx = _lookup_vocab_index(vocab, "<pad>")
    sos_idx = _lookup_vocab_index(vocab, "<sos>")
    eos_idx = _lookup_vocab_index(vocab, "<eos>")

    tokens: list[str] = []
    for index in indices:
        if index == eos_idx:
            break
        if index in {pad_idx, sos_idx}:
            continue
        tokens.append(_lookup_vocab_token(vocab, index))
    return tokens


def _extract_ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _corpus_bleu(
    references: list[list[list[str]]],
    hypotheses: list[list[str]],
    max_n: int = 4,
) -> float:
    if not hypotheses:
        return 0.0

    clipped_counts = [0] * max_n
    total_counts = [0] * max_n
    reference_length = 0
    hypothesis_length = 0

    for refs, hyp in zip(references, hypotheses):
        hypothesis_length += len(hyp)
        reference_lengths = [len(ref) for ref in refs]
        reference_length += min(reference_lengths, key=lambda ref_len: (abs(ref_len - len(hyp)), ref_len))

        for n in range(1, max_n + 1):
            hyp_ngrams = _extract_ngrams(hyp, n)
            total_counts[n - 1] += sum(hyp_ngrams.values())

            if not hyp_ngrams:
                continue

            max_ref_counts = Counter()
            for ref in refs:
                ref_ngrams = _extract_ngrams(ref, n)
                for ngram, count in ref_ngrams.items():
                    max_ref_counts[ngram] = max(max_ref_counts[ngram], count)

            clipped_counts[n - 1] += sum(
                min(count, max_ref_counts[ngram]) for ngram, count in hyp_ngrams.items()
            )

    precisions = []
    for clipped, total in zip(clipped_counts, total_counts):
        if total == 0:
            return 0.0
        precisions.append(clipped / total)

    if min(precisions) == 0:
        return 0.0

    brevity_penalty = 1.0
    if hypothesis_length < reference_length:
        brevity_penalty = math.exp(1.0 - reference_length / max(hypothesis_length, 1))

    bleu = brevity_penalty * math.exp(sum(math.log(p) for p in precisions) / max_n)
    return 100.0 * bleu


# ══════════════════════════════════════════════════════════════════════
#   BLEU EVALUATION  
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Args:
        model           : Trained Transformer (in eval mode).
        test_dataloader : DataLoader over the test split.
                          Each batch yields (src, tgt) token-index tensors.
        tgt_vocab       : Vocabulary object with idx_to_token mapping.
                          Must support  tgt_vocab.itos[idx]  or
                          tgt_vocab.lookup_token(idx).
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).
    """
    start_symbol = _lookup_vocab_index(tgt_vocab, "<sos>")
    end_symbol = _lookup_vocab_index(tgt_vocab, "<eos>")
    pad_idx = _lookup_vocab_index(tgt_vocab, "<pad>")

    references: list[list[list[str]]] = []
    hypotheses: list[list[str]] = []

    was_training = model.training
    model.eval()

    with torch.no_grad():
        for src_batch, tgt_batch in test_dataloader:
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            for src, tgt in zip(src_batch, tgt_batch):
                src = src.unsqueeze(0)
                tgt = tgt.unsqueeze(0)
                src_mask = make_src_mask(src, pad_idx=pad_idx)
                prediction = greedy_decode(
                    model,
                    src,
                    src_mask,
                    max_len=max_len,
                    start_symbol=start_symbol,
                    end_symbol=end_symbol,
                    device=device,
                )

                hypotheses.append(_strip_special_tokens(prediction.squeeze(0).tolist(), tgt_vocab))
                references.append([_strip_special_tokens(tgt.squeeze(0).tolist(), tgt_vocab)])

    model.train(was_training)
    return _corpus_bleu(references, hypotheses)


# ══════════════════════════════════════════════════════════════════════
# ➅  CHECKPOINT UTILITIES  (autograder loads your model from disk)
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimiser + scheduler state to disk.

    The autograder will call load_checkpoint to restore your model.
    Do NOT change the keys in the saved dict.

    Args:
        model     : Transformer instance.
        optimizer : Optimizer instance.
        scheduler : NoamScheduler instance.
        epoch     : Current epoch number.
        path      : File path to save to (default 'checkpoint.pt').

    Saves a dict with keys:
        'epoch', 'model_state_dict', 'optimizer_state_dict',
        'scheduler_state_dict', 'model_config'

    model_config must contain all kwargs needed to reconstruct
    Transformer(**model_config), e.g.:
        {'src_vocab_size': ..., 'tgt_vocab_size': ...,
         'd_model': ..., 'N': ..., 'num_heads': ...,
         'd_ff': ..., 'dropout': ...}
    """
    model_config = getattr(model, "model_config", None)
    if model_config is None:
        raise AttributeError("Transformer model must expose `model_config` for checkpoint saving.")

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "model_config": model_config,
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.

    Args:
        path      : Path to checkpoint file saved by save_checkpoint.
        model     : Uninitialised Transformer with matching architecture.
        optimizer : Optimizer to restore (pass None to skip).
        scheduler : Scheduler to restore (pass None to skip).

    Returns:
        epoch : The epoch at which the checkpoint was saved (int).
    """
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return int(checkpoint["epoch"])


def visualize_attention(
    model: Transformer,
    dataset: "Multi30kDataset",
    device: str,
    save_path: str = "attention_maps.png",
):
    """
    Task 2.3: Visualize attention maps for the last encoder layer.
    """
    was_training = model.training
    model.eval()

    example = dataset[0]
    src, _ = example
    src = src.unsqueeze(0).to(device)
    src_mask = make_src_mask(src)
    
    with torch.no_grad():
        _ = model.encode(src, src_mask)
    
    # Get the last encoder layer's self-attention weights
    last_layer = model.encoder.layers[-1]
    attn_weights = last_layer.self_attn.attn_weights.squeeze(0).detach().cpu()
    
    num_heads = attn_weights.size(0)
    num_cols = min(4, num_heads)
    num_rows = ceil(num_heads / num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4.5 * num_rows))
    axes = axes.flatten()
    
    src_tokens = [dataset.src_vocab.lookup_token(idx.item()) for idx in src[0]]
    
    for i in range(num_heads):
        ax = axes[i]
        im = ax.imshow(attn_weights[i].numpy(), cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(f"Head {i+1}")
        ax.set_xticks(range(len(src_tokens)))
        ax.set_xticklabels(src_tokens, rotation=90)
        ax.set_yticks(range(len(src_tokens)))
        ax.set_yticklabels(src_tokens)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[num_heads:]:
        ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Attention maps saved to {save_path}")
    model.train(was_training)
    
    wandb_module = _ensure_wandb()
    if wandb_module is not None and wandb_module.run is not None:
        try:
            wandb_module.log({"attention_maps": wandb_module.Image(save_path)})
        except Exception as e:
            print(f"Warning: W&B attention maps log failed: {e}")


# ══════════════════════════════════════════════════════════════════════
#   EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    """
    Set up and run the full training experiment.

    Steps:
        1. Init W&B:   wandb.init(project="da6401-a3", config={...})
        2. Build dataset / vocabs from dataset.py
        3. Create DataLoaders for train / val splits
        4. Instantiate Transformer with hyperparameters from config
        5. Instantiate Adam optimizer (β1=0.9, β2=0.98, ε=1e-9)
        6. Instantiate NoamScheduler(optimizer, d_model, warmup_steps=4000)
        7. Instantiate LabelSmoothingLoss(vocab_size, pad_idx, smoothing=0.1)
        8. Training loop:
               for epoch in range(num_epochs):
                   run_epoch(train_loader, model, loss_fn,
                             optimizer, scheduler, epoch, is_train=True)
                   run_epoch(val_loader, model, loss_fn,
                             None, None, epoch, is_train=False)
                   save_checkpoint(model, optimizer, scheduler, epoch)
        9. Final BLEU on test set:
               bleu = evaluate_bleu(model, test_loader, tgt_vocab)
               wandb.log({'test_bleu': bleu})
    """
    from dataset import Multi30kDataset, collate_batch
    from lr_scheduler import NoamScheduler

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--N", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--scheduler", type=str, default="noam", choices=["noam", "fixed"])
    parser.add_argument("--use_scale", action="store_true", default=True)
    parser.add_argument("--no_scale", action="store_false", dest="use_scale")
    parser.add_argument("--pos_encoding", type=str, default="sinusoidal", choices=["sinusoidal", "learned"])
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--project", type=str, default="da6401-a3")
    parser.add_argument("--grad_log_steps", type=int, default=1000)
    parser.add_argument("--val_bleu_every", type=int, default=1)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    config = vars(args)
    config["device"] = device
    ls_suffix = "" if args.label_smoothing == 0.1 else f"_ls{args.label_smoothing}"
    experiment_name = f"{args.scheduler}_{args.pos_encoding}_{'scale' if args.use_scale else 'no_scale'}{ls_suffix}"
    config["checkpoint_path"] = f"checkpoint_{experiment_name}.pt"

    # Only log step-level gradients for the scaling ablation (Exp 2: baseline and unscaled)
    # Disable for other experiments to avoid network socket overhead
    is_scaling_ablation = (
        args.scheduler == "noam"
        and args.pos_encoding == "sinusoidal"
        and args.label_smoothing == 0.1
    )
    if not is_scaling_ablation and args.grad_log_steps == 1000:
        config["grad_log_steps"] = 0
    
    # Store metrics locally to prevent W&B socket crashes
    local_metrics = []

    wandb_run = None
    wandb_module = _ensure_wandb()
    if wandb_module is not None:
        init_kwargs = {"project": args.project, "config": config}
        if "WANDB_MODE" in os.environ:
            init_kwargs["mode"] = os.environ["WANDB_MODE"]
        wandb_run = wandb_module.init(**init_kwargs)
        config = dict(wandb_module.config)

    print("Initializing datasets and vocabulary...")
    train_dataset = Multi30kDataset(
        split="train",
        min_freq=config["min_freq"],
        max_length=config["max_length"],
    )
    val_dataset = Multi30kDataset(
        split="validation",
        src_vocab=train_dataset.src_vocab,
        tgt_vocab=train_dataset.tgt_vocab,
        min_freq=config["min_freq"],
        max_length=config["max_length"],
    )
    test_dataset = Multi30kDataset(
        split="test",
        src_vocab=train_dataset.src_vocab,
        tgt_vocab=train_dataset.tgt_vocab,
        min_freq=config["min_freq"],
        max_length=config["max_length"],
    )

    pad_idx = train_dataset.tgt_vocab.pad_idx

    def collate_fn(batch):
        return collate_batch(batch, pad_idx=pad_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    print(f"Building Transformer model (d_model={config['d_model']}, N={config['N']}, pos={config['pos_encoding']}, scale={config['use_scale']})...")
    model = Transformer(
        src_vocab_size=len(train_dataset.src_vocab),
        tgt_vocab_size=len(train_dataset.tgt_vocab),
        d_model=config["d_model"],
        N=config["N"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
        pos_encoding_type=config["pos_encoding"],
        use_scale=config["use_scale"],
    ).to(device)
    print("Model built and moved to device.")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr if args.scheduler == "fixed" else 1.0,
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    
    scheduler = None
    if args.scheduler == "noam":
        scheduler = NoamScheduler(
            optimizer,
            d_model=config["d_model"],
            warmup_steps=config["warmup_steps"],
        )
    loss_fn = LabelSmoothingLoss(
        vocab_size=len(train_dataset.tgt_vocab),
        pad_idx=pad_idx,
        smoothing=config["label_smoothing"],
    ).to(device)

    print(f"Starting training for {config['num_epochs']} epochs...")
    best_val_loss = float("inf")
    global_step = 0
    for epoch in range(config["num_epochs"]):
        train_loss, train_conf, train_acc, global_step, grad_buffer = run_epoch(
            train_loader,
            model,
            loss_fn,
            optimizer,
            scheduler=scheduler,
            epoch_num=epoch,
            is_train=True,
            device=device,
            wandb_module=wandb_module,
            global_step_start=global_step,
            grad_log_steps=config["grad_log_steps"],
        )
        val_loss, val_conf, val_acc, _, _ = run_epoch(
            val_loader,
            model,
            loss_fn,
            optimizer=None,
            scheduler=None,
            epoch_num=epoch,
            is_train=False,
            device=device,
        )

        val_bleu = None
        if config["val_bleu_every"] > 0 and (epoch + 1) % config["val_bleu_every"] == 0:
            val_bleu = evaluate_bleu(
                model,
                val_loader,
                train_dataset.tgt_vocab,
                device=device,
                max_len=config["max_length"],
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, path=config["checkpoint_path"])

        log_payload = {
            "epoch": epoch + 1,
            "train_step": global_step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "train_prediction_confidence": train_conf,
            "val_prediction_confidence": val_conf,
            "lr": optimizer.param_groups[0]["lr"],
        }
        if val_bleu is not None:
            log_payload["val_bleu"] = val_bleu

        # Always print epoch summary to console
        print(f"\n[Epoch {epoch + 1} Summary] "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%"
              f"{f' | Val BLEU: {val_bleu:.2f}' if val_bleu is not None else ''}")

        # Single epoch-level W&B log — no step-level calls
        if wandb_run is not None:
            try:
                wandb.log(log_payload)
            except Exception as e:
                print(f"Warning: W&B epoch log failed: {e}")

        local_epoch_entry = {
            "epoch": epoch + 1,
            "train_step": global_step,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "train/accuracy": train_acc,
            "val/accuracy": val_acc,
            "train/prediction_confidence": train_conf,
            "val/prediction_confidence": val_conf,
            "lr": optimizer.param_groups[0]["lr"],
        }
        if val_bleu is not None:
            local_epoch_entry["val/bleu"] = val_bleu
        # Embed gradient norm buffer inline so everything is in one file
        if grad_buffer:
            local_epoch_entry["grad_norms"] = grad_buffer
        local_metrics.append(local_epoch_entry)

        import json
        with open(f"metrics_{experiment_name}.json", "w") as f:
            json.dump(local_metrics, f, indent=2)

        # Task 2.3: Visualization
        if args.visualize and (epoch == config["num_epochs"] - 1):
            visualize_attention(model, train_dataset, device, save_path=f"attention_maps_{experiment_name}.png")

    if os.path.exists(config["checkpoint_path"]):
        load_checkpoint(config["checkpoint_path"], model, optimizer=None, scheduler=None)

    bleu = evaluate_bleu(
        model,
        test_loader,
        train_dataset.tgt_vocab,
        device=device,
        max_len=config["max_length"],
    )

    # Save test BLEU locally so it is preserved
    local_metrics.append({"test_bleu": bleu})
    import json
    with open(f"metrics_{experiment_name}.json", "w") as f:
        json.dump(local_metrics, f)

    print(f"\nFinal Test BLEU: {bleu:.2f}")

    if wandb_run is not None:
        try:
            wandb.log({"test_bleu": bleu})
        except Exception as e:
            print(f"Warning: W&B final log failed: {e}")
        try:
            wandb_run.finish()
        except Exception as e:
            print(f"Warning: W&B finish failed: {e}")


if __name__ == "__main__":
    run_training_experiment()
