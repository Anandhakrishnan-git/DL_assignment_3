"""
train.py - Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"
"""

import math
import os
from collections import Counter
from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

try:
    import wandb
except ImportError:
    wandb = None

from model import Transformer, make_src_mask, make_tgt_mask


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need".
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


def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.
    """
    if is_train and optimizer is None:
        raise ValueError("An optimizer is required when is_train=True.")

    pad_idx = getattr(loss_fn, "pad_idx", 1)
    model.train(is_train)

    total_loss = 0.0
    total_tokens = 0

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
            
            # Task 2.2: Log gradient norms
            if scheduler is not None and hasattr(scheduler, "d_model") and wandb is not None and wandb.run is not None:
                q_grad = model.encoder.layers[0].self_attn.W_q.weight.grad
                k_grad = model.encoder.layers[0].self_attn.W_k.weight.grad
                if q_grad is not None and k_grad is not None:
                    wandb.log({
                        "grad_norm_Wq": q_grad.norm().item(),
                        "grad_norm_Wk": k_grad.norm().item(),
                    }, commit=False)
            
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        non_pad_tokens = int(decoder_target.ne(pad_idx).sum().item())
        total_loss += loss.item() * max(non_pad_tokens, 1)
        total_tokens += max(non_pad_tokens, 1)
        
        pbar.set_postfix(loss=loss.item())

    return total_loss / max(total_tokens, 1)


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


def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.
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


def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimiser + scheduler state to disk.
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
    model.eval()
    # Pick a sample sentence
    example = dataset[0]
    src, _ = example
    src = src.unsqueeze(0).to(device)
    src_mask = make_src_mask(src)
    
    with torch.no_grad():
        _ = model.encode(src, src_mask)
    
    # Get the last encoder layer's self-attention weights
    last_layer = model.encoder.layers[-1]
    attn_weights = last_layer.self_attn.attn_weights.squeeze(0) # [num_heads, src_len, src_len]
    
    num_heads = attn_weights.size(0)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    src_tokens = [dataset.src_vocab.lookup_token(idx.item()) for idx in src[0]]
    
    for i in range(num_heads):
        ax = axes[i]
        im = ax.imshow(attn_weights[i].cpu().numpy(), cmap="viridis")
        ax.set_title(f"Head {i+1}")
        ax.set_xticks(range(len(src_tokens)))
        ax.set_xticklabels(src_tokens, rotation=90)
        ax.set_yticks(range(len(src_tokens)))
        ax.set_yticklabels(src_tokens)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Attention maps saved to {save_path}")
    
    import wandb
    if wandb is not None and wandb.run is not None:
        wandb.log({"attention_maps": wandb.Image(save_path)})


def run_training_experiment() -> None:
    """
    Set up and run the full training experiment.
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
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    config = vars(args)
    config["device"] = device
    config["checkpoint_path"] = f"checkpoint_{args.scheduler}_{args.pos_encoding}_{'scale' if args.use_scale else 'no_scale'}.pt"

    wandb_run = None
    if wandb is not None:
        init_kwargs = {"project": args.project, "config": config}
        if "WANDB_MODE" in os.environ:
            init_kwargs["mode"] = os.environ["WANDB_MODE"]
        wandb_run = wandb.init(**init_kwargs)
        config = dict(wandb.config)

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
    for epoch in range(config["num_epochs"]):
        train_loss = run_epoch(
            train_loader,
            model,
            loss_fn,
            optimizer,
            scheduler=scheduler,
            epoch_num=epoch,
            is_train=True,
            device=device,
        )
        val_loss = run_epoch(
            val_loader,
            model,
            loss_fn,
            optimizer=None,
            scheduler=None,
            epoch_num=epoch,
            is_train=False,
            device=device,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, path=config["checkpoint_path"])

        log_payload = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
        }

        if wandb_run is not None:
            wandb.log(log_payload)
        else:
            print(log_payload)
            
        # Task 2.3: Visualization
        if args.visualize and (epoch == config["num_epochs"] - 1):
            visualize_attention(model, train_dataset, device)

    if os.path.exists(config["checkpoint_path"]):
        load_checkpoint(config["checkpoint_path"], model, optimizer=None, scheduler=None)

    bleu = evaluate_bleu(
        model,
        test_loader,
        train_dataset.tgt_vocab,
        device=device,
        max_len=config["max_length"],
    )

    if wandb_run is not None:
        wandb.log({"test_bleu": bleu})
        wandb_run.finish()
    else:
        print({"test_bleu": bleu})


if __name__ == "__main__":
    run_training_experiment()
