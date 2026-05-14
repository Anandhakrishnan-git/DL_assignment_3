from collections import Counter
from typing import Callable, Iterable

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm


SPECIAL_TOKENS = ("<unk>", "<pad>", "<sos>", "<eos>")


class SimpleVocab:
    def __init__(self, counter: Counter, min_freq: int = 1) -> None:
        self.itos = list(SPECIAL_TOKENS)
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}

        for token, freq in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
            if freq >= min_freq and token not in self.stoi:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)

        self.idx_to_token = self.itos
        self.token_to_idx = self.stoi
        self.unk_idx = self.stoi["<unk>"]
        self.pad_idx = self.stoi["<pad>"]
        self.sos_idx = self.stoi["<sos>"]
        self.eos_idx = self.stoi["<eos>"]

    def __len__(self) -> int:
        return len(self.itos)

    def lookup_token(self, idx: int) -> str:
        return self.itos[idx]

    def lookup_indices(self, tokens: Iterable[str]) -> list[int]:
        return [self.stoi.get(token, self.unk_idx) for token in tokens]


def _normalize_split_name(split: str) -> str:
    aliases = {
        "val": "validation",
        "valid": "validation",
        "dev": "validation",
    }
    return aliases.get(split, split)


def _load_spacy_tokenizer(language_code: str) -> Callable[[str], list[str]]:
    try:
        import spacy
    except ImportError as exc:
        raise ImportError(
            "spaCy is required for the Multi30k dataset pipeline. "
            "Install it with `pip install spacy`."
        ) from exc

    model_candidates = {
        "de": ["de_core_news_sm", "de_core_news_md"],
        "en": ["en_core_web_sm", "en_core_web_md"],
    }

    nlp = None
    for model_name in model_candidates.get(language_code, []):
        try:
            nlp = spacy.load(model_name)
            break
        except OSError:
            continue

    if nlp is None:
        nlp = spacy.blank(language_code)

    def tokenize(text: str) -> list[str]:
        return [token.text.lower() for token in nlp.tokenizer(text.strip()) if token.text.strip()]

    return tokenize


def _load_split(split: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The `datasets` package is required to load Multi30k. "
            "Install it with `pip install datasets`."
        ) from exc

    return load_dataset("bentrevett/multi30k", split=_normalize_split_name(split))


def _extract_translation_pair(example: dict) -> tuple[str, str]:
    if "translation" in example:
        translation = example["translation"]
        return translation["de"], translation["en"]
    if "de" in example and "en" in example:
        return example["de"], example["en"]
    raise KeyError("Unable to find German-English translation fields in the dataset example.")


class Multi30kDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        src_vocab: SimpleVocab | None = None,
        tgt_vocab: SimpleVocab | None = None,
        min_freq: int = 2,
        max_length: int | None = None,
    ):
        """
        Loads the Multi30k dataset and prepares tokenizers.
        """
        self.split = _normalize_split_name(split)
        self.min_freq = min_freq
        self.max_length = max_length
        self.src_tokenizer = _load_spacy_tokenizer("de")
        self.tgt_tokenizer = _load_spacy_tokenizer("en")
        
        print(f"Loading Multi30k dataset (split: {self.split})...")
        self.raw_dataset = _load_split(self.split)

        if src_vocab is None or tgt_vocab is None:
            self.src_vocab, self.tgt_vocab = self.build_vocab()
        else:
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab

        self.examples = self.process_data()

    def build_vocab(self) -> tuple[SimpleVocab, SimpleVocab]:
        """
        Builds the vocabulary mapping for src (de) and tgt (en), including:
        <unk>, <pad>, <sos>, <eos>
        """
        print("Building vocabulary from training split...")
        train_split = _load_split("train")
        src_counter: Counter = Counter()
        tgt_counter: Counter = Counter()

        for example in tqdm(train_split, desc="Building Vocab", unit="ex"):
            src_text, tgt_text = _extract_translation_pair(example)
            src_counter.update(self.src_tokenizer(src_text))
            tgt_counter.update(self.tgt_tokenizer(tgt_text))

        return SimpleVocab(src_counter, self.min_freq), SimpleVocab(tgt_counter, self.min_freq)

    def process_data(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Convert English and German sentences into integer token lists using
        spaCy and the defined vocabulary.
        """
        examples: list[tuple[torch.Tensor, torch.Tensor]] = []

        for example in tqdm(self.raw_dataset, desc=f"Processing {self.split}", unit="ex"):
            src_text, tgt_text = _extract_translation_pair(example)
            src_tokens = self.src_tokenizer(src_text)
            tgt_tokens = self.tgt_tokenizer(tgt_text)

            if self.max_length is not None:
                usable_length = max(self.max_length - 2, 0)
                src_tokens = src_tokens[:usable_length]
                tgt_tokens = tgt_tokens[:usable_length]

            src_ids = [self.src_vocab.sos_idx]
            src_ids += self.src_vocab.lookup_indices(src_tokens)
            src_ids += [self.src_vocab.eos_idx]

            tgt_ids = [self.tgt_vocab.sos_idx]
            tgt_ids += self.tgt_vocab.lookup_indices(tgt_tokens)
            tgt_ids += [self.tgt_vocab.eos_idx]

            examples.append(
                (
                    torch.tensor(src_ids, dtype=torch.long),
                    torch.tensor(tgt_ids, dtype=torch.long),
                )
            )

        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.examples[index]


def collate_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
    pad_idx: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    src_batch, tgt_batch = zip(*batch)
    src_batch = [torch.as_tensor(src, dtype=torch.long) for src in src_batch]
    tgt_batch = [torch.as_tensor(tgt, dtype=torch.long) for tgt in tgt_batch]
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
    return src_padded, tgt_padded
