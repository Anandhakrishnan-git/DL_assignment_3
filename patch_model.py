import sys
import os

with open('model.py', 'r', encoding='utf-8') as f:
    content = f.read()

with open('vocab_b64.txt', 'r', encoding='ascii') as f:
    b64_str = f.read().strip()

start_str = "        # Load vocabularies lazily to save time during multiple inferences"
end_str = "        dataset = Transformer._cached_dataset"

start_idx = content.find(start_str)
end_idx = content.find(end_str)

if start_idx != -1 and end_idx != -1:
    end_idx += len(end_str)
    
    replacement = f"""        # Load vocabularies lazily to save time during multiple inferences
        if Transformer._cached_dataset is None:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            
            loaded = False
            import torch
            import glob
            # Try to load from checkpoint first
            for pt_file in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "*.pt")):
                try:
                    ckpt = torch.load(pt_file, map_location="cpu", weights_only=False)
                    if "src_vocab" in ckpt and "tgt_vocab" in ckpt:
                        from dataset import _load_spacy_tokenizer
                        class DummyDataset: pass
                        Transformer._cached_dataset = DummyDataset()
                        Transformer._cached_dataset.src_vocab = ckpt["src_vocab"]
                        Transformer._cached_dataset.tgt_vocab = ckpt["tgt_vocab"]
                        Transformer._cached_dataset.src_tokenizer = _load_spacy_tokenizer("de")
                        loaded = True
                        break
                except:
                    pass
            
            if not loaded:
                try:
                    from dataset import SimpleVocab, _load_spacy_tokenizer
                    import pickle, zlib, base64
                    b64_data = "{b64_str}"
                    data = pickle.loads(zlib.decompress(base64.b64decode(b64_data)))
                    class MockCounter:
                        def items(self): return []
                    src_v = SimpleVocab(MockCounter(), min_freq=1)
                    src_v.itos = data["src_itos"]
                    src_v.stoi = {{tok: i for i, tok in enumerate(src_v.itos)}}
                    tgt_v = SimpleVocab(MockCounter(), min_freq=1)
                    tgt_v.itos = data["tgt_itos"]
                    tgt_v.stoi = {{tok: i for i, tok in enumerate(tgt_v.itos)}}
                    class DummyDataset: pass
                    Transformer._cached_dataset = DummyDataset()
                    Transformer._cached_dataset.src_vocab = src_v
                    Transformer._cached_dataset.tgt_vocab = tgt_v
                    Transformer._cached_dataset.src_tokenizer = _load_spacy_tokenizer("de")
                    loaded = True
                except:
                    pass

            if not loaded:
                from dataset import Multi30kDataset
                Transformer._cached_dataset = Multi30kDataset(split="test", min_freq=2, max_length=100)
                
        dataset = Transformer._cached_dataset"""
        
    content = content[:start_idx] + replacement + content[end_idx:]
    with open('model.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('Successfully patched model.py')
else:
    print('Could not find target strings in model.py')
