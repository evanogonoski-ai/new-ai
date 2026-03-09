import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors


TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'tokenizer.json')


def train_tokenizer(texts: list, vocab_size: int = 8192, save_path: str = None) -> Tokenizer:
    """
    Train a BPE tokenizer on provided texts.

    Args:
        texts: list of text strings
        vocab_size: vocabulary size
        save_path: where to save the tokenizer
    Returns:
        trained Tokenizer
    """
    if save_path is None:
        save_path = TOKENIZER_PATH

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        min_frequency=2,
    )

    tokenizer.train_from_iterator(texts, trainer=trainer)

    # Add post-processing for BOS/EOS
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"<bos>:0 $A:0 <eos>:0",
        pair=f"<bos>:0 $A:0 <eos>:0 <bos>:1 $B:1 <eos>:1",
        special_tokens=[
            ("<bos>", bos_id),
            ("<eos>", eos_id),
        ],
    )

    # Enable padding
    tokenizer.enable_padding(pad_id=0, pad_token="<pad>")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokenizer.save(save_path)
    return tokenizer


def load_tokenizer(path: str = None) -> Tokenizer:
    """Load a saved tokenizer."""
    if path is None:
        path = TOKENIZER_PATH
    return Tokenizer.from_file(path)
