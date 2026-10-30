import pytest
import torch

from mt.basic import BasicTransformer, train_tokenizer, encode, BOS, EOS
from mt.prepare import clean_pairs, normalize
from mt.evaluate import metrics


def test_deduplication_and_normalization():
    rows = [{"translation": {"en": a, "fr": b}} for a, b in [
        ("Hello  world", "Bonjour monde"), ("HELLO world", "Salut monde"),
        ("Good morning", "Bonjour monde"), ("A small cat", "Un petit chat"), ("", "rien")]]
    clean = clean_pairs(rows, 60)
    assert len(clean) == 2
    assert normalize(" cafe\u0301 \n") == "café"


def test_causal_mask_does_not_leak_future_tokens():
    torch.manual_seed(1)
    model = BasicTransformer(32, d_model=16, nhead=2, layers=1, feedforward=32, dropout=0).eval()
    source = torch.tensor([[1, 4, 5, 2]])
    a, b = torch.tensor([[1, 6, 7]]), torch.tensor([[1, 6, 9]])
    with torch.no_grad():
        first, second = model(source, a), model(source, b)
    assert torch.allclose(first[:, :2], second[:, :2], atol=1e-6)


def test_padding_does_not_change_unpadded_output():
    torch.manual_seed(2)
    model = BasicTransformer(32, d_model=16, nhead=2, layers=1, feedforward=32, dropout=0).eval()
    target = torch.tensor([[1, 6]])
    with torch.no_grad():
        a = model(torch.tensor([[1, 4, 2]]), target)
        b = model(torch.tensor([[1, 4, 2, 0, 0]]), target)
    assert torch.allclose(a, b, atol=1e-5)


def test_tokenizer_and_optimizer_step():
    rows = [{"source": "The cat is here.", "reference": "Le chat est ici."}]
    tokenizer = train_tokenizer(rows, 300)
    source = torch.tensor([encode(tokenizer, rows[0]["source"], 32)])
    target = torch.tensor([encode(tokenizer, rows[0]["reference"], 32)])
    assert source[0, 0] == BOS and source[0, -1] == EOS
    model = BasicTransformer(tokenizer.get_vocab_size(), d_model=16, nhead=2, layers=1, feedforward=32, dropout=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    initial = model.output.weight.detach().clone()
    loss = torch.nn.functional.cross_entropy(model(source, target[:, :-1]).flatten(0, 1), target[:, 1:].flatten())
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert not torch.equal(initial, model.output.weight)
    result = model.generate(source, 5)
    assert result.shape[1] <= 6


def test_metrics_and_alignment_guard():
    reference = ["Le petit chat noir dort sur le tapis."]
    assert metrics(reference, reference)["bleu"] == pytest.approx(100)
    assert metrics(reference, reference)["chrf"] == pytest.approx(100)
    with pytest.raises(ValueError):
        metrics([], reference)


def test_huggingface_architecture_apis_without_downloads():
    # Tiny random models verify API compatibility, not pretrained-model quality.
    from transformers import MT5Config, MT5ForConditionalGeneration, LlamaConfig, LlamaForCausalLM
    from transformers import Gemma3TextConfig, Gemma3ForCausalLM
    mt5 = MT5ForConditionalGeneration(MT5Config(vocab_size=32, d_model=16, d_kv=8,
        d_ff=32, num_layers=1, num_decoder_layers=1, num_heads=2,
        decoder_start_token_id=0, pad_token_id=0, eos_token_id=2))
    source = torch.tensor([[1, 4, 2]])
    loss = mt5(input_ids=source, labels=torch.tensor([[6, 2, -100]])).loss
    loss.backward()
    assert torch.isfinite(loss)
    for cls, cfg_cls in [(LlamaForCausalLM, LlamaConfig), (Gemma3ForCausalLM, Gemma3TextConfig)]:
        cfg = cfg_cls(vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            head_dim=8, max_position_embeddings=64, bos_token_id=1, eos_token_id=2, pad_token_id=0)
        model = cls(cfg).eval()
        output = model.generate(source, attention_mask=torch.ones_like(source), max_new_tokens=2, do_sample=False)
        assert 3 < output.shape[1] <= 5
