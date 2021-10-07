import sentencepiece as spm
import sentencepiece.sentencepiece_model_pb2 as model

def _bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if chr(b) == ' ':
            bs.append(b)
            cs.append(ord("▁"))
            n += 1
            continue

        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

byte_encoder = _bytes_to_unicode()
byte_decoder = {v: k for k, v in byte_encoder.items()}
print(byte_encoder)

merges = []
with open("merges.txt") as f:
    for line in f:
        line = line.strip()
        if line.startswith("#"):
            continue

        line = line.replace("Ġ", "▁")
        merges.append(line.split(" "))

def get_piece(token, score, type_ = 1):
    new_piece = model.ModelProto.SentencePiece()
    new_piece.piece = token
    new_piece.score = score
    new_piece.type = type_
    return new_piece

m = model.ModelProto()
m.normalizer_spec.add_dummy_prefix = False
m.normalizer_spec.remove_extra_whitespaces = False
m.pieces.append(get_piece("<unk>", 0, model.ModelProto.SentencePiece.Type.UNKNOWN))
m.pieces.append(get_piece("<s>", 0, model.ModelProto.SentencePiece.Type.CONTROL))
m.pieces.append(get_piece("</s>", 0, model.ModelProto.SentencePiece.Type.CONTROL))
for index, merge in enumerate(merges):
    m.pieces.append(get_piece("".join(merge), index * -0.1 - 0.1))
for index, b in enumerate(byte_decoder.keys()):
    m.pieces.append(get_piece(b, index * -0.1 - len(merges) * -0.1 - 0.1))
m.trainer_spec.model_type = model.TrainerSpec.ModelType.BPE

with open("new.model", "wb") as f:
    f.write(m.SerializeToString())

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
sp = spm.SentencePieceProcessor(model_file="new.model")
texts = [
    "Hello World!",
    "안녕!",
    "1234!!?:)Helloaa😆🙇‍♂️",
    "abαβ 累計7239人",
    "xy.,z     de",
    "I've fused the lights.",
    "abαβ123🙇‍♂️🙇‍♂️🙇‍♂️🙇‍♂️          ",
]
for text in texts:
    print(sp.encode("".join([byte_encoder[c] for c in text.encode("utf-8")]), out_type=str))
    print(tokenizer.tokenize(text))
