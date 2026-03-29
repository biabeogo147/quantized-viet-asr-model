import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file="assets/zipformer/bpe.model")

with open("assets/zipformer/tokens.txt", "w", encoding="utf-8") as f:
    for i in range(sp.get_piece_size()):
        f.write(f"{sp.id_to_piece(i)} {i}\n")