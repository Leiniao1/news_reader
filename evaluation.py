import numpy as np

def evaluation(encoder_and_decoder, ids, reserved_tokens):
    if not reserved_tokens:
        return encoder_and_decoder.decode(ids.flatten().tolist())
    tokens = np.where(ids < reserved_tokens)[0]
    if not tokens.size:
        return encoder_and_decoder.decode(ids.flatten().tolist())
    else:
        #[1000, 37189, 23, 1000, 40, 1] -> [Hello Word] ["<23>"] [Hello] ["<40>"].
        #[1000, 37189] [23, 1000] [40] [1]
        split_locations = np.union1d(tokens, tokens + 1)
        ids_list = np.split(ids, split_locations)
        text_list = [
            "<%d>" %
            i if len(i) == 1 and i < reserved_tokens else encoder_and_decoder.decode(i.tolist())
            for i in ids_list
            ]
        return " ".join(text_list)
