import torch


def get_embedding(sequence: str, esm_model, esm_alphabet, device: str) -> torch.Tensor:
    batch_converter = esm_alphabet.get_batch_converter()
    
    seqs = list([("seq", s) for s in [sequence]])
    labels, strs, toks = batch_converter(seqs)
    
    repr_layers = esm_model.num_layers
    
    toks = toks.to(device)


    minibatch_max_length = toks.size(1)

    # if sequence is short, embed it directly
    if minibatch_max_length <= 1024:
        with torch.no_grad():
            results = esm_model(toks, repr_layers=[repr_layers], return_contacts=False)
        embedding = results["representations"][repr_layers][0, 1:-1]
        return embedding.to(device)

    # if sequence is long, embed in chunks
    tokens_list = []
    end = 0
    while end < minibatch_max_length:
        start = end
        
        # define chunk end
        end = start + 1022
        if end >= minibatch_max_length:
            end = minibatch_max_length
        
        current_toks = toks[:, start:end]
        
        with torch.no_grad():
            tokens = esm_model(current_toks.to('cpu'), repr_layers=[repr_layers], return_contacts=False)["representations"][repr_layers]
            
        # for the first chunk, remove BOS token, for the last chunk, remove EOS token
        if start == 0 and end < minibatch_max_length:
            tokens_list.append(tokens[:, 1:, :])
        elif start > 0 and end == minibatch_max_length:
            tokens_list.append(tokens[:, :-1, :])
        elif start == 0 and end == minibatch_max_length:
            tokens_list.append(tokens[:, 1:-1, :])
        else:
            tokens_list.append(tokens)

    out = torch.cat(tokens_list, dim=1)

    # set nan to zeros
    out[out!=out] = 0.0
    
    embedding = out[0, :, :]
    
    return embedding.to(device)