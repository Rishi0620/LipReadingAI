def greedy_decode(log_probs, idx2char):
    """
    log_probs: Tensor of shape [T, B, C] from model output
    idx2char: Dictionary mapping index to character
    """
    log_probs = log_probs.cpu().detach().numpy()
    sequences = []

    for b in range(log_probs.shape[1]):  # for each batch item
        best_path = log_probs[:, b, :].argmax(axis=-1)  # [T]
        prev = None
        sequence = []
        for idx in best_path:
            if idx != prev and idx != 0:  # remove repeated and blank (index 0 = '-')
                sequence.append(idx)
            prev = idx
        decoded = ''.join([idx2char[i] for i in sequence])
        sequences.append(decoded)
    return sequences