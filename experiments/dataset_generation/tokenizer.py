import itertools

def tokenize_sequences(list_of_sequences, embed="one_hot", kmer=1):
    """
    Tokenize sequences with one-hot encoding.
    """
    big_tokens = []
    letters = ["A", "C", "G", "T"]
    kmer_list = [''.join(i) for i in itertools.product(letters, repeat=kmer)]
    dict_vocab = {}

    if embed == "one_hot":
        # Generate k-mers
        kmer_length = len(kmer_list)
        for i in range(kmer_length):
            empty = [0]*kmer_length
            empty[i] = 1
            dict_vocab[kmer_list[i]] = empty
    elif embed == "ENAC":
        # Generate k-mers
        kmer_length = 4
        for i, km in enumerate(kmer_list):
            to_fill = [0]*kmer_length
            for letter in km:
                j = letters.index(letter)
                to_fill[j] += 1

            dict_vocab[km] = to_fill

    elif embed == "embeddings": #this is only for the pre-trained transformer training
        kmer_list = ["P", "A", "C", "G", "T"] #pad_token_idx = 0
        dict_vocab = {k: i for i, k in enumerate(kmer_list)}
        assert kmer == 1, "kmer must be 1 for embeddings"
        for seq in list_of_sequences:
            seq = seq.upper()
            #kmerize
            seq_tokens = [dict_vocab[seq[i]] for i in range(len(seq))]
            big_tokens.append(seq_tokens)
        
        return big_tokens
    
    for seq in list_of_sequences:
        seq = seq.upper()
        #kmerize
        seq = [seq[i:i+kmer] for i in range(len(seq)-kmer+1)]
        seq_tokens = []
        for element in seq:
            if "P" in element:
                seq_tokens.append([0]*kmer_length)
            else:
                seq_tokens.append(dict_vocab[element])

        big_tokens.append(seq_tokens)

    return big_tokens


