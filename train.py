from model import NameClassifier as Model
from aux import convert_string_to_ints, clean_string
import random


if __name__ == '__main__':
    vocab_size = 256
    seq_length = 32
    model = Model(seq_length = seq_length, vocab_size = vocab_size, memory_dim = 512, batch_size = 1)

    file = open('data/train_names.txt')
    lines = file.readlines()
    data = []

    for line in lines:
        line = clean_string(line)
        X = convert_string_to_ints(line, vocab_size, seq_length)
        y = [[1,0]]
        data.append([X,y])
        
    file = open('data/non_names.txt')
    lines = file.readlines()
    for line in lines:
        line = clean_string(line)
        X = convert_string_to_ints(line, vocab_size, seq_length)
        y = [[0,1]]
        data.append([X,y])
        
    data = sorted(data, key= lambda x: random.random())
    
    try:
        model.train(data)
    except:
        pass
        
    model.save('true_names.nn')
