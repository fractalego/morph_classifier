def get_vector(i,vocab_size):
    vect = [0]*vocab_size
    vect[i] = 1
    return vect

def convert_string_to_ints (sentence, vocab_size, max_length):
    integer_vector = [get_vector(ord(sentence[i]),vocab_size) for i in range (len(sentence))]
    if len(integer_vector) >= max_length:
        integer_vector = integer_vector[:max_length]
    for i in range(max_length - len(integer_vector)):
        integer_vector.append([0]*vocab_size)
    return integer_vector

def clean_string(line):
    line = line.replace('\n','')
    line = line.lower()
    return line
