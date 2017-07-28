from model import NameClassifier as Model


if __name__ == '__main__':
    model = Model.load('model.nn')

    file = open('data/test_names.txt')
    lines = file.readlines()
    tot_names = float(len(lines))
    num_classified_names = 0
    for line in lines:
        if model.classify(line):
            num_classified_names += 1
    print 'Names classified correctly from the test set: %.1f%%' % (num_classified_names/tot_names*100)
    file.close()
    
    file = open('data/test_non_names.txt')
    lines = file.readlines()
    tot_non_names = float(len(lines))
    num_classified_non_names = 0
    for line in lines:
        if not model.classify(line) :
            num_classified_non_names += 1
    print 'Non names classified correctly from the test set: %.1f%%' % (num_classified_non_names/tot_non_names*100)
    file.close()
