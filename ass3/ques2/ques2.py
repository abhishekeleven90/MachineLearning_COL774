
# coding: utf-8

# In[19]:

import random
from collections import Counter
import math
import time

def chunks(l, n):
    n = max(1, n)
    return [l[i:i + n] for i in range(0, len(l), n)]

def getAllLines():
    filename=open('20ng-rec_talk.txt','r')
    categories = []
    vocab_set = set()
    list_of_all_freq_counters_per_doc = []
    original_counter = Counter()
    all_lines = filename.readlines()
    random.shuffle(all_lines) ##randomize
    return all_lines


def getVocabAndCategorySets(all_lines):
    categories = []
    vocab_set = set()
    for line in all_lines: ##vocab on entire data!
        line = line.split()
        categories.append(line[0])
        line = line[1:]
        set_line = set(line)
        vocab_set = vocab_set | set_line
    return vocab_set, set(categories)

def prepareTrainAndTestSet(all_lines, split_count = 5): ##Part D change here!
    (l1,l2,l3,l4,l5) = chunks(all_lines,len(all_lines)/split_count)
    train_list = []
    test_list = []
    
    ##5 is test
    train_list.append(l1+l2+l3+l4)
    test_list.append(l5)
    
    ##4 is test
    train_list.append(l1+l2+l3+l5)
    test_list.append(l4)
    
    ##3 is test
    train_list.append(l1+l2+l5+l4)
    test_list.append(l3)
    
    ##2 is test
    train_list.append(l1+l5+l3+l4)
    test_list.append(l2)
    
    ##1 is test
    train_list.append(l5+l2+l3+l4)
    test_list.append(l1)
        
    return train_list, test_list

def getProbForEactCategory(categories, categories_set):
    ##this is on the training set
    counter_category_all =  Counter(categories)
    #print counter_category_all ##can work directly on this using python i think!
    #print '\n'
    prob_cat_all = Counter()
    total_cat = len(categories)
    for cat in categories_set:
        prob_cat_all[cat] = counter_category_all[cat]/float(total_cat)
    return prob_cat_all ##OUR PVJ
#prob_cat_all = getProbForEactCategory(categories)
#print prob_cat_all

def getCoreDataStructures1(train_list, categories_set):
    ##now everything on train data
    categories = []
    category_lenth_Counter = Counter() ##total number of words in each category
    category_word_Counter = Counter() ##counter for each word per category
    for cat in categories_set:
        category_lenth_Counter[cat] = 0
        category_word_Counter[cat] = Counter()
    for line in train_list:
        line = line.split()
        presernt_cat = line[0]
        categories.append(presernt_cat)    
        line = line[1:]
        category_lenth_Counter[presernt_cat] = category_lenth_Counter[presernt_cat] + len(line) 
        counts_dict_present_doc = Counter(line)
        category_word_Counter[presernt_cat] = category_word_Counter[presernt_cat] + counts_dict_present_doc 
        #list_of_all_freq_counters_per_doc.append(counts_dict_each_doc)
        #original_counter = original_counter + counts_dict_each_doc
    return category_lenth_Counter, category_word_Counter, categories
# start = time.time()
# category_lenth_Counter, category_word_Counter, categories = getCoreDataStructures1(train_list, categories_set)
# end = time.time()
# print str(end-start) + ' seconds'

##categories_vocab_set is a bug, remove this argument from all places
def getCoreDataStructures2(vocab_set, categories_vocab_set, category_lenth_Counter, category_word_Counter, categories_set):
    dim1 = Counter() ##counter against words
    vocab_count = len(vocab_set)
    for wk in vocab_set:
            dim2 = Counter() ##counter against labels/categories
            for catk in categories_set:
                n = category_lenth_Counter[catk]
                nk = category_word_Counter[catk][wk]
                dim2[catk] =  float(nk+1)/float(n+vocab_count)
            dim1[wk] = dim2
    return dim1

def getTestAccuracy(test_list, dim1, categories_set, prob_cat_all):
    count = 0
    for line in test_list:
        line = line.split()
        presernt_cat = line[0] ##actually the category
        #print presernt_cat,
        line = line[1:]
        prob_Counter_per_cat = Counter()
        output_cat = '' ##our predicted category
        max_prob_cat = -10000000000
        for curr_cat in categories_set:
            curr_prob = math.log10(prob_cat_all[curr_cat])
            for word in line:
                curr_prob = curr_prob + math.log10(dim1[word][curr_cat])
            prob_Counter_per_cat[curr_cat] = curr_prob
            if curr_prob > max_prob_cat:
                max_prob_cat = curr_prob
                output_cat = curr_cat
        #print output_cat
        if presernt_cat == output_cat:
            count = count +1
    return float(count)/len(test_list)

def getTestAccuracyWithConfusionMatrix(test_list, dim1, categories_set, prob_cat_all):
    count = 0
    for line in test_list:
        line = line.split()
        presernt_cat = line[0] ##actually the category
        #print presernt_cat,
        line = line[1:]
        prob_Counter_per_cat = Counter()
        output_cat = '' ##our predicted category
        max_prob_cat = -10000000000
        for curr_cat in categories_set:
            curr_prob = math.log10(prob_cat_all[curr_cat])
            for word in line:
                curr_prob = curr_prob + math.log10(dim1[word][curr_cat])
            prob_Counter_per_cat[curr_cat] = curr_prob
            if curr_prob > max_prob_cat:
                max_prob_cat = curr_prob
                output_cat = curr_cat
        
        #print output_cat
        if presernt_cat == output_cat:
            count = count +1
        ##dict of confusion
        ##gloabl
        
        global confusionDict
        confusionDict[presernt_cat][output_cat] = confusionDict[presernt_cat][output_cat] + 1 
        
    return float(count)/len(test_list)

def printConfusionDict(someDict):
    allKeys = someDict.keys()
    print '\t',
    for key in allKeys:
        print key+' ',
    for key1 in allKeys:
        print key1 + ' '
        for key2 in allKeys:
            print str(someDict[key1][key2])+'\t', 

    
def partA():
    start = time.time()
    all_lines = getAllLines()
    vocab_set, categories_set = getVocabAndCategorySets(all_lines)
    #print len(vocab_set), len(categories_set)
    print categories_set
    train_list_collection, test_list_collection = prepareTrainAndTestSet(all_lines)
    #print len(train_list), len(test_list)
    ##Code for part a
    total_acc = 0.0

    for i in range(len(train_list_collection)):
        print 'Collection ' +str(i)
        category_lenth_Counter, category_word_Counter, categories = getCoreDataStructures1(train_list_collection[i], categories_set)

        prob_cat_all = getProbForEactCategory(categories, categories_set)
        #print prob_cat_all

        dim1 = getCoreDataStructures2(vocab_set, categories_set, category_lenth_Counter, category_word_Counter, categories_set)
        curr_acc = getTestAccuracy(test_list_collection[i], dim1, categories_set, prob_cat_all)
        print curr_acc
        total_acc = total_acc + curr_acc

    end = time.time()
    print str(end-start) + ' seconds'
    print total_acc/float(len(train_list_collection))
    
#Usage: partA()

def partBHelper(test_list_collection, categories_set):
    unique_categories_list = list(categories_set)
    #print unique_categories_list
    total_acc = 0.0
    for i in range(len(test_list_collection)):
        #print 'Collection: '+str(i)
        count = 0
        #print 'Length of Collection '+str(len(test_list_collection[i]))
        for curr_line in test_list_collection[i]:
            actual_cat = curr_line.split()[0]
            our_output_cat = random.choice(unique_categories_list)
            if actual_cat == our_output_cat:
                count = count + 1
        total_acc = total_acc + (float(count)/float(len(test_list_collection[i])))
    total_acc = total_acc/float(len(test_list_collection))
    print 'Randomized: ' + str(total_acc)
    
    
def partB():
    
    start = time.time()
    all_lines = getAllLines()
    vocab_set, categories_set = getVocabAndCategorySets(all_lines)
    #print len(vocab_set), len(categories_set)
    #print categories_set
    train_list_collection, test_list_collection = prepareTrainAndTestSet(all_lines)
    #print len(train_list), len(test_list)
    ##Code for part a
    total_acc = 0.0
    
    partBHelper(test_list_collection, categories_set)
    

    
def partE():
    
    
    
    start = time.time()
    all_lines = getAllLines()
    vocab_set, categories_set = getVocabAndCategorySets(all_lines)
    
    global confusionDict
    confusionDict = {}
    
    for category1 in categories_set:
        confusionDict[category1] = {}
        for category2 in categories_set:
            confusionDict[category1][category2] = 0
    
    #print len(vocab_set), len(categories_set)
    #print categories_set
    train_list_collection, test_list_collection = prepareTrainAndTestSet(all_lines)
    #print len(train_list), len(test_list)
    ##Code for part a
    total_acc = 0.0

    for i in range(len(train_list_collection)):
        print 'Collection ' +str(i)
        category_lenth_Counter, category_word_Counter, categories = getCoreDataStructures1(train_list_collection[i], categories_set)

        prob_cat_all = getProbForEactCategory(categories, categories_set)
        #print prob_cat_all

        dim1 = getCoreDataStructures2(vocab_set, categories_set, category_lenth_Counter, category_word_Counter, categories_set)
        curr_acc = getTestAccuracyWithConfusionMatrix(test_list_collection[i], dim1, categories_set, prob_cat_all)
        print curr_acc
        total_acc = total_acc + curr_acc

    end = time.time()
    print str(end-start) + ' seconds'
    print total_acc/float(len(train_list_collection))
    printConfusionDict(confusionDict)
    
#Usage: partE()

def partDToPlot(testl,trainl,lengthl):
    import matplotlib.pyplot as plt
    plt.plot(lengthl, trainl)
    plt.plot(lengthl, testl)
    plt.legend(['Train','Test'], loc='upper left')
    plt.ylabel('Accuracy')
    plt.xlabel('Number of examples')
    plt.show()
    
def readAndPlotPartD():
    filename = 'partD.txt'
    filename =  open(filename,'r')
    allLines = filename.readlines()
    testAll = allLines[0].split()
    trainAll = allLines[1].split()
    lengthAll = allLines[2].split()
    partDToPlot(testAll, trainAll, lengthAll)
    
# readAndPlotPartD()
def partDHelper(total_len, train_list_collection, test_list_collection, vocab_set, categories_set):
    ##Code for part a
    total_acc = 0.0
    train_acc = 0.0

    for i in range(len(train_list_collection)):
        print 'Collection ' +str(i)
        category_lenth_Counter, category_word_Counter, categories = getCoreDataStructures1(train_list_collection[i][:total_len], categories_set)
    
        prob_cat_all = getProbForEactCategory(categories, categories_set)
        #print prob_cat_all
    
        dim1 = getCoreDataStructures2(vocab_set, categories_set, category_lenth_Counter, category_word_Counter, categories_set)
        curr_acc = getTestAccuracy(test_list_collection[i], dim1, categories_set, prob_cat_all)
        curr_train_acc = getTestAccuracy(train_list_collection[i][:total_len], dim1, categories_set, prob_cat_all) 
        print curr_acc, curr_train_acc
        total_acc = total_acc + curr_acc
        train_acc = train_acc + curr_train_acc
    
    #end = time.time()
    print 'total_len '+str(total_len)
    #print str(end-start) + ' seconds'
    print total_acc/float(len(train_list_collection)),train_acc/float(len(train_list_collection))
    return total_acc/float(len(train_list_collection)),train_acc/float(len(train_list_collection))

def partD():
    
    all_lines = getAllLines()
    vocab_set, categories_set = getVocabAndCategorySets(all_lines)
    #print len(vocab_set), len(categories_set)
    print categories_set
    train_list_collection, test_list_collection = prepareTrainAndTestSet(all_lines)
    
    
    #lengths = [1000,2000]
    lengths = [1000,2000,3000,4000,5000,5784]
    testacc_list = []
    trainacc_list = []
    for length in lengths:
        print '##### Length ######'
        testacc,trainacc = partDHelper(length, train_list_collection, test_list_collection, vocab_set, categories_set)
        testacc_list.append(testacc)
        trainacc_list.append(trainacc)
        
    print testacc_list
    print trainacc_list
    print lengths
    partDToPlot(testacc_list, trainacc_list, lengths)

#Usage: partA(), partB(), etc.
partA()
