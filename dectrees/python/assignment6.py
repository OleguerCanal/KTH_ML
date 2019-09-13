import dtree as dt
import monkdata as m
import random
import numpy as np
import matplotlib.pyplot as plt

def partition(data, fraction):
    ldata = list(data)
    random.seed = 1
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

def get_prunned_tree(train, val):
    tree = dt.buildTree(train, m.attributes)
    initial_val_accuracy = dt.check(tree, val)
    # print("InitiVal E: " + str(initial_val_accuracy))

    iterations = 0
    val_accuracy = initial_val_accuracy
    while (val_accuracy >= initial_val_accuracy):
        iterations += 1
        # print("Iteration: " + str(iterations))

        pruned_trees = dt.allPruned(tree)
        val_accuracy = -1
        for pruned_tree in pruned_trees:
            current_val = dt.check(pruned_tree, val)
            if current_val >= val_accuracy:
                val_accuracy = current_val
                tree = pruned_tree
        # print(val_accuracy)
    return tree



if __name__ == "__main__":
    trains = [m.monk1, m.monk3]
    tests = [m.monk1test, m.monk3test]
    partitions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    number_of_samples = 100

    x = range(number_of_samples)
    dataset = "MONK1"
    for train, test in zip(trains, tests):
        for part in partitions:
            accuracy = []
            for i in range(number_of_samples):           
                train2, val = partition(train, part)
                # print(len(train2))
                # print(len(val))
                prunned_tree = get_prunned_tree(train = train2, val = val)
                accuracy.append(dt.check(prunned_tree, test))
            
            # Compute average and variance
            average = sum(accuracy)/len(accuracy)
            variance = 0
            for val in accuracy:
                variance += (average - val) ** 2

            title = "DATASET: " + dataset + ", PARTITION: " + str(part) 
            legend = "Mean: " + str(average) + "\n" + "Variance: " + str(variance) + "\n" + "Samples: " + str(number_of_samples)
            plt.boxplot(accuracy)
            axes = plt.gca()
            # axes.set_xlim([0,1])
            axes.set_ylim([0,1])
            plt.title(title)
            plt.gca().legend((legend,))
            # plt.show()
            plt.savefig(title + ".png")
            plt.close()
        dataset = "MONK3"
            

            
            



    

