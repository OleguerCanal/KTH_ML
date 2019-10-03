import dtree as dt
import monkdata as m
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


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
        train_means = []
        val_means = []
        test_means = []
        train_stds = []
        val_stds = []
        test_stds = []
        for part in partitions:
            train_accuracy = []
            val_accuracy = []
            test_accuracy = []
            for i in range(number_of_samples):           
                train_subset, val = partition(train, part)
                prunned_tree = get_prunned_tree(train = train_subset, val = val)
                train_accuracy.append(dt.check(prunned_tree, train_subset))
                val_accuracy.append(dt.check(prunned_tree, val))
                test_accuracy.append(dt.check(prunned_tree, test))
            
            # Compute average and variance
            train_means.append(np.mean(train_accuracy))
            train_stds.append(np.std(train_accuracy))
            val_means.append(np.mean(val_accuracy))
            val_stds.append(np.std(val_accuracy))
            test_means.append(np.mean(test_accuracy))
            test_stds.append(np.std(test_accuracy))

        title = "DATASET: " + dataset
        legend = "Samples: " + str(number_of_samples)
        # plt.errorbar(partitions, train_means, train_stds, linestyle='None', marker='^')
        # plt.errorbar(np.array(partitions) + 0.05, val_means, val_stds, linestyle='None', marker='^')
        plt.errorbar(np.array(partitions), test_means, test_stds, linestyle='None', marker='^')
        axes = plt.gca()
        axes.set_ylim([0,1])
        plt.xlabel('Partitions')
        plt.ylabel('Test Error')
        plt.title(title)
        plt.gca().legend((legend,))
        for i, v in enumerate(np.array(test_means, dtype=np.float32)):
            axes.text(partitions[i], v+0.1, "%f" %float(v), ha="center")
        plt.show()
        plt.savefig(title + ".png")
        plt.close()
        dataset = "MONK3"
            

            
            



    

