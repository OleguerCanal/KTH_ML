import dtree as dt
import monkdata as m
import drawtree_qt5 as draw

if __name__ == "__main__":
    # ASSIGNMENT 0
    print("MONK 1:", dt.entropy(m.monk1))
    print("MONK 2:", dt.entropy(m.monk2))
    print("MONK 3:", dt.entropy(m.monk3))
    print("")

    # ASSIGNMENT 3
    monks = [m.monk1, m.monk2, m.monk3]
    for monk_id, monk in enumerate(monks):
        print("Monk:", monk_id+1)
        for i in range(6):
            gain = dt.averageGain(monk, m.attributes[i])
            print("A" + str(i+1) + ": " + str(gain))
    print("")

    # ASSIGNMENT 4
    for monk_id, monk in enumerate(monks):
        print("Monk:", monk_id+1)
        best_atribute = dt.bestAttribute(monk, m.attributes)
        print("Best attribute: " + str(best_atribute))
        for value in best_atribute.values:
            subset = dt.select(monk, best_atribute, value)
            entropy = dt.entropy(subset)
            print("Entropy " + str(value) + ": " + str(entropy))
            # print("Next level information gains:")
            # for i in range(6):
            #     gain = dt.averageGain(monk, m.attributes[i])
            #     print("A" + str(i+1) + ": " + str(gain))
    print("")


    best_atribute = dt.bestAttribute(m.monk1, m.attributes)
    for value in best_atribute.values:
        subset = dt.select(m.monk1, best_atribute, value)
        entropy = dt.entropy(subset)
        print("Attribute value:" + str(value))
        for i in range(6):
            gain = dt.averageGain(subset, m.attributes[i])
            print("A" + str(i+1) + ": " + str(gain))
    print("")

    # Assignment 5
    best_atribute = dt.bestAttribute(m.monk1, m.attributes)
    for value in best_atribute.values:
        subset = dt.select(m.monk1, best_atribute, value)
        best_atribute2 = dt.bestAttribute(subset, m.attributes)
        print(str(best_atribute) + " = " + str(value))
        for value2 in best_atribute2.values:
            subset2 = dt.select(subset, best_atribute2, value2)
            common = dt.mostCommon(subset2)
            print("  " + str(best_atribute2) + "=" + str(value2) + ": " + str(common))
    
    tree = dt.buildTree(monk1, m.attributes, 2)
    draw.drawTree(tree)