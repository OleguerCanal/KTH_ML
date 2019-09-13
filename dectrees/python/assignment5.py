import dtree as dt
import monkdata as m
# import drawtree_qt5 as draw

if __name__ == "__main__":
    monks = [m.monk1, m.monk2, m.monk3]
    tests = [m.monk1test, m.monk2test, m.monk3test]
    for monk_id, monk in enumerate(monks):
        print("Monk:" + str(monk_id + 1))
        tree = dt.buildTree(monk, m.attributes)
        print("Train E: " + str(dt.check(tree, monk)))
        print("Test E: " + str(dt.check(tree, tests[monk_id])))