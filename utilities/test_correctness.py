# This is for the purpose of checking the algorithm is definitely working as expected

def testAlgorithmCorrectness(algorithm, instances):

    for instance in instances:
        packing = algorithm(instance["bin_capacity"], instance["weights"])

        # First check we don't exceed bin_capacity
        for binWeight in packing["bin_weights"]:
            if binWeight > instance["bin_capacity"]:
                raise Exception("Error: capacity exceeded")

        # Check if the sum of weights is correct
        packingTotalWeight = sum(packing["bin_weights"])
        instanceTotalWeight = sum(instance["weights"])
        if packingTotalWeight != instanceTotalWeight:
            if packingTotalWeight < instanceTotalWeight:
                raise Exception("Error: the total weight is less than expected")
            else:
                raise Exception("Error: the total weight is greater than expected")

        # Finally check we haven't done better than the optimal case
        if len(packing["bin_weights"]) < instance["optimal_solution"]:
            raise Exception("Error: negative waste, our algorithm is cheating")

    print("Bin capacity is never exceeded")
    print("Our solutions never do better than the best possible case")
    print("Our solutions have total weight equal to the instance\'s total")
    return True




