# This is for the purpose of checking the algorithm is definitely working as expected

def validatePacking(instance, packing, algBins, optBins):
    totalWeight = sum(packing["bin_weights"])
    if totalWeight - sum(instance["weights"]) != 0:
        raise Exception("weight of instance differs to sum of bin weights")
   
    if len(packing["bin_weights"]) != len(packing["packing"]):
        raise Exception("number of bin weights does not match number of bins in packing")

    seen = set()
    for bin_pack in packing["packing"]:
        for idx in bin_pack:
            if idx in seen:
                raise Exception("Item packed twice")
            seen.add(idx)
    if len(seen) != len(instance["weights"]):
        raise Exception("Missing items in packing")
   
    for x in range(0, len(packing["packing"])):
        binPack = packing["packing"][x]
        claimedWeight = packing["bin_weights"][x]

        binWeight = 0
        for index in binPack:
            binWeight += instance["weights"][index]
        if claimedWeight != binWeight:
            raise Exception("weight of a bin does not match the items packed into it")
        if binWeight > instance["bin_capacity"]:
            """print(binWeight)
            for index in binPack:
                print(instance["weights"][index])
            print(instance["bin_capacity"])"""
            raise Exception("bin capacity exceeded")
   
    if algBins < optBins:
        """ print(instance["bin_capacity"])
        print(packing["bin_weights"])
        print(packing["packing"])
        print(instance["optimal_solution"])"""
        raise Exception("algorithm used fewer bins that the optimal")
   
def quickValidatePacking(instance, packing, algBins, optBins):
    totalWeight = sum(packing["bin_weights"])
    if totalWeight - sum(instance["weights"]) != 0:
        raise Exception("weight of instance differs to sum of bin weights")
   
    for weight in packing["bin_weights"]:
        if weight > instance["bin_capacity"]:
            raise Exception("weight of bin exceeds capacity")
   
    if algBins < optBins:
      raise Exception("algorithm used fewer bins than the optimal")
   

def testAlgorithmCorrectness(algorithm, instances):

    for instance in instances:
        timeLimit = 1/600 * len(instance["weights"])
        packing = algorithm(instance["bin_capacity"], instance["weights"], timeLimit)
        
        algBins = len(packing["bin_weights"])
        optBins = instance["optimal_solution"]

        validatePacking(instance, packing, algBins, optBins)

    print("Bin capacity is never exceeded")
    print("Our solutions never do better than the best possible case")
    print("Our solutions have total weight equal to the instance\'s total")
    return True




