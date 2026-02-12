# This is for the purpose of checking the algorithm is definitely working as expected

def validatePacking(instance, packing, algBins, optBins):
    totalWeight = sum(packing["bin_weights"])
    if totalWeight - sum(instance["weights"]) != 0:
        raise Exception("Error: weight of instance differs to sum of bin weights")
   
    seen = set()
    for bin_pack in packing["packing"]:
        for idx in bin_pack:
            if idx in seen:
                raise Exception("Item packed twice")
            seen.add(idx)
    if len(seen) != len(instance["weights"]):
        raise Exception("Missing items in packing")
   
    for binPack in packing["packing"]:
        binWeight = 0
        for index in binPack:
            binWeight += instance["weights"][index]
        if binWeight > instance["bin_capacity"]:
            print(binWeight)
            for index in binPack:
                print(instance["weights"][index])
            print(instance["bin_capacity"])
            raise Exception("bin capacity exceeded")
   
    if algBins < optBins:
        """ print(instance["bin_capacity"])
        print(packing["bin_weights"])
        print(packing["packing"])
        print(instance["optimal_solution"])"""
        raise Exception("Error: algorithm used fewer bins that the optimal")
   
def quickValidatePacking(instance, packing, algBins, optBins):
    totalWeight = sum(packing["bin_weights"])
    if totalWeight - sum(instance["weights"]) != 0:
        raise Exception("Error: weight of instance differs to sum of bin weights")
   
    for weight in packing["bin_weights"]:
        if weight > instance["bin_capacity"]:
            raise Exception("Error: weight of bin exceeds capacity")
   
    if algBins < optBins:
      raise Exception("Error: algorithm used fewer bins that the optimal")
   

def testAlgorithmCorrectness(algorithm, instances):

    for instance in instances:
        packing = algorithm(instance["bin_capacity"], instance["weights"])
        
        algBins = len(packing["bin_weights"])
        optBins = instance["optimal_solution"]

        validatePacking(instance, packing, algBins, optBins)

    print("Bin capacity is never exceeded")
    print("Our solutions never do better than the best possible case")
    print("Our solutions have total weight equal to the instance\'s total")
    return True




