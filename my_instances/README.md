## Uniformly generated instance datasets

Includes several datasets of uniformly generated instances for bin packing.

### our-u-n (uniform random instances)
Each our-u-n set contains 20 instances, each with:
- n items, and
- bin capacity C = 150.

File format
- Line 1: number of items (n)
- Line 2: bin capacity (C)
- Line 3 onward: item weights (one per line)

Generation procedure
For each instance, the generator first selects:
- a minimum weight uniformly at random in \[10%, 30%\] of the bin capacity, and
- a maximum weight uniformly at random in \[50%, 70%\] of the bin capacity.

Item weights are then sampled uniformly at random from the resulting interval.

The generator is implemented in utilities/load_data.py under the function getOurRandomInstances.

---

### test-u
The test-u set is a small collection intended for validation of algorithms before running on the full datasets. It contains 20 instances in total:
- 10 instances with 400 items
- 10 instances with 800 items

These instances are generated using the same procedure as the our-u-n sets, with bin capacity C = 150.
