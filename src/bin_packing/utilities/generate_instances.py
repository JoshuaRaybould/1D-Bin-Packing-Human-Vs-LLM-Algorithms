import argparse
from . import load_data

def main():
    parser = argparse.ArgumentParser(description="Generate uniform bin packing instances.")
    parser.add_argument("--count", type=int, default=20, help="Number of instances to generate.")
    parser.add_argument("--cap", type=int, default=100, help="Bin capacity.")
    parser.add_argument("--items", type=int, default=100, help="Number of items per instance.")

    args = parser.parse_args()

    instances = load_data.getOurRandomInstances(
        numInstances=args.count,
        capacity=args.cap,
        numItems=args.items,
        distribution="u"
    )

    # number of items determines name of file
    prefix = f"our_u_{args.items}"
    
    out_dir = load_data.saveInstancesAsTxt(instances, prefix)
    print(f"Wrote {len(instances)} instances to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()