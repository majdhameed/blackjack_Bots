import argparse

from ml.training_plots import generate_training_plots

def main(arguments=None):
    parser = argparse.ArgumentParser(description="Plotting")
    parser.add_argument("metrics_path")
    args = parser.parse_args(arguments)

    output_paths = generate_training_plots(args.metrics_path)

    for plot_name, output_path in output_paths.items():
        print(plot_name)
        print(output_path)

    return output_paths

if __name__ == "__main__":
    main()
