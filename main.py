from processing.xor import run_xor
from processing.regression import run_regression

def main():
    run_xor()
    run_regression()

    print()
    print("All tasks completed.")
    print("Check the 'output/' directory for generated figures.")

if __name__ == "__main__":
    main()