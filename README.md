# Matsumoto-Imai-Decryption Script

## Description

This Python script is designed to decrypt a ciphertext, which was encrypted by the Matsumoto-Imai cryptosystem, with the given corresponding public key. While some runtime improvements have been implemented, it's important to note that the code might run slow due to the nature of Python and unoptimized functions. It's an amateur project with a focus on educational purposes rather than runtime.

## Usage

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/timobeer73/Matsumoto-Imai-Bruteforce.git
   cd Matsumoto-Imai-Bruteforce
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Script:**
   ```bash
   python main.py inputFile.txt
   ```

   Optionally, use the `-v` or `--verbose` flag for additional information.

## Example Input File

Check out the `exampleChallenge.txt` file for a properly formatted input file.

## Important Notice

This script uses `eval()` for certain calculations. Only run this script with trusted input files to prevent security risks. Ensure that your input files are formatted correctly to avoid unexpected behavior.

## License

This project is licensed under the [MIT License](LICENSE.md).
