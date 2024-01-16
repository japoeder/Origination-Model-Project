# Lab 
### Introduction

This is my submission for the first lab of Data Structures (605.202).  The module follows the process outlined
here:

1. Read input files supplied by the user
2. Process each input file iteratively, and then within a file iteratively line by line and letter by letter.
3. Each letter is pushed to a stack object to reverse the order.
4. The prefix to postfix conversion process is completed on the reversed string.
5. Output is sent to the provided directory with the prefix 'out_' in the file name.

### Running Lab 1

1. Download and install Python on your computer
2. Navigate to [jpoeder1_lab1](.) directory (containing the README.md)
3. Run the program as a module: `python -m lab1 -h`. This will print the help message.
4. Run the program as a module (with real inputs):
   1. `python -m lab1 -i <some_input_directory> -o <some_output_directory> -p <some_files_prefix>`
   2. `python -m proj0 -i 'resources/input' -o 'resources/output' -p 'myprefix_'`

The command will read in all files in the specified directory with the prefix a user has identified.  Once
the file has been processed, the output will be written to the same directory as the input if not
specified.  Output files will have the prefix 'out_' in either case.

### Lab 1 Usage:

```commandline
usage: python -m proj0 [-h] in_path [out_path] prefix

positional arguments:
  in_path           Input directory path
  [out_path]        Output directory path
  prefix            Prefix for files to be read in

optional arguments:
  -h, --help        show this help message and exit
  -o, --out_path    the output directory
```

*Note that the code is built to handle the '^' character for exponentiation.*

## Python Packaging

This code is packaged as a python module, with the structure outlined in the section below.

### Project Layout

* [jpoeder1_lab1](lab1): The parent or "root" folder containing all of these files. Can technically have any name.
    * [README.md](README.md):
      The guide you're reading.
    * [`__init__.py`](lab1/__init__.py)
      For package / module structure.
    * [lab1](.): This is the *module* in our *package*.
      * [`__init__.py`](lab1/__init__.py)
        Expose what functions, variables, classes, etc when scripts import this module.
      * [`__main__.py`](lab1/__main__.py)
        This file is the entrypoint to your program when ran as a program.
      * `lab1.py`
        Driver program for the module
      * `LinkedStack.py` 
        Stack stack class for module
      * `prefixToPostfix.py` 
        Contains function for prefix to postfix conversion.
    