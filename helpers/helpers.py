import os
import datetime

def wOut(message, filename="output.out"):
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mode = "a" if os.path.exists(filename) else "w"

    with open(filename, mode) as file:
        file.write(f'{timestamp} - {message}\n')

def initialize_output_file(filename="output.out"):
    if os.path.exists("output.out"):
        os.remove("output.out")
    with open(filename, "a") as file:
        file.write("""
.------..------..------..------..------..------.
|O.--. ||U.--. ||T.--. ||P.--. ||U.--. ||T.--. |
| :/\: || (\/) || :/\: || :/\: || (\/) || :/\: |
| :\/: || :\/: || (__) || (__) || :\/: || (__) |
| '--'O|| '--'U|| '--'T|| '--'P|| '--'U|| '--'T|
`------'`------'`------'`------'`------'`------'
                 \n""")
    