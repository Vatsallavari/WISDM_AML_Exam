input_file = "dataset\WISDM_ar_v1.1_raw.txt"
output_file = "dataset\WISDM_raw.csv"

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        clean_line = line.strip().rstrip(";")  # remove trailing semicolon
        outfile.write(clean_line + "\n")  # write as comma-separated
