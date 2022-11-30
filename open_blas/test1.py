import os

os.system("rm -rf cache*")
os.system("make clean")

os.system("make lstm")
os.system("valgrind --tool=cachegrind ./lstm")
os.system("mv cache* lstm.out")
os.system("cg_annotate --auto=yes --threshold=0.01 lstm.out")

os.system("make bn1d")
os.system("valgrind --tool=cachegrind ./bn1d")
os.system("mv cache* bn1d.out")
os.system("cg_annotate --auto=yes --threshold=0.01 bn1d.out")

