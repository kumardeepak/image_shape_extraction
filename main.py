import sys, getopt
from src.process import detect_tables_and_lines, detect_tables_and_lines_v1

def main(argv):
    inputfile = ''
    try:
        if len(argv) < 2:
            print ('main.py -i <inputfile>')
            sys.exit()
        
        opts, args = getopt.getopt(argv,"hi:",["ifile="])
    except getopt.GetoptError:
        print ('main.py -i <inputfile>')
        sys.exit(2)
    
    for opt, arg in opts:
        if opt == '-h':
            print ('main.py -i <inputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
    print('received inputfile [%s]' % (inputfile))

    tables, lines = detect_tables_and_lines_v1(inputfile)
    
    print(tables)
    print(lines)
    print('no. of tables: {%d}, no. of lines: {%d}' % (len(tables), len(lines)))


if __name__ == "__main__":
    main(sys.argv[1:])