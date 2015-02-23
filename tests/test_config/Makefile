include ../../Makefile.inc

SOURCES_CPP = parse_input.cpp

# Parse input.ini file
parse_input: parse_input.cpp $(PROJ_INCDIR)/slab_config.h
	$(CC) $(CFLAGS) -o parse_input parse_input.cpp ../../obj/slab_config.o $(INCLUDES) $(LFLAGS)
