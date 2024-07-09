# TARGET = test

# SRC_DIR = src 

# SRCS = linear.c conv.c model.c test.c utils.c testcase.c

# OBJS = $(SRCS:.c=.o)

# CC = gcc -msse4.2
# CFLAGS = -Wall -Wextra -std=c11

# all: $(TARGET)
# 	./test
# 	rm -f $(TARGET) *.o

# $(TARGET): $(OBJS)
# 	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS)

# %.o: %.c
# 	$(CC) $(CFLAGS) -c $< -o $@

# clean:
# 	rm -f $(TARGET) $(OBJS)

# .PHONY: all clean

# Variable to specify the compiler
CC = gcc -msse4.2

# Compilation flags
CFLAGS = -Wall -Wextra -I./src

# Directory containing the source files
SRC_DIR = src

# Main source file
MAIN = test.c

# Other source files
SRCS = $(SRC_DIR)/conv.c $(SRC_DIR)/linear.c $(SRC_DIR)/model.c $(SRC_DIR)/utils.c

# Object files generated from the source files
OBJS = $(MAIN:.c=.o) $(SRCS:.c=.o)

# Name of the executable file
TARGET = main

# Default build rule
all: $(TARGET)
	make run
	make clean

# Rule to build the executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

# Rule to compile .c files to .o files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Rule to clean up generated files
clean:
	rm -f $(OBJS) $(TARGET)

# Rule to run the program
run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run