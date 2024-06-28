TARGET = test

SRCS = test.c bnn_model.c fp_model.c

OBJS = $(SRCS:.c=.o)

CC = gcc -msse4.2
CFLAGS = -Wall -Wextra -std=c11

all: $(TARGET)
	./test
	rm -f $(TARGET) $(OBJS)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)

.PHONY: all clean