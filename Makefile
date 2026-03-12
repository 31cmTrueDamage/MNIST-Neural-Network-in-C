CC      = gcc
CFLAGS  = -Wall -Wextra -O2
LIBS    = -lm
TARGET  = nn
SRCS    = main.c network.c mnist.c

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRCS) $(LIBS)

clean:
	rm -f $(TARGET)