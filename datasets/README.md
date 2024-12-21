# Data Format for Oboe

The format of the trace is as follows:

For each pair of lines starting at the beginning of the file

```txt
t1 b1

t2 b1
```

t1 and t2 represent the start time and end time of a chunk (**in milliseconds**), and b1 represents the average bandwidth (**in Kbps**) during the time (t1, t2), and is repeated on both lines.

For two consecutive chunks,

```txt
t1 b1

t2 b1

t3 b2

t4 b2
```

It indicates that the first chunk download starts at t1 and ends at t2, and average bandwidth during the time (t1, t2) is b1. Similary, the second chunk download starts at t3 and ends at t4, and average bandwidth during the time (t3, t4) is b2. The bandwidth during the time (t2, t3) could be estimated using linear interpolation.
