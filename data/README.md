# Benchmark instances

The 10 Brandimarte instances MK01-MK10 and the 21 Barnes instances (mt10*, setb4*, seti5*) were taken from the
public collection https://github.com/SchedulingLab/fjsp-instances (MIT License), retrieved in October 2026.
`SHA256SUMS` in each folder lists the checksum of every file. `barnes/BOUNDS.txt` gives, per Barnes instance, the
number of jobs, the number of machines and the reference makespan; the Brandimarte reference values are in
`analyze.py`. The reference values come from Behnke and Geiger (2012) and Schutt et al. (2013).

Original sources:
- P. Brandimarte, Routing and scheduling in a flexible job shop by tabu search, Annals of Operations Research
  41(3), 157-183, 1993, https://doi.org/10.1007/BF02023073
- J. W. Barnes and J. B. Chambers, Flexible job shop scheduling by tabu search, Technical Report ORP96-09,
  The University of Texas at Austin, 1996

Format: first line `<jobs> <machines>`; then one line per job with the number of operations and, for each
operation, the number of eligible machines followed by `<machine> <processing time>` pairs, machines from 0.
