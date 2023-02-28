import pstats

p = pstats.Stats('out1.prof')
p.sort_stats('cumulative').print_stats(10)