""""
"""

from stn import stn
from stn_nonconsecutive import stn_nonconsecutive

# examples of superstructure generation for consecutive splits 
# n = 3  # specify the number of components in the feed mixture
# network = stn(n)
# network.generate_tree()

# print()
# print(f'{n} Component separation network')

# network.print_tree()
# network.generate_index_sets()
# # network.print_sets()
# network.display_tree()

# examples of superstructure generation for non-consecutive splits
m = 4  # specify the number of components in the feed mixture
network = stn_nonconsecutive(m)
network.generate_tree()

print()
print(f'{m} Component separation network')

network.print_tree()
network.generate_index_sets()
# network.print_sets()
network.display_tree()