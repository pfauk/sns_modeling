"""Testing generation of index sets by stn class for a 4 component mixture matches
with manually generated index sets"""

import pytest
from superstructure.stn import stn


n = 4  # specify the number of components in the feed mixture
network = stn(n)
network.generate_tree()
network.generate_index_sets()

# manually generated index sets
FEED = ['ABCD']
COMP = ['A', 'B', 'C', 'D']
TASKS = ['A/BCD', 'AB/CD', 'ABC/D', 'AB/C', 'A/BC', 'BC/D', 'B/CD', 'A/B', 'B/C', 'C/D']
STATES = ['ABCD', 'ABC', 'BCD', 'AB', 'BC', 'CD', 'A', 'B', 'C', 'D']

FS_f = ['A/BCD', 'AB/CD', 'ABC/D']

TS_s = {'ABCD': ('ABC/D', 'AB/CD', 'A/BCD'),
        'ABC': ('A/BC', 'AB/C'), 'BCD': ('BC/D', 'B/CD'),
        'AB': ('A/B',), 'BC': ('B/C',), 'CD': ('C/D',)}

ST_s = {'ABC': ('ABC/D',), 'BCD': ('A/BCD',),
        'AB': ('AB/CD', 'AB/C',), 'BC': ('A/BC', 'BC/D'), 'CD': ('AB/CD', 'B/CD'),
        'A': ('A/BCD', 'A/BC', 'A/B'), 'B': ('B/CD', 'A/B', 'B/C'),
        'C': ('AB/C', 'B/C', 'C/D'), 'D': ('ABC/D', 'BC/D', 'C/D')}

ISTATE = ['ABC', 'BCD', 'AB', 'BC', 'CD']

PRE_i = {'A': ('A/BCD', 'A/BC', 'A/B'),
         'B': ('B/CD', 'B/C'),
         'C': ('C/D',)}

PST_i = {'B': ('A/B',),
         'C': ('AB/C', 'B/C'),
         'D': ('ABC/D', 'BC/D', 'C/D')}

STRIP_s = {'BCD': ('A/BCD',),
           'BC': ('A/BC',), 'CD': ('AB/CD', 'B/CD'),
           'B': ('A/B',), 'C': ('B/C', 'AB/C'), 'D': ('ABC/D', 'BC/D', 'C/D')}

RECT_s = {'ABC': ('ABC/D',),
          'AB': ('AB/C', 'AB/CD'), 'BC': ('BC/D',),
          'A': ('A/BCD', 'A/BC', 'A/B'), 'B': ('B/CD', 'B/C'), 'C': ('C/D',)}

LK = {'A/BCD': 'A', 'AB/CD': 'B', 'ABC/D': 'C', 'AB/C': 'B', 'A/BC': 'A',
      'BC/D': 'C', 'B/CD': 'B', 'A/B': 'A', 'B/C': 'B', 'C/D': 'C'}

HK = {'A/BCD': 'B', 'AB/CD': 'C', 'ABC/D': 'D', 'AB/C': 'C', 'A/BC': 'B',
      'BC/D': 'D', 'B/CD': 'C', 'A/B': 'B', 'B/C': 'C', 'C/D': 'D'}

IREC_m = {'ABC': ('ABC/D',),
          'AB': ('AB/C', 'AB/CD'), 'BC': ('BC/D',)}

ISTRIP_m = {'BCD': ('A/BCD',),
            'BC': ('A/BC',),
            'CD': ('AB/CD', 'B/CD')}

r = ['r1', 'r2', 'r3']

RUA = {'A/BCD': ('r1', 'r2', 'r3'), 'AB/CD': ('r1', 'r2', 'r3'),
       'ABC/D': ('r1', 'r2', 'r3'),
       'AB/C': ('r1', 'r2'), 'A/BC': ('r1', 'r2'),
       'BC/D': ('r2', 'r3'), 'B/CD': ('r2', 'r3'),
       'A/B': ('r1',), 'B/C': ('r2',), 'C/D': ('r3',)}


def test_lists():
    """Comparison of all index sets that are lists"""
    assert sorted(network.TASKS) == sorted(TASKS)
    assert sorted(network.STATES) == sorted(STATES)
    assert network.FEED == FEED
    assert sorted(network.COMP) == sorted(COMP)
    assert sorted(network.FSf) == sorted(FS_f)
    assert sorted(network.ISTATE) == sorted(ISTATE)
    assert sorted(network.r) == sorted(r)

def test_dictionaries():
    """Comparison of all index sets that are dictionaries.
    values may not be in same order; compare keys and compare values using sets"""
    assert network.TSs.keys() == TS_s.keys()
    for key, values in network.TSs.items():
        assert set(values) == set(TS_s[key])

    assert network.STs.keys() == ST_s.keys()
    for key, values in network.STs.items():
        assert set(values) == set(ST_s[key])

    assert network.PREi.keys() == PRE_i.keys()
    for key, values in network.PREi.items():
        assert set(values) == set(PRE_i[key])

    assert network.PSTi.keys() == PST_i.keys()
    for key, values in network.PSTi.items():
        assert set(values) == set(PST_i[key])

    assert network.STRIPs.keys() == STRIP_s.keys()
    for key, values in network.STRIPs.items():
        assert set(values) == set(STRIP_s[key])

    assert network.RECTs.keys() == RECT_s.keys()
    for key, values in network.RECTs.items():
        assert set(values) == set(RECT_s[key])

    assert network.IRECs.keys() == IREC_m.keys()
    for key, values in network.IRECs.items():
        assert set(values) == set(IREC_m[key])

    assert network.ISTRIPs.keys() == ISTRIP_m.keys()
    for key, values in network.ISTRIPs.items():
        assert set(values) == set(ISTRIP_m[key])

    assert network.LK.keys() == LK.keys()
    assert sorted(network.LK.items()) == sorted(LK.items())

    assert network.HK.keys() == HK.keys()
    assert sorted(network.HK.items()) == sorted(HK.items())

    assert network.RUA.keys() == RUA.keys()
    for key, values in network.RUA.items():
        assert set(values) == set(RUA[key])


if __name__ == "__main__":
    pytest.main()
