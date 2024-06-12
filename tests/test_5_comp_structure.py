"""Testing generation of index sets by stn class for a 5 component mixture matches
with manually generated index sets"""

import pytest
from superstructure.stn import stn

n = 5  # specify the number of components in the feed mixture
network = stn(n)
network.generate_tree()
network.generate_index_sets()

# manually generated index sets
FEED = ['ABCDE']
COMP = ['A', 'B', 'C', 'D', 'E']
TASKS = ['A/BCDE', 'AB/CDE', 'ABC/DE', 'ABCD/E', 'A/BCD', 'AB/CD', 'ABC/D', 'B/CDE', 'BC/DE', 'BCD/E', 'A/BC',
         'AB/C', 'B/CD', 'BC/D', 'C/DE', 'CD/E', 'A/B', 'B/C', 'C/D', 'D/E']

STATES = ['ABCDE', 'ABCD', 'BCDE', 'ABC', 'BCD', 'CDE',
          'AB', 'BC', 'CD', 'DE', 'A', 'B', 'C', 'D', 'E']

FS_f = ['A/BCDE', 'AB/CDE', 'ABC/DE', 'ABCD/E']

TS_s = {'ABCDE': ('A/BCDE', 'AB/CDE', 'ABC/DE', 'ABCD/E'),
        'ABCD': ('A/BCD', 'AB/CD', 'ABC/D'),
        'BCDE': ('B/CDE', 'BC/DE', 'BCD/E'),
        'ABC': ('A/BC', 'AB/C'), 'BCD': ('B/CD', 'BC/D'),
        'CDE': ('C/DE', 'CD/E'),
        'AB': ('A/B',), 'BC': ('B/C',), 'CD': ('C/D',), 'DE': ('D/E',)}

ST_s = {'ABCD': ('ABCD/E',),
        'BCDE': ('A/BCDE',),
        'ABC': ('ABC/DE', 'ABC/D'),
        'BCD': ('A/BCD', 'BCD/E'),
        'CDE': ('AB/CDE', 'B/CDE'),
        'AB': ('AB/CDE', 'AB/CD', 'AB/C'),
        'BC': ('BC/DE', 'A/BC', 'BC/D'),
        'CD': ('AB/CD', 'B/CD', 'CD/E'),
        'DE': ('ABC/DE', 'BC/DE', 'C/DE')}

ISTATE = ['ABCD', 'BCDE', 'ABC', 'BCD', 'CDE', 'AB', 'BC', 'CD', 'DE']

PRE_i = {'A': ('A/BCDE', 'A/BCD', 'A/BC', 'A/B'),
         'B': ('B/CDE', 'B/CD', 'B/C'),
         'C': ('C/DE', 'C/D'),
         'D': ('D/E',)}

PST_i = {'B': ('A/B',),
         'C': ('AB/C', 'B/C'),
         'D': ('ABC/D', 'BC/D', 'C/D'),
         'E': ('ABCD/E', 'BCD/E', 'CD/E', 'D/E')}

STRIP_s = {'BCDE': ('A/BCDE',),
           'BCD': ('A/BCD',),
           'CDE': ('AB/CDE', 'B/CDE',),
           'BC': ('A/BC',),
           'CD': ('AB/CD', 'B/CD'),
           'DE': ('ABC/DE', 'BC/DE', 'C/DE'),
           'B': ('A/B',),
           'C': ('AB/C', 'B/C'),
           'D': ('ABC/D', 'BC/D', 'C/D'),
           'E': ('ABCD/E', 'BCD/E', 'CD/E', 'D/E')}

RECT_s = {'ABCD': ('ABCD/E',),
          'ABC': ('ABC/DE', 'ABC/D'),
          'BCD': ('BCD/E',),
          'AB': ('AB/CDE', 'AB/CD', 'AB/C'),
          'BC': ('BC/DE', 'BC/D'), 'CD': ('CD/E',),
          'A': ('A/BCDE', 'A/BCD', 'A/BC', 'A/B'),
          'B': ('B/CDE', 'B/CD', 'B/C'),
          'C': ('C/DE', 'C/D'),
          'D': ('D/E',)}

LK = {'A/BCDE': 'A', 'AB/CDE': 'B', 'ABC/DE': 'C', 'ABCD/E': 'D', 'A/BCD': 'A',
      'AB/CD': 'B', 'ABC/D': 'C', 'B/CDE': 'B', 'BC/DE': 'C', 'BCD/E': 'D',
      'A/BC': 'A', 'AB/C': 'B', 'B/CD': 'B', 'BC/D': 'C',
      'C/DE': 'C', 'CD/E': 'D', 'A/B': 'A', 'B/C': 'B', 'C/D': 'C', 'D/E': 'D'}

HK = {'A/BCDE': 'B', 'AB/CDE': 'C', 'ABC/DE': 'D', 'ABCD/E': 'E',
      'A/BCD': 'B', 'AB/CD': 'C', 'ABC/D': 'D', 'B/CDE': 'C', 'BC/DE': 'D', 'BCD/E': 'E',
      'A/BC': 'B', 'AB/C': 'C', 'B/CD': 'C', 'BC/D': 'D',
      'C/DE': 'D', 'CD/E': 'E', 'A/B': 'B', 'B/C': 'C', 'C/D': 'D', 'D/E': 'E'}

IREC_m = {'ABCD': ('ABCD/E',),
          'ABC': ('ABC/DE', 'ABC/D'),
          'BCD': ('BCD/E',),
          'AB': ('AB/CDE', 'AB/CD', 'AB/C'),
          'BC': ('BC/DE', 'BC/D'), 'CD': ('CD/E',)}

ISTRIP_m = {'BCDE': ('A/BCDE',),
            'BCD': ('A/BCD',),
            'CDE': ('AB/CDE', 'B/CDE',),
            'BC': ('A/BC',),
            'CD': ('AB/CD', 'B/CD'),
            'DE': ('ABC/DE', 'BC/DE', 'C/DE')}

r = ['r1', 'r2', 'r3', 'r4']

RUA = {'ABCDE': ('r1', 'r2', 'r3', 'r4'), 'ABCD': ('r1', 'r2', 'r3'),
       'BCDE': ('r2', 'r3', 'r4'), 'ABC': ('r1', 'r2'), 'BCD': ('r2', 'r3'),
       'CDE': ('r3', 'r4'), 'AB': ('r1',), 'BC': ('r2',), 'CD': ('r3',), 'DE': ('r4',)}

RUA = {'A/BCDE': ('r1', 'r2', 'r3', 'r4'), 'AB/CDE': ('r1', 'r2', 'r3', 'r4'), 'ABC/DE': ('r1', 'r2', 'r3', 'r4'),
       'ABCD/E': ('r1', 'r2', 'r3', 'r4'), 'A/BCD': ('r1', 'r2', 'r3'), 'AB/CD': ('r1', 'r2', 'r3'),
       'ABC/D': ('r1', 'r2', 'r3'), 'B/CDE': ('r2', 'r3', 'r4'), 'BC/DE': ('r2', 'r3', 'r4'),
       'BCD/E': ('r2', 'r3', 'r4'), 'A/BC': ('r1', 'r2'),
       'AB/C': ('r1', 'r2'), 'B/CD': ('r2', 'r3'), 'BC/D': ('r2', 'r3'),
       'C/DE': ('r3', 'r4'), 'CD/E': ('r3', 'r4'), 'A/B': ('r1',), 'B/C': ('r2',), 'C/D': ('r3',), 'D/E': ('r4',)}

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

    # assert network.STs.keys() == ST_s.keys()
    # for key, values in network.STs.items():
    #     assert set(values) == set(ST_s[key])

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
