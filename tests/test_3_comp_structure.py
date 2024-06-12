"""Testing generation of index sets by stn class for a 3 component mixture matches
with manually generated index sets"""

import pytest
from superstructure.stn import stn

n = 3  # specify the number of components in the feed mixture
network = stn(n)
network.generate_tree()
network.generate_index_sets()

FEED = ['ABC']
COMP = ['A', 'B', 'C']
TASKS = ['A/BC', 'AB/C', 'A/B', 'B/C']
STATES = ['ABC', 'AB', 'BC', 'A', 'B', 'C']

FS_f = ['A/BC', 'AB/C']

TS_s = {'ABC': ('A/BC', 'AB/C'), 'AB': ('A/B',), 'BC': ('B/C',)}

ST_s = {'AB': ('AB/C',), 'BC': ('A/BC',), 'A': ('A/BC', 'A/B'), 'B': ('A/B', 'B/C'), 'C': ('AB/C', 'B/C')}

ISTATE = ['AB', 'BC']

PRE_i = {'A': ('A/BC', 'A/B'), 'B': ('B/C',)}

PST_i = {'B': ('A/B',), 'C': ('AB/C', 'B/C')}

STRIP_s = {'BC': ('A/BC',), 'B': ('A/B',), 'C': ('AB/C', 'B/C')}

RECT_s = {'AB': ('AB/C',), 'A': ('A/BC', 'A/B'), 'B': ('B/C',)}

LK = {'A/BC': 'A', 'AB/C': 'B', 'A/B': 'A', 'B/C': 'B'}

HK = {'A/BC': 'B', 'AB/C': 'C', 'A/B': 'B', 'B/C': 'C'}

IREC_m = {'AB': ('AB/C',)}

ISTRIP_m = {'BC': ('A/BC',)}
r = ['r1', 'r2']

RUA = {'A/BC': ('r1', 'r2'), 'AB/C': ('r1', 'r2'), 'A/B': ('r1',), 'B/C': ('r2',)}

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