"""Resolve exact zero-beta aliases, preserving the ordinary-threshold control."""
from research_singleton_free_20261108.common import *
FULL=RUN/'full'


def source_arm(p,arm):
    visited=set()
    while arm in p['aliases']:
        assert arm not in visited;visited.add(arm);arm=p['aliases'][arm]
    return arm


def raw_path(p,arm,video):
    arm=source_arm(p,arm)
    return ROOT/p['videos'][video]['raw_array'] if arm=='B0' else FULL/'rollouts'/arm/(video+'.npz')


def output_path(arm,video):return FULL/'outputs'/arm/(video+'.json')
